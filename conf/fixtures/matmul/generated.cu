/*
 * High-performance BF16 matrix multiplication kernel (MMA tensor core).
 *
 * Computes C = A @ B where A is M×K row-major, B is K×N row-major,
 * C is M×N row-major.  All pointers are __nv_bfloat16; MMA accumulates
 * in FP32, results are converted back to BF16 on store.
 *
 * B(K×N) is pre-transposed to B_t(N×K) in kernel_run() so that B loading
 * uses the same contiguous-along-K cp.async path as A.
 *
 * kernel_run contract:
 *   extern "C" int kernel_run(__nv_bfloat16** inputs, int num_inputs,
 *                             __nv_bfloat16** outputs, int num_outputs,
 *                             int n, cudaStream_t stream);
 */
#include <mma.h>
#include <cuda_bf16.h>

#define N_STAGES 3

/* -------------------------------------------------------------------------
 * PTX helper: cp.async 16-byte global→shared copy
 * ------------------------------------------------------------------------- */
__forceinline__
__device__ void cp_async(uint4 *dstAddr, const uint4 *srcAddr) {
    unsigned ptxDstAddr = __cvta_generic_to_shared(dstAddr);
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
        :: "r"(ptxDstAddr),
        "l"(srcAddr),
        "n"(16));
}

/* -------------------------------------------------------------------------
 * PTX helper: ldmatrix.sync.aligned.x4 — load 4 registers from SMEM
 * ------------------------------------------------------------------------- */
__forceinline__
__device__ void load_matrix_x4(unsigned *destReg, uint4 *srcAddr) {
    unsigned ptxSrcAddr = __cvta_generic_to_shared(srcAddr);
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(destReg[0]), "=r"(destReg[1]), "=r"(destReg[2]), "=r"(destReg[3])
        :  "r"(ptxSrcAddr)
    );
}

/* -------------------------------------------------------------------------
 * PTX helper: ldmatrix.sync.aligned.x2 — load 2 registers from SMEM
 * ------------------------------------------------------------------------- */
__forceinline__
__device__ void load_matrix_x2(unsigned *destReg, uint4 *srcAddr) {
    unsigned ptxSrcAddr = __cvta_generic_to_shared(srcAddr);
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(destReg[0]), "=r"(destReg[1])
        :  "r"(ptxSrcAddr)
    );
}

/* -------------------------------------------------------------------------
 * PTX helper: mma.sync.aligned.m16n8k16 — BF16 inputs, FP32 accumulate
 * ------------------------------------------------------------------------- */
__forceinline__
__device__ void mma_m16n8k16(const unsigned *A, const unsigned *B, float *C, float *D) {
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        :
        "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
    );
}

/* -------------------------------------------------------------------------
 * Transpose kernel: B(K, N) row-major → B_t(N, K) row-major
 *
 * Uses 32×32 tiled transpose with +1 SMEM padding to avoid bank conflicts.
 * Block dim: (32, 8).  Each thread handles 4 rows of the 32-row tile.
 * Grid dim: (ceil(N/32), ceil(K/32)).
 * ------------------------------------------------------------------------- */
__global__ void transpose_bf16(const __nv_bfloat16 *__restrict__ src,
                                __nv_bfloat16 *__restrict__ dst,
                                int K, int N) {
    /* +1 padding per row to avoid shared memory bank conflicts */
    __shared__ __nv_bfloat16 tile[32][33];

    int bx = blockIdx.x * 32;  /* N tile start */
    int by = blockIdx.y * 32;  /* K tile start */

    int tx = threadIdx.x;      /* 0..31 */
    int ty = threadIdx.y;      /* 0..7  */

    /* Load: each thread loads 4 elements along K (stride 8 in ty) */
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int k = by + ty * 4 + i;
        int n = bx + tx;
        if (k < K && n < N)
            tile[ty * 4 + i][tx] = src[k * N + n];
        else if (k < 32 || n < 32)  /* keep tile valid for out-of-bounds */
            tile[ty * 4 + i][tx] = __float2bfloat16(0.f);
    }

    __syncthreads();

    /* Store: transposed — each thread writes 4 elements along N (stride 8 in ty) */
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int n = bx + ty * 4 + i;
        int k = by + tx;
        if (n < N && k < K)
            dst[n * K + k] = tile[tx][ty * 4 + i];
    }
}

/* -------------------------------------------------------------------------
 * Main GEMM kernel: 128×128 block, 4×4 MMA tiling, 3-stage pipeline
 *
 * A   is M×K row-major (__nv_bfloat16)  — loaded via cp.async
 * B   is N×K row-major (__nv_bfloat16)  — pre-transposed, loaded via cp.async
 * C   is M×N row-major (__nv_bfloat16)  — output in BF16
 * ------------------------------------------------------------------------- */
__launch_bounds__(16 * 16)
__global__ void mma_matmul_bf16(const __nv_bfloat16 *A, const __nv_bfloat16 *B,
                                __nv_bfloat16 *C, int M, int N, int K) {
    __shared__ uint4 As[N_STAGES * 64][8];
    __shared__ uint4 Bs[N_STAGES * 64][8];

    uint4 (*aLoadPtr)[8];
    uint4 (*bLoadPtr)[8];
    uint4 (*aStorePtr)[8];
    uint4 (*bStorePtr)[8];

    int blockRowStart = blockIdx.y * 128;
    int blockColStart = blockIdx.x * 128;
    const uint4 *globalTileA = reinterpret_cast<const uint4 *>(A + blockRowStart * K);
    const uint4 *globalTileB = reinterpret_cast<const uint4 *>(B + blockColStart * K);

    int threadID = threadIdx.y * blockDim.x + threadIdx.x;
    int warpID   = threadID / 32;
    int laneID   = threadID % 32;
    int warpOffsetA = 32 * (warpID / 4);
    int warpOffsetB = 16 * (warpID % 4);

    unsigned aReg[4][8];
    unsigned bReg[4][4];
    float dReg[4][4][4] = {0.f};

    int storeRow = warpID * 4 + laneID / 8;
    int storeCol = (laneID % 8) ^ (laneID / 8);

    int loadRowA = (laneID % 16) / 2;
    int loadColA = (laneID / 16 + 4 * (laneID % 2)) ^ (loadRowA % 4);
    int loadRowB = (laneID % 8) / 2;
    int loadColB = (laneID / 8 + 4 * (laneID % 2)) ^ (loadRowB % 4);

    int K8 = K / 8;

    const uint4 *aGlobalAddress = globalTileA + (warpID * 8 + laneID / 4) * K8 + laneID % 4;
    const uint4 *bGlobalAddress = globalTileB + (warpID * 8 + laneID / 4) * K8 + laneID % 4;

    int kStartMax = (K / 32 - 1) * 4;
    int numKBlocks = K / 32;

    /* ---- PRELUDE: load first (N_STAGES - 1) tiles ---- */
    for (int nStage = 0; nStage < N_STAGES - 1; nStage++) {
        int kStart = nStage * 4;
        aStorePtr = As + 64 * nStage;
        bStorePtr = Bs + 64 * nStage;

        /* A: cp.async (contiguous along K) */
        cp_async(aStorePtr[storeRow     ] + storeCol, aGlobalAddress + kStart);
        cp_async(aStorePtr[storeRow + 32] + storeCol, aGlobalAddress + 64 * K8 + kStart);

        /* B: cp.async (contiguous along K — B is now N×K row-major) */
        cp_async(bStorePtr[storeRow     ] + storeCol, bGlobalAddress + kStart);
        cp_async(bStorePtr[storeRow + 32] + storeCol, bGlobalAddress + 64 * K8 + kStart);

        asm volatile("cp.async.commit_group;\n" ::);
    }

    /* ---- MAIN LOOP OVER K BLOCKS ---- */
    for (int nStage = 0; nStage < numKBlocks; nStage++) {
        int kStart = (N_STAGES - 1 + nStage) * 4;
        aStorePtr = As + 64 * ((nStage + N_STAGES - 1) % N_STAGES);
        bStorePtr = Bs + 64 * ((nStage + N_STAGES - 1) % N_STAGES);
        aLoadPtr  = As + 64 * (nStage % N_STAGES);
        bLoadPtr  = Bs + 64 * (nStage % N_STAGES);

        asm volatile("cp.async.wait_group %0;\n" :: "n"(N_STAGES - 2));
        __syncthreads();

        /* Load A fragments: 4 tiles × 2 k-halves */
        for (int m = 0; m < 4; m++) {
            load_matrix_x4(aReg[m]    , aLoadPtr[m * 8 + warpOffsetA + loadRowA] + loadColA);
            load_matrix_x4(aReg[m] + 4, aLoadPtr[m * 8 + warpOffsetA + loadRowA] + (loadColA ^ 2));
        }
        /* Load B fragments: 4 tiles × 2 k-halves */
        for (int n = 0; n < 4; n++) {
            load_matrix_x2(bReg[n]    , bLoadPtr[n * 4 + warpOffsetB + loadRowB] + loadColB);
            load_matrix_x2(bReg[n] + 2, bLoadPtr[n * 4 + warpOffsetB + loadRowB] + (loadColB ^ 2));
        }

        /* Prefetch next A and B tiles via cp.async (single commit group) */
        kStart = (kStart > kStartMax) ? kStartMax : kStart;
        cp_async(aStorePtr[storeRow     ] + storeCol, aGlobalAddress + kStart);
        cp_async(aStorePtr[storeRow + 32] + storeCol, aGlobalAddress + 64 * K8 + kStart);
        cp_async(bStorePtr[storeRow     ] + storeCol, bGlobalAddress + kStart);
        cp_async(bStorePtr[storeRow + 32] + storeCol, bGlobalAddress + 64 * K8 + kStart);
        asm volatile("cp.async.commit_group;\n" ::);

        /* Compute the 4×4 MMA tiles */
        for (int m = 0; m < 4; m++) {
            for (int n = 0; n < 4; n++) {
                mma_m16n8k16(aReg[m]    , bReg[n]    , dReg[m][n], dReg[m][n]);
                mma_m16n8k16(aReg[m] + 4, bReg[n] + 2, dReg[m][n], dReg[m][n]);
            }
        }
    }

    /* ---- EPILOGUE: FP32 → BF16 store ---- */
    int groupID     = laneID >> 2;
    int groupLaneID = laneID & 3;
    for (int m = 0; m < 4; m++) {
        for (int n = 0; n < 4; n++) {
            int row0 = blockRowStart + m * 16 + 2 * warpOffsetA + groupID;
            int col0 = blockColStart + n * 8  + 2 * warpOffsetB + 2 * groupLaneID;
            __nv_bfloat162 packed0 = __floats2bfloat162_rn(dReg[m][n][0], dReg[m][n][1]);
            *reinterpret_cast<__nv_bfloat162 *>(&C[row0 * N + col0]) = packed0;

            int row1 = row0 + 8;
            __nv_bfloat162 packed1 = __floats2bfloat162_rn(dReg[m][n][2], dReg[m][n][3]);
            *reinterpret_cast<__nv_bfloat162 *>(&C[row1 * N + col0]) = packed1;
        }
    }
}

/* -------------------------------------------------------------------------
 * kernel_run — entry point.
 *
 * Allocates a static B_t(N×K) buffer on first call, launches the transpose
 * kernel to convert B(K×N) → B_t(N×K), then runs the GEMM.
 * ------------------------------------------------------------------------- */
static int s_M = 0, s_N = 0, s_K = 0;
static __nv_bfloat16 *s_Bt = nullptr;

static void ensure_shape(int n) {
    if (s_M > 0) return;
    const char *shape = getenv("CUDA_EXEC_PARAM_SHAPE");
    if (shape) {
        int d0 = 0, d1 = 0;
        const char *p = shape;
        while (*p && *p != '[') ++p;
        if (*p == '[') ++p;
        while (*p == ' ') ++p;
        while (*p >= '0' && *p <= '9') { d0 = d0 * 10 + (*p - '0'); ++p; }
        while (*p == ' ' || *p == ',') ++p;
        while (*p >= '0' && *p <= '9') { d1 = d1 * 10 + (*p - '0'); ++p; }
        s_M = d0; s_N = d1; s_K = d0;
    } else {
        s_M = (int)sqrtf((float)n);
        s_N = s_M; s_K = s_M;
    }
}

extern "C" int kernel_run(__nv_bfloat16 **inputs,  int num_inputs,
                          __nv_bfloat16 **outputs, int num_outputs,
                          int n, cudaStream_t stream) {
    const __nv_bfloat16 *A = inputs[0];
    const __nv_bfloat16 *B = inputs[1];
    __nv_bfloat16       *C = outputs[0];

    ensure_shape(n);
    int M = s_M, N = s_N, K = s_K;

    /* Allocate B_t buffer once (N×K, same element count as B) */
    if (s_Bt == nullptr)
        cudaMalloc(&s_Bt, (size_t)N * K * sizeof(__nv_bfloat16));

    /* Transpose B(K×N) → B_t(N×K) */
    dim3 transposeBlock(32, 8);
    dim3 transposeGrid((N + 31) / 32, (K + 31) / 32);
    transpose_bf16<<<transposeGrid, transposeBlock, 0, stream>>>(B, s_Bt, K, N);

    /* GEMM over A(M×K) and B_t(N×K) */
    dim3 threads(16, 16);
    dim3 grid(N / 128, M / 128);
    mma_matmul_bf16<<<grid, threads, 0, stream>>>(A, s_Bt, C, M, N, K);

    return 0;
}
