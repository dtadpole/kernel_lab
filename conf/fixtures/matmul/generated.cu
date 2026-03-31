/*
 * High-performance BF16 matrix multiplication kernel (MMA tensor core).
 * K-tile=64 + ldmatrix.trans: doubled K-tile, no transpose kernel.
 *
 * Computes C = A @ B where A is M×K row-major, B is K×N row-major,
 * C is M×N row-major.  All pointers are __nv_bfloat16; MMA accumulates
 * in FP32, results are converted back to BF16 on store.
 *
 * B(K×N) is loaded directly along N (contiguous) via cp.async into SMEM.
 * ldmatrix.trans transposes the 8×8 BF16 tiles during SMEM-to-register
 * load, producing K-contiguous register data for the MMA .col operand.
 * No separate transpose kernel is needed.
 *
 * kernel_run contract:
 *   extern "C" int kernel_run(__nv_bfloat16** inputs, int num_inputs,
 *                             __nv_bfloat16** outputs, int num_outputs,
 *                             int n, cudaStream_t stream);
 */
#include <mma.h>
#include <cuda_bf16.h>

#define N_STAGES 3
#define SMEM_ROWS 128   /* SMEM rows per matrix per stage (K-tile=64) */

/* Total dynamic SMEM: 2 matrices × N_STAGES × 128 rows × 8 cols × 16 bytes */
static const size_t GEMM_SMEM_BYTES = 2u * N_STAGES * SMEM_ROWS * 8 * sizeof(uint4);


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
 * PTX helper: ldmatrix.sync.aligned.x2.trans — load 2 regs transposed
 * ------------------------------------------------------------------------- */
__forceinline__
__device__ void load_matrix_x2_trans(unsigned *destReg, uint4 *srcAddr) {
    unsigned ptxSrcAddr = __cvta_generic_to_shared(srcAddr);
    asm volatile(
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
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
 * Main GEMM kernel: 128×128 block, 4×4 MMA tiling, K-tile=64, 3-stage pipe
 *
 * A   is M×K row-major (__nv_bfloat16)  — loaded via cp.async, ldmatrix.x4
 * B   is K×N row-major (__nv_bfloat16)  — loaded via cp.async along N,
 *                                          ldmatrix.x2.trans transposes S2R
 * C   is M×N row-major (__nv_bfloat16)  — output in BF16
 *
 * Uses dynamic shared memory (96 KB) to exceed the 48 KB static limit.
 * ------------------------------------------------------------------------- */
__launch_bounds__(16 * 16)
__global__ void mma_matmul_bf16(const __nv_bfloat16 *A, const __nv_bfloat16 *B,
                                __nv_bfloat16 *C, int M, int N, int K) {
    /* Dynamic shared memory: As followed by Bs */
    extern __shared__ char smem_buf[];
    uint4 (*As)[8] = reinterpret_cast<uint4(*)[8]>(smem_buf);
    uint4 (*Bs)[8] = reinterpret_cast<uint4(*)[8]>(
        smem_buf + N_STAGES * SMEM_ROWS * 8 * sizeof(uint4));

    uint4 (*aLoadPtr)[8];
    uint4 (*bLoadPtr)[8];
    uint4 (*aStorePtr)[8];
    uint4 (*bStorePtr)[8];

    int blockRowStart = blockIdx.y * 128;
    int blockColStart = blockIdx.x * 128;
    const uint4 *globalTileA = reinterpret_cast<const uint4 *>(A + blockRowStart * K);

    int threadID = threadIdx.y * blockDim.x + threadIdx.x;
    int warpID   = threadID / 32;
    int laneID   = threadID % 32;
    int warpOffsetA = 32 * (warpID / 4);
    int warpOffsetB = 16 * (warpID % 4);

    unsigned aReg[4][8];
    unsigned bReg[4][4];
    float dReg[4][4][4] = {0.f};

    /* A store indices (K-contiguous in SMEM) */
    int storeRow = warpID * 4 + laneID / 8;
    int storeCol = (laneID % 8) ^ (laneID / 8);

    /* B store indices (N-contiguous in SMEM, K-rows × N-chunks layout) */
    int bStoreRow = warpID * 4 + laneID / 8;
    int bStoreCol = (laneID % 8) ^ ((warpID * 2 + laneID / 16) & 7);

    /* A load indices (ldmatrix.x4, non-transposed) */
    int loadRowA = (laneID % 16) / 2;
    int loadColA = (laneID / 16 + 4 * (laneID % 2)) ^ (loadRowA % 4);

    /* B load indices (ldmatrix.x2.trans — rows are K, cols are N-chunks) */
    int loadRowB = (laneID % 16) * 2 + (warpID % 4) / 2;
    int loadColB_base = (((warpID % 4) & 1) * 4) ^ (laneID & 7);

    int K8 = K / 8;
    int N8 = N / 8;

    /* A: row-major M×K, load K-contiguous uint4 */
    const uint4 *aGlobalAddress = globalTileA + (warpID * 8 + laneID / 4) * K8 + laneID % 4;

    /* B: row-major K×N, load N-contiguous uint4.
     * Thread mapping: k_local = warpID*2 + laneID/16 (0..15),
     *                 n_local = laneID%16 (0..15).
     * Per stage: 4 cp.async calls cover 64 K-rows (4 × 16 K-rows). */
    const uint4 *globalB = reinterpret_cast<const uint4 *>(B);
    const uint4 *bGlobalBase = globalB
        + (warpID * 2 + laneID / 16) * N8
        + blockColStart / 8 + laneID % 16;

    int numKBlocks = K / 64;
    int kStartMax  = (numKBlocks - 1) * 8;   /* max kStart in uint4 units */

    /* ---- PRELUDE: load first (N_STAGES - 1) tiles ---- */
    for (int nStage = 0; nStage < N_STAGES - 1; nStage++) {
        int kStart = nStage * 8;
        aStorePtr = As + SMEM_ROWS * nStage;
        bStorePtr = Bs + SMEM_ROWS * nStage;

        /* A: first K-half (k=0..31), two M-groups */
        cp_async(aStorePtr[storeRow     ] + storeCol, aGlobalAddress + kStart);
        cp_async(aStorePtr[storeRow + 32] + storeCol, aGlobalAddress + 64 * K8 + kStart);
        /* A: second K-half (k=32..63) */
        cp_async(aStorePtr[storeRow + 64] + storeCol, aGlobalAddress + kStart + 4);
        cp_async(aStorePtr[storeRow + 96] + storeCol, aGlobalAddress + 64 * K8 + kStart + 4);

        /* B: 4 groups of 16 K-rows each (kStart*8 is BF16 K-row offset) */
        cp_async(bStorePtr[bStoreRow     ] + bStoreCol, bGlobalBase + kStart * 8 * N8);
        cp_async(bStorePtr[bStoreRow + 32] + bStoreCol, bGlobalBase + (kStart * 8 + 16) * N8);
        cp_async(bStorePtr[bStoreRow + 64] + bStoreCol, bGlobalBase + (kStart * 8 + 32) * N8);
        cp_async(bStorePtr[bStoreRow + 96] + bStoreCol, bGlobalBase + (kStart * 8 + 48) * N8);

        asm volatile("cp.async.commit_group;\n" ::);
    }

    /* ---- MAIN LOOP OVER K BLOCKS (K-tile = 64) ---- */
    for (int nStage = 0; nStage < numKBlocks; nStage++) {
        int kStart = (N_STAGES - 1 + nStage) * 8;
        aStorePtr = As + SMEM_ROWS * ((nStage + N_STAGES - 1) % N_STAGES);
        bStorePtr = Bs + SMEM_ROWS * ((nStage + N_STAGES - 1) % N_STAGES);
        aLoadPtr  = As + SMEM_ROWS * (nStage % N_STAGES);
        bLoadPtr  = Bs + SMEM_ROWS * (nStage % N_STAGES);

        asm volatile("cp.async.wait_group %0;\n" :: "n"(N_STAGES - 2));
        __syncthreads();

        /* ---- Load FIRST K-half (k=0..31) from SMEM to registers ---- */
        /* A: ldmatrix.x4 (non-transposed) */
        for (int m = 0; m < 4; m++) {
            load_matrix_x4(aReg[m]    , aLoadPtr[m * 8 + warpOffsetA + loadRowA] + loadColA);
            load_matrix_x4(aReg[m] + 4, aLoadPtr[m * 8 + warpOffsetA + loadRowA] + (loadColA ^ 2));
        }
        /* B: ldmatrix.x2.trans (first 32 K-rows → SMEM rows 0..63) */
        for (int n = 0; n < 4; n++) {
            int loadColB = loadColB_base ^ n;
            load_matrix_x2_trans(bReg[n]    , bLoadPtr[loadRowB     ] + loadColB);
            load_matrix_x2_trans(bReg[n] + 2, bLoadPtr[loadRowB + 32] + loadColB);
        }

        /* ---- Issue cp.async for NEXT pipeline stage (overlaps with MMA) ---- */
        kStart = (kStart > kStartMax) ? kStartMax : kStart;
        cp_async(aStorePtr[storeRow     ] + storeCol, aGlobalAddress + kStart);
        cp_async(aStorePtr[storeRow + 32] + storeCol, aGlobalAddress + 64 * K8 + kStart);
        cp_async(aStorePtr[storeRow + 64] + storeCol, aGlobalAddress + kStart + 4);
        cp_async(aStorePtr[storeRow + 96] + storeCol, aGlobalAddress + 64 * K8 + kStart + 4);
        cp_async(bStorePtr[bStoreRow     ] + bStoreCol, bGlobalBase + kStart * 8 * N8);
        cp_async(bStorePtr[bStoreRow + 32] + bStoreCol, bGlobalBase + (kStart * 8 + 16) * N8);
        cp_async(bStorePtr[bStoreRow + 64] + bStoreCol, bGlobalBase + (kStart * 8 + 32) * N8);
        cp_async(bStorePtr[bStoreRow + 96] + bStoreCol, bGlobalBase + (kStart * 8 + 48) * N8);
        asm volatile("cp.async.commit_group;\n" ::);

        /* ---- MMA FIRST K-half (k=0..31): 32 HMMA instructions ---- */
        for (int m = 0; m < 4; m++) {
            for (int n = 0; n < 4; n++) {
                mma_m16n8k16(aReg[m]    , bReg[n]    , dReg[m][n], dReg[m][n]);
                mma_m16n8k16(aReg[m] + 4, bReg[n] + 2, dReg[m][n], dReg[m][n]);
            }
        }

        /* ---- Load SECOND K-half (k=32..63) from SMEM to registers ---- */
        /* A: +64 offset for second K-half */
        for (int m = 0; m < 4; m++) {
            load_matrix_x4(aReg[m]    , aLoadPtr[m * 8 + warpOffsetA + loadRowA + 64] + loadColA);
            load_matrix_x4(aReg[m] + 4, aLoadPtr[m * 8 + warpOffsetA + loadRowA + 64] + (loadColA ^ 2));
        }
        /* B: +64 offset for second K-half (SMEM rows 64..127) */
        for (int n = 0; n < 4; n++) {
            int loadColB = loadColB_base ^ n;
            load_matrix_x2_trans(bReg[n]    , bLoadPtr[loadRowB + 64] + loadColB);
            load_matrix_x2_trans(bReg[n] + 2, bLoadPtr[loadRowB + 96] + loadColB);
        }

        /* ---- MMA SECOND K-half (k=32..63): 32 HMMA instructions ---- */
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
 * Launches the GEMM directly on A(M×K) and B(K×N).  No transpose needed:
 * B is loaded N-contiguous via cp.async, and ldmatrix.trans transposes
 * during the SMEM-to-register load.
 * ------------------------------------------------------------------------- */
static int s_M = 0, s_N = 0, s_K = 0;
static bool s_smem_configured = false;

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

    /* Configure dynamic SMEM for GEMM kernel (once) */
    if (!s_smem_configured) {
        cudaFuncSetAttribute(mma_matmul_bf16,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)GEMM_SMEM_BYTES);
        s_smem_configured = true;
    }

    /* GEMM over A(M×K) and B(K×N) — no transpose needed */
    dim3 threads(16, 16);
    dim3 grid(N / 128, M / 128);
    mma_matmul_bf16<<<grid, threads, GEMM_SMEM_BYTES, stream>>>(A, B, C, M, N, K);

    return 0;
}
