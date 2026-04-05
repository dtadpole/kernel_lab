/*
 * TMA-based BF16 matrix multiplication kernel with multi-tile dispatch.
 *
 * Two kernel variants dispatched by matrix size:
 *   Big:   256×128 tile, 256 threads, 3-stage — for M,N ≥ 2048
 *   Small: 128×64  tile, 128 threads, 4-stage — for M,N < 2048
 *
 * Uses cp.async.bulk.tensor (TMA) for global→shared loads with hardware
 * swizzle, mbarrier for pipeline synchronization, and mma.sync.aligned
 * m16n8k16 for compute.  Architecture: SM 12.0 (Blackwell).
 *
 * kernel_run contract:
 *   extern "C" int kernel_run(__nv_bfloat16** inputs, int num_inputs,
 *                             __nv_bfloat16** outputs, int num_outputs,
 *                             int n, cudaStream_t stream);
 */
#include <mma.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* =========================================================================
 * Constants
 * ========================================================================= */
#define TILE_K 32

/* Big kernel: 256×128, 256 threads, 3 stages */
#define BIG_TILE_M   256
#define BIG_TILE_N   128
#define BIG_STAGES   3
#define BIG_A_STAGE  16384   /* 256 rows × 64 B */
#define BIG_B0_STAGE 4096    /* 32 rows × 128 B */
#define BIG_B1_STAGE 4096
#define BIG_SMEM_STAGE (BIG_A_STAGE + BIG_B0_STAGE + BIG_B1_STAGE)  /* 24576 */
#define BIG_SMEM_BYTES 73856 /* 3×24576 + 128 (mbar+align) */

/* Small kernel: 128×64, 128 threads, 4 stages */
#define SMALL_TILE_M   128
#define SMALL_TILE_N   64
#define SMALL_STAGES   4
#define SMALL_A_STAGE  8192  /* 128 rows × 64 B */
#define SMALL_B_STAGE  4096  /* 32 rows × 128 B (single sub-tile for 64 N-cols) */
#define SMALL_SMEM_STAGE (SMALL_A_STAGE + SMALL_B_STAGE)  /* 12288 */
#define SMALL_SMEM_BYTES 49280 /* 4×12288 + 128 (mbar+align) */

/* =========================================================================
 * PTX helpers (shared by both kernels)
 * ========================================================================= */
__device__ __forceinline__
void mbarrier_init(uint64_t* mbar, unsigned count) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(addr), "r"(count));
}

__device__ __forceinline__
void mbarrier_arrive_expect_tx(uint64_t* mbar, unsigned tx_bytes) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(addr), "r"(tx_bytes));
}

__device__ __forceinline__
void mbarrier_wait_parity(uint64_t* mbar, unsigned phase) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    unsigned result;
    do {
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n"
            "  selp.u32 %0, 1, 0, p;\n"
            "}\n"
            : "=r"(result) : "r"(addr), "r"(phase));
    } while (result == 0);
}

__device__ __forceinline__
void mbarrier_inval(uint64_t* mbar) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.inval.shared.b64 [%0];\n" :: "r"(addr));
}

__device__ __forceinline__
void tma_load_2d(void* smem_dst, const void* tma_desc,
                 int coord0, int coord1, uint64_t* mbar) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_dst);
    unsigned mbar_addr = __cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(smem_addr), "l"(tma_desc),
           "r"(coord0), "r"(coord1), "r"(mbar_addr)
        : "memory");
}

__device__ __forceinline__
void load_matrix_x4(unsigned* destReg, void* srcAddr) {
    unsigned s = __cvta_generic_to_shared(srcAddr);
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(destReg[0]),"=r"(destReg[1]),"=r"(destReg[2]),"=r"(destReg[3])
        : "r"(s));
}

__device__ __forceinline__
void load_matrix_x2_trans(unsigned* destReg, void* srcAddr) {
    unsigned s = __cvta_generic_to_shared(srcAddr);
    asm volatile(
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(destReg[0]),"=r"(destReg[1]) : "r"(s));
}

__device__ __forceinline__
void mma_m16n8k16(const unsigned* A, const unsigned* B, float* C, float* D) {
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]),"=f"(D[1]),"=f"(D[2]),"=f"(D[3])
        : "r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),
          "r"(B[0]),"r"(B[1]),
          "f"(C[0]),"f"(C[1]),"f"(C[2]),"f"(C[3]));
}

/* =========================================================================
 * Big kernel: 256×128 tile, 256 threads, 3-stage pipeline
 * ========================================================================= */
__launch_bounds__(256, 1)
__global__ void mma_matmul_tma_big(
        __nv_bfloat16* __restrict__ C, int M, int N, int K,
        int totalTiles, int nTilesN,
        const __grid_constant__ CUtensorMap tma_A,
        const __grid_constant__ CUtensorMap tma_B) {
    extern __shared__ char smem_buf[];
    uint64_t* mbar = reinterpret_cast<uint64_t*>(
        smem_buf + BIG_STAGES * BIG_SMEM_STAGE);

    int threadID = threadIdx.x;
    int warpID   = threadID / 32;
    int laneID   = threadID % 32;

    /* 2M×4N warp layout, each warp: 128M×32N */
    int warpM = warpID / 4;
    int warpN = warpID % 4;
    int warpOffsetA_rows = warpM * 128;

    int aLdRow = (laneID % 8) + ((laneID / 8) & 1) * 8;
    int aLdChunkHalf = laneID / 16;
    int bLdRow = (laneID % 8) + ((laneID / 8) & 1) * 8;
    int b_sub = warpN / 2;
    int b_half = warpN % 2;

    unsigned aReg[2][8];
    unsigned bReg[4][4];
    float dReg[8][4][4];

    int numKBlocks = K / TILE_K;
    int nTilesM = totalTiles / nTilesN;

    if (threadID == 0)
        for (int s = 0; s < BIG_STAGES; s++) mbarrier_init(&mbar[s], 1);
    __syncthreads();

    for (int tileIdx = blockIdx.x; tileIdx < totalTiles; tileIdx += gridDim.x) {
        int tile_m, tile_n;
        if (nTilesN >= 4 && nTilesM >= 4) {
            int nSuperN = nTilesN / 4;
            int superIdx = tileIdx / 16;
            int localIdx = tileIdx % 16;
            tile_m = (superIdx / nSuperN) * 4 + localIdx / 4;
            tile_n = (superIdx % nSuperN) * 4 + localIdx % 4;
        } else {
            tile_m = tileIdx / nTilesN;
            tile_n = tileIdx % nTilesN;
        }

        __syncthreads();
        if (threadID == 0)
            for (int s = 0; s < BIG_STAGES; s++) {
                mbarrier_inval(&mbar[s]); mbarrier_init(&mbar[s], 1); }
        __syncthreads();

        #pragma unroll
        for (int m = 0; m < 8; m++)
            #pragma unroll
            for (int n = 0; n < 4; n++)
                dReg[m][n][0] = dReg[m][n][1] = dReg[m][n][2] = dReg[m][n][3] = 0.f;

        int a_m = tile_m * BIG_TILE_M;
        int b_n0 = tile_n * BIG_TILE_N;
        int b_n1 = b_n0 + 64;

        /* Prelude */
        int prelude = (numKBlocks < BIG_STAGES - 1) ? numKBlocks : (BIG_STAGES - 1);
        for (int s = 0; s < prelude; s++) {
            if (threadID == 0) {
                int kc = s * TILE_K;
                char* as = smem_buf + s * BIG_SMEM_STAGE;
                char* b0s = as + BIG_A_STAGE;
                char* b1s = b0s + BIG_B0_STAGE;
                mbarrier_arrive_expect_tx(&mbar[s], BIG_SMEM_STAGE);
                tma_load_2d(as, &tma_A, kc, a_m, &mbar[s]);
                tma_load_2d(b0s, &tma_B, b_n0, kc, &mbar[s]);
                tma_load_2d(b1s, &tma_B, b_n1, kc, &mbar[s]);
            }
        }

        /* K-loop */
        for (int kb = 0; kb < numKBlocks; kb++) {
            int sc = kb % BIG_STAGES;
            int sl = (kb + BIG_STAGES - 1) % BIG_STAGES;
            int phase = (kb / BIG_STAGES) & 1;

            mbarrier_wait_parity(&mbar[sc], phase);

            char* A_s = smem_buf + sc * BIG_SMEM_STAGE;
            char* B0_s = A_s + BIG_A_STAGE;
            char* B1_s = B0_s + BIG_B0_STAGE;
            char* B_s = (b_sub == 0) ? B0_s : B1_s;

            /* Load B */
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                int bc = b_half * 4 + n;
                int r0 = bLdRow, r1 = 16 + bLdRow;
                load_matrix_x2_trans(bReg[n],     B_s + r0 * 128 + (bc ^ (r0 & 7)) * 16);
                load_matrix_x2_trans(bReg[n] + 2, B_s + r1 * 128 + (bc ^ (r1 & 7)) * 16);
            }

            /* TMA next */
            if (threadID == 0 && kb + BIG_STAGES - 1 < numKBlocks) {
                int nk = (kb + BIG_STAGES - 1) * TILE_K;
                char* as = smem_buf + sl * BIG_SMEM_STAGE;
                char* b0s = as + BIG_A_STAGE;
                char* b1s = b0s + BIG_B0_STAGE;
                mbarrier_arrive_expect_tx(&mbar[sl], BIG_SMEM_STAGE);
                tma_load_2d(as, &tma_A, nk, a_m, &mbar[sl]);
                tma_load_2d(b0s, &tma_B, b_n0, nk, &mbar[sl]);
                tma_load_2d(b1s, &tma_B, b_n1, nk, &mbar[sl]);
            }

            /* Interleaved A load + MMA */
            #pragma unroll
            for (int mp = 0; mp < 4; mp++) {
                #pragma unroll
                for (int mi = 0; mi < 2; mi++) {
                    int m = mp * 2 + mi;
                    int ar = warpOffsetA_rows + m * 16 + aLdRow;
                    int xk = (ar >> 1) & 3;
                    load_matrix_x4(aReg[mi],     A_s + ar * 64 + ((0 + aLdChunkHalf) ^ xk) * 16);
                    load_matrix_x4(aReg[mi] + 4, A_s + ar * 64 + ((2 + aLdChunkHalf) ^ xk) * 16);
                }
                #pragma unroll
                for (int n = 0; n < 4; n++) {
                    mma_m16n8k16(aReg[0],     bReg[n],     dReg[mp*2][n],   dReg[mp*2][n]);
                    mma_m16n8k16(aReg[0] + 4, bReg[n] + 2, dReg[mp*2][n],   dReg[mp*2][n]);
                    mma_m16n8k16(aReg[1],     bReg[n],     dReg[mp*2+1][n], dReg[mp*2+1][n]);
                    mma_m16n8k16(aReg[1] + 4, bReg[n] + 2, dReg[mp*2+1][n], dReg[mp*2+1][n]);
                }
            }
        }

        /* Epilogue */
        int rBase = tile_m * BIG_TILE_M;
        int cBase = tile_n * BIG_TILE_N;
        int wCol = b_sub * 64 + b_half * 32;
        int gid = laneID >> 2, glid = laneID & 3;
        #pragma unroll
        for (int m = 0; m < 8; m++) {
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                int r0 = rBase + warpOffsetA_rows + m * 16 + gid;
                int c0 = cBase + wCol + n * 8 + 2 * glid;
                *reinterpret_cast<__nv_bfloat162*>(&C[r0 * N + c0]) =
                    __floats2bfloat162_rn(dReg[m][n][0], dReg[m][n][1]);
                *reinterpret_cast<__nv_bfloat162*>(&C[(r0+8) * N + c0]) =
                    __floats2bfloat162_rn(dReg[m][n][2], dReg[m][n][3]);
            }
        }
    }
}

/* =========================================================================
 * Small kernel: 128×64 tile, 128 threads, 4-stage pipeline
 * ========================================================================= */
__launch_bounds__(128, 1)
__global__ void mma_matmul_tma_small(
        __nv_bfloat16* __restrict__ C, int M, int N, int K,
        int totalTiles, int nTilesN,
        const __grid_constant__ CUtensorMap tma_A,
        const __grid_constant__ CUtensorMap tma_B) {
    extern __shared__ char smem_buf[];
    uint64_t* mbar = reinterpret_cast<uint64_t*>(
        smem_buf + SMALL_STAGES * SMALL_SMEM_STAGE);

    int threadID = threadIdx.x;
    int warpID   = threadID / 32;
    int laneID   = threadID % 32;

    /* 2M×2N warp layout, each warp: 64M×32N */
    int warpM = warpID / 2;
    int warpN = warpID % 2;
    int warpOffsetA_rows = warpM * 64;

    int aLdRow = (laneID % 8) + ((laneID / 8) & 1) * 8;
    int aLdChunkHalf = laneID / 16;
    int bLdRow = (laneID % 8) + ((laneID / 8) & 1) * 8;

    unsigned aReg[4][8];
    unsigned bReg[4][4];
    float dReg[4][4][4];

    int numKBlocks = K / TILE_K;
    int nTilesM = totalTiles / nTilesN;

    if (threadID == 0)
        for (int s = 0; s < SMALL_STAGES; s++) mbarrier_init(&mbar[s], 1);
    __syncthreads();

    for (int tileIdx = blockIdx.x; tileIdx < totalTiles; tileIdx += gridDim.x) {
        int tile_m, tile_n;
        if (nTilesN >= 4 && nTilesM >= 4) {
            int nSuperN = nTilesN / 4;
            int superIdx = tileIdx / 16;
            int localIdx = tileIdx % 16;
            tile_m = (superIdx / nSuperN) * 4 + localIdx / 4;
            tile_n = (superIdx % nSuperN) * 4 + localIdx % 4;
        } else {
            tile_m = tileIdx / nTilesN;
            tile_n = tileIdx % nTilesN;
        }

        __syncthreads();
        if (threadID == 0)
            for (int s = 0; s < SMALL_STAGES; s++) {
                mbarrier_inval(&mbar[s]); mbarrier_init(&mbar[s], 1); }
        __syncthreads();

        #pragma unroll
        for (int m = 0; m < 4; m++)
            #pragma unroll
            for (int n = 0; n < 4; n++)
                dReg[m][n][0] = dReg[m][n][1] = dReg[m][n][2] = dReg[m][n][3] = 0.f;

        int a_m = tile_m * SMALL_TILE_M;
        int b_n = tile_n * SMALL_TILE_N;

        /* Prelude */
        int prelude = (numKBlocks < SMALL_STAGES - 1) ? numKBlocks : (SMALL_STAGES - 1);
        for (int s = 0; s < prelude; s++) {
            if (threadID == 0) {
                int kc = s * TILE_K;
                char* as = smem_buf + s * SMALL_SMEM_STAGE;
                char* bs = as + SMALL_A_STAGE;
                mbarrier_arrive_expect_tx(&mbar[s], SMALL_SMEM_STAGE);
                tma_load_2d(as, &tma_A, kc, a_m, &mbar[s]);
                tma_load_2d(bs, &tma_B, b_n, kc, &mbar[s]);
            }
        }

        /* K-loop */
        for (int kb = 0; kb < numKBlocks; kb++) {
            int sc = kb % SMALL_STAGES;
            int sl = (kb + SMALL_STAGES - 1) % SMALL_STAGES;
            int phase = (kb / SMALL_STAGES) & 1;

            mbarrier_wait_parity(&mbar[sc], phase);

            char* A_s = smem_buf + sc * SMALL_SMEM_STAGE;
            char* B_s = A_s + SMALL_A_STAGE;

            /* Load A: 4 M-tiles (all at once, 64 floats accumulators fit) */
            #pragma unroll
            for (int m = 0; m < 4; m++) {
                int ar = warpOffsetA_rows + m * 16 + aLdRow;
                int xk = (ar >> 1) & 3;
                load_matrix_x4(aReg[m],     A_s + ar * 64 + ((0 + aLdChunkHalf) ^ xk) * 16);
                load_matrix_x4(aReg[m] + 4, A_s + ar * 64 + ((2 + aLdChunkHalf) ^ xk) * 16);
            }

            /* Load B: 4 N-tiles from single sub-tile
             * warpN=0 → chunks 0..3, warpN=1 → chunks 4..7 */
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                int bc = warpN * 4 + n;
                int r0 = bLdRow, r1 = 16 + bLdRow;
                load_matrix_x2_trans(bReg[n],     B_s + r0 * 128 + (bc ^ (r0 & 7)) * 16);
                load_matrix_x2_trans(bReg[n] + 2, B_s + r1 * 128 + (bc ^ (r1 & 7)) * 16);
            }

            /* TMA next */
            if (threadID == 0 && kb + SMALL_STAGES - 1 < numKBlocks) {
                int nk = (kb + SMALL_STAGES - 1) * TILE_K;
                char* as = smem_buf + sl * SMALL_SMEM_STAGE;
                char* bs = as + SMALL_A_STAGE;
                mbarrier_arrive_expect_tx(&mbar[sl], SMALL_SMEM_STAGE);
                tma_load_2d(as, &tma_A, nk, a_m, &mbar[sl]);
                tma_load_2d(bs, &tma_B, b_n, nk, &mbar[sl]);
            }

            /* MMA: 4×4 sub-tiles × 2 K-halves */
            #pragma unroll
            for (int m = 0; m < 4; m++) {
                #pragma unroll
                for (int n = 0; n < 4; n++) {
                    mma_m16n8k16(aReg[m],     bReg[n],     dReg[m][n], dReg[m][n]);
                    mma_m16n8k16(aReg[m] + 4, bReg[n] + 2, dReg[m][n], dReg[m][n]);
                }
            }
        }

        /* Epilogue */
        int rBase = tile_m * SMALL_TILE_M;
        int cBase = tile_n * SMALL_TILE_N;
        int wCol = warpN * 32;
        int gid = laneID >> 2, glid = laneID & 3;
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                int r0 = rBase + warpOffsetA_rows + m * 16 + gid;
                int c0 = cBase + wCol + n * 8 + 2 * glid;
                *reinterpret_cast<__nv_bfloat162*>(&C[r0 * N + c0]) =
                    __floats2bfloat162_rn(dReg[m][n][0], dReg[m][n][1]);
                *reinterpret_cast<__nv_bfloat162*>(&C[(r0+8) * N + c0]) =
                    __floats2bfloat162_rn(dReg[m][n][2], dReg[m][n][3]);
            }
        }
    }
}

/* =========================================================================
 * Host: TMA encoder, shape parsing, kernel dispatch
 * ========================================================================= */
typedef CUresult (*cuTensorMapEncodeTiled_fn)(
    CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
    const cuuint64_t*, const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
    CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static cuTensorMapEncodeTiled_fn s_encodeTiled = nullptr;

static bool init_tma_encoder() {
    if (s_encodeTiled) return true;
    void* handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!handle) handle = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!handle) handle = dlopen("libcuda.so", RTLD_LAZY);
    if (!handle) return false;

    typedef CUresult (*getProc_fn)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
    getProc_fn getProc = (getProc_fn)dlsym(handle, "cuGetProcAddress_v2");
    if (!getProc) getProc = (getProc_fn)dlsym(handle, "cuGetProcAddress");
    if (!getProc) return false;

    CUdriverProcAddressQueryResult status;
    CUresult res = getProc("cuTensorMapEncodeTiled",
                           (void**)&s_encodeTiled, 12000,
                           CU_GET_PROC_ADDRESS_DEFAULT, &status);
    return (res == CUDA_SUCCESS && s_encodeTiled);
}

static int s_M = 0, s_N = 0, s_K = 0;
static void ensure_shape(int n) {
    if (s_M > 0) return;
    const char* shape = getenv("CUDA_EXEC_PARAM_SHAPE");
    if (shape) {
        int d0 = 0, d1 = 0;
        const char* p = shape;
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

extern "C" int kernel_run(__nv_bfloat16** inputs, int num_inputs,
                          __nv_bfloat16** outputs, int num_outputs,
                          int n, cudaStream_t stream) {
    const __nv_bfloat16* A = inputs[0];
    const __nv_bfloat16* B = inputs[1];
    __nv_bfloat16*       C = outputs[0];

    ensure_shape(n);
    int M = s_M, N = s_N, K = s_K;

    if (!init_tma_encoder()) return -1;

    static int s_numSMs = 0;
    if (s_numSMs == 0)
        cudaDeviceGetAttribute(&s_numSMs, cudaDevAttrMultiProcessorCount, 0);

    /* Dispatch: big kernel for ≥2048, small kernel otherwise */
    bool use_big = (M >= 2048 && N >= 2048);

    if (use_big) {
        /* --- Big: 256×128 tile --- */
        int tileM = BIG_TILE_M, tileN = BIG_TILE_N;
        int nTilesM = M / tileM, nTilesN = N / tileN;
        int totalTiles = nTilesM * nTilesN;

        CUtensorMap tma_A, tma_B;
        {
            cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)M};
            cuuint64_t strides[1] = {(cuuint64_t)K * 2};
            cuuint32_t box[2] = {32, 256};
            cuuint32_t elem[2] = {1, 1};
            if (s_encodeTiled(&tma_A, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)A,
                dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) return -2;
        }
        {
            cuuint64_t dims[2] = {(cuuint64_t)N, (cuuint64_t)K};
            cuuint64_t strides[1] = {(cuuint64_t)N * 2};
            cuuint32_t box[2] = {64, 32};
            cuuint32_t elem[2] = {1, 1};
            if (s_encodeTiled(&tma_B, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)B,
                dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) return -3;
        }

        static bool cfg_big = false;
        if (!cfg_big) {
            cudaFuncSetAttribute(mma_matmul_tma_big,
                cudaFuncAttributeMaxDynamicSharedMemorySize, BIG_SMEM_BYTES);
            cfg_big = true;
        }

        int grid = totalTiles < s_numSMs ? totalTiles : s_numSMs;
        mma_matmul_tma_big<<<grid, 256, BIG_SMEM_BYTES, stream>>>(
            C, M, N, K, totalTiles, nTilesN, tma_A, tma_B);
    } else {
        /* --- Small: 128×64 tile --- */
        int tileM = SMALL_TILE_M, tileN = SMALL_TILE_N;
        int nTilesM = M / tileM, nTilesN = N / tileN;
        int totalTiles = nTilesM * nTilesN;

        CUtensorMap tma_A, tma_B;
        {
            cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)M};
            cuuint64_t strides[1] = {(cuuint64_t)K * 2};
            cuuint32_t box[2] = {32, 128};
            cuuint32_t elem[2] = {1, 1};
            if (s_encodeTiled(&tma_A, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)A,
                dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) return -2;
        }
        {
            cuuint64_t dims[2] = {(cuuint64_t)N, (cuuint64_t)K};
            cuuint64_t strides[1] = {(cuuint64_t)N * 2};
            cuuint32_t box[2] = {64, 32};
            cuuint32_t elem[2] = {1, 1};
            if (s_encodeTiled(&tma_B, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)B,
                dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) return -3;
        }

        static bool cfg_small = false;
        if (!cfg_small) {
            cudaFuncSetAttribute(mma_matmul_tma_small,
                cudaFuncAttributeMaxDynamicSharedMemorySize, SMALL_SMEM_BYTES);
            cfg_small = true;
        }

        int grid = totalTiles < s_numSMs ? totalTiles : s_numSMs;
        mma_matmul_tma_small<<<grid, 128, SMALL_SMEM_BYTES, stream>>>(
            C, M, N, K, totalTiles, nTilesN, tma_A, tma_B);
    }

    return 0;
}
