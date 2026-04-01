/*
 * TMA-based BF16 matrix multiplication kernel (mma.sync tensor core).
 *
 * Uses cp.async.bulk.tensor (TMA) for global->shared loads with hardware
 * swizzle, mbarrier for pipeline synchronization, and mma.sync.aligned
 * m16n8k16 for compute.
 *
 * Architecture: SM 12.0 (RTX PRO 6000 Blackwell)
 * Tile: 256x128 output, K-tile=32, 3-stage pipeline
 * Threads: 256 (8 warps)
 *
 * TMA tile sizes:
 *   A: 256 M-rows x 32 K-cols, SWIZZLE_64B  -> 16384 bytes/stage
 *   B: split into two 64-col sub-tiles per stage:
 *      B0: 32 K-rows x 64 N-cols, SWIZZLE_128B -> 4096 bytes
 *      B1: 32 K-rows x 64 N-cols, SWIZZLE_128B -> 4096 bytes
 *      Total B per stage: 8192 bytes
 *   3 stages: 3 * (16384 + 8192) = 73728 bytes
 *   mbarrier: 3 * 8 = 24 bytes
 *   Total SMEM: ~74 KB
 *
 * Computes C = A @ B where A is M*K row-major, B is K*N row-major,
 * C is M*N row-major.  All BF16; MMA accumulates in FP32.
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

/* -------------------------------------------------------------------------
 * Constants
 * ------------------------------------------------------------------------- */
#define N_STAGES 3
#define TILE_M   256
#define TILE_N   128
#define TILE_K   32

/* SMEM layout per stage:
 *   A: 256 rows x 64 bytes/row = 16384 bytes (SWIZZLE_64B)
 *   B0: 32 rows x 128 bytes/row = 4096 bytes (SWIZZLE_128B, N-cols 0..63)
 *   B1: 32 rows x 128 bytes/row = 4096 bytes (SWIZZLE_128B, N-cols 64..127)
 * Total per stage: 24576 bytes
 * 3 stages: 73728 bytes
 * mbarrier array: 3 * 8 = 24 bytes at offset 73728
 * Total: 73752 rounded to 128-byte alignment = 73856 bytes */
static const size_t SMEM_A_STAGE  = 16384;  /* 256 rows * 64 bytes */
static const size_t SMEM_B0_STAGE = 4096;   /* 32 rows * 128 bytes */
static const size_t SMEM_B1_STAGE = 4096;   /* 32 rows * 128 bytes */
static const size_t SMEM_STAGE    = SMEM_A_STAGE + SMEM_B0_STAGE + SMEM_B1_STAGE;
static const size_t SMEM_TOTAL_AB = N_STAGES * SMEM_STAGE;        /* 73728 */
static const size_t SMEM_MBAR_OFFSET = SMEM_TOTAL_AB;
static const size_t GEMM_SMEM_BYTES  = 73856;

/* -------------------------------------------------------------------------
 * PTX helpers: mbarrier
 * ------------------------------------------------------------------------- */
__device__ __forceinline__
void mbarrier_init(uint64_t* mbar, unsigned count) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
        :: "r"(addr), "r"(count));
}

__device__ __forceinline__
void mbarrier_arrive_expect_tx(uint64_t* mbar, unsigned tx_bytes) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
        :: "r"(addr), "r"(tx_bytes));
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

/* -------------------------------------------------------------------------
 * PTX helper: TMA 2D load
 * ------------------------------------------------------------------------- */
__device__ __forceinline__
void tma_load_2d(void* smem_dst, const void* tma_desc,
                 int coord0, int coord1, uint64_t* mbar) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_dst);
    unsigned mbar_addr = __cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(smem_addr),
           "l"(tma_desc),
           "r"(coord0), "r"(coord1),
           "r"(mbar_addr)
        : "memory");
}

/* -------------------------------------------------------------------------
 * PTX helper: ldmatrix.sync.aligned.x4
 * ------------------------------------------------------------------------- */
__device__ __forceinline__
void load_matrix_x4(unsigned* destReg, void* srcAddr) {
    unsigned ptxSrcAddr = __cvta_generic_to_shared(srcAddr);
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(destReg[0]), "=r"(destReg[1]), "=r"(destReg[2]), "=r"(destReg[3])
        : "r"(ptxSrcAddr));
}

/* -------------------------------------------------------------------------
 * PTX helper: ldmatrix.sync.aligned.x2.trans
 * ------------------------------------------------------------------------- */
__device__ __forceinline__
void load_matrix_x2_trans(unsigned* destReg, void* srcAddr) {
    unsigned ptxSrcAddr = __cvta_generic_to_shared(srcAddr);
    asm volatile(
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(destReg[0]), "=r"(destReg[1])
        : "r"(ptxSrcAddr));
}

/* -------------------------------------------------------------------------
 * PTX helper: mma.sync.aligned.m16n8k16
 * ------------------------------------------------------------------------- */
__device__ __forceinline__
void mma_m16n8k16(const unsigned* A, const unsigned* B, float* C, float* D) {
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
}

/* -------------------------------------------------------------------------
 * Main GEMM kernel — 256x128 tile, 3-stage TMA pipeline
 * ------------------------------------------------------------------------- */
__launch_bounds__(256, 1)
__global__ void mma_matmul_tma(__nv_bfloat16* __restrict__ C,
                               int M, int N, int K,
                               int totalTiles, int nTilesN,
                               const __grid_constant__ CUtensorMap tma_A,
                               const __grid_constant__ CUtensorMap tma_B) {
    extern __shared__ char smem_buf[];

    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem_buf + SMEM_MBAR_OFFSET);

    int threadID = threadIdx.x;
    int warpID   = threadID / 32;
    int laneID   = threadID % 32;

    /* Warp tiling: 2M x 4N — each warp covers 128M x 32N */
    int warpM = warpID / 4;
    int warpN = warpID % 4;
    int warpOffsetA_rows = warpM * 128;  /* 0 or 128 */

    /* ldmatrix A constants (SWIZZLE_64B, 64 bytes/row, 4 chunks of 16 bytes) */
    int aLdRow_in_tile = (laneID % 8) + ((laneID / 8) & 1) * 8;
    int aLdChunkHalf = laneID / 16;

    /* ldmatrix B constants (SWIZZLE_128B, 128 bytes/row within each sub-tile) */
    int bLdRow_in_tile = (laneID % 8) + ((laneID / 8) & 1) * 8;
    int b_subtile_sel = warpN / 2;
    int b_warp_in_subtile = warpN % 2;

    /* Registers: 8 M-tiles x 4 N-tiles accumulator, ping-pong A regs */
    unsigned aReg[2][8];
    unsigned bReg[4][4];
    float dReg[8][4][4];

    int numKBlocks = K / TILE_K;
    int nTilesM = totalTiles / nTilesN;

    /* Initialize mbarriers (thread 0 only) */
    if (threadID == 0) {
        for (int s = 0; s < N_STAGES; s++) {
            mbarrier_init(&mbar[s], 1);
        }
    }
    __syncthreads();

    /* ---- Persistent tile loop ---- */
    for (int tileIdx = blockIdx.x; tileIdx < totalTiles; tileIdx += gridDim.x) {
        int tile_m, tile_n;
        /* Swizzled tile mapping (4x4 super-tiles) */
        if (nTilesN >= 4 && nTilesM >= 4) {
            int stile_n = 4, stile_m = 4;
            int nSuperN = nTilesN / stile_n;
            int superIdx = tileIdx / (stile_m * stile_n);
            int localIdx = tileIdx % (stile_m * stile_n);
            tile_m = (superIdx / nSuperN) * stile_m + localIdx / stile_n;
            tile_n = (superIdx % nSuperN) * stile_n + localIdx % stile_n;
        } else {
            tile_m = tileIdx / nTilesN;
            tile_n = tileIdx % nTilesN;
        }

        /* Re-init mbarriers for this tile */
        __syncthreads();
        if (threadID == 0) {
            for (int s = 0; s < N_STAGES; s++) {
                mbarrier_inval(&mbar[s]);
                mbarrier_init(&mbar[s], 1);
            }
        }
        __syncthreads();

        /* Reset accumulator */
        #pragma unroll
        for (int m = 0; m < 8; m++)
            #pragma unroll
            for (int n = 0; n < 4; n++)
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    dReg[m][n][i] = 0.f;

        /* TMA coordinates */
        int a_coord1_base = tile_m * TILE_M;
        int b_coord0_N0   = tile_n * TILE_N;
        int b_coord0_N1   = tile_n * TILE_N + 64;

        /* ---- PRELUDE: TMA loads for first (N_STAGES-1) stages ---- */
        #pragma unroll
        for (int s = 0; s < N_STAGES - 1; s++) {
            if (threadID == 0) {
                int k_coord = s * TILE_K;
                char* a_smem  = smem_buf + s * SMEM_STAGE;
                char* b0_smem = smem_buf + s * SMEM_STAGE + SMEM_A_STAGE;
                char* b1_smem = smem_buf + s * SMEM_STAGE + SMEM_A_STAGE + SMEM_B0_STAGE;
                mbarrier_arrive_expect_tx(&mbar[s], SMEM_A_STAGE + SMEM_B0_STAGE + SMEM_B1_STAGE);
                tma_load_2d(a_smem, &tma_A, k_coord, a_coord1_base, &mbar[s]);
                tma_load_2d(b0_smem, &tma_B, b_coord0_N0, k_coord, &mbar[s]);
                tma_load_2d(b1_smem, &tma_B, b_coord0_N1, k_coord, &mbar[s]);
            }
        }

        /* ---- MAIN K-LOOP ---- */
        for (int kBlock = 0; kBlock < numKBlocks; kBlock++) {
            int stage_compute = kBlock % N_STAGES;
            int stage_load = (kBlock + N_STAGES - 1) % N_STAGES;
            int phase = (kBlock / N_STAGES) & 1;

            /* Wait for compute stage data */
            mbarrier_wait_parity(&mbar[stage_compute], phase);

            /* Stage SMEM pointers */
            char* A_stage  = smem_buf + stage_compute * SMEM_STAGE;
            char* B0_stage = smem_buf + stage_compute * SMEM_STAGE + SMEM_A_STAGE;
            char* B1_stage = smem_buf + stage_compute * SMEM_STAGE + SMEM_A_STAGE + SMEM_B0_STAGE;
            char* B_sub_stage = (b_subtile_sel == 0) ? B0_stage : B1_stage;

            /* ---- Load B from SMEM via ldmatrix.x2.trans (all 4 N-tiles) ---- */
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                int b_n_chunk = b_warp_in_subtile * 4 + n;
                int b_k_row0 = bLdRow_in_tile;
                int b_pc0 = b_n_chunk ^ (b_k_row0 & 7);
                load_matrix_x2_trans(bReg[n], B_sub_stage + b_k_row0 * 128 + b_pc0 * 16);
                int b_k_row1 = 16 + bLdRow_in_tile;
                int b_pc1 = b_n_chunk ^ (b_k_row1 & 7);
                load_matrix_x2_trans(bReg[n] + 2, B_sub_stage + b_k_row1 * 128 + b_pc1 * 16);
            }

            /* ---- Issue TMA for next stage (thread 0 only) ---- */
            if (threadID == 0 && kBlock + N_STAGES - 1 < numKBlocks) {
                int next_k = (kBlock + N_STAGES - 1) * TILE_K;
                char* a_smem  = smem_buf + stage_load * SMEM_STAGE;
                char* b0_smem = smem_buf + stage_load * SMEM_STAGE + SMEM_A_STAGE;
                char* b1_smem = smem_buf + stage_load * SMEM_STAGE + SMEM_A_STAGE + SMEM_B0_STAGE;
                mbarrier_arrive_expect_tx(&mbar[stage_load],
                                          SMEM_A_STAGE + SMEM_B0_STAGE + SMEM_B1_STAGE);
                tma_load_2d(a_smem, &tma_A, next_k, a_coord1_base, &mbar[stage_load]);
                tma_load_2d(b0_smem, &tma_B, b_coord0_N0, next_k, &mbar[stage_load]);
                tma_load_2d(b1_smem, &tma_B, b_coord0_N1, next_k, &mbar[stage_load]);
            }

            /* ---- Interleaved A load + MMA: process 8 M-tiles in 4 pairs ---- */
            #pragma unroll
            for (int mp = 0; mp < 4; mp++) {
                /* Load 2 A tiles for this pair */
                #pragma unroll
                for (int mi = 0; mi < 2; mi++) {
                    int m = mp * 2 + mi;
                    int a_row = warpOffsetA_rows + m * 16 + aLdRow_in_tile;
                    int a_xor_key = (a_row >> 1) & 3;
                    int a_lc0 = 0 + aLdChunkHalf;
                    int a_pc0 = a_lc0 ^ a_xor_key;
                    load_matrix_x4(aReg[mi], A_stage + a_row * 64 + a_pc0 * 16);
                    int a_lc1 = 2 + aLdChunkHalf;
                    int a_pc1 = a_lc1 ^ a_xor_key;
                    load_matrix_x4(aReg[mi] + 4, A_stage + a_row * 64 + a_pc1 * 16);
                }
                /* MMA: 2 A tiles x 4 B tiles x 2 K-halves */
                #pragma unroll
                for (int n = 0; n < 4; n++) {
                    mma_m16n8k16(aReg[0],     bReg[n],     dReg[mp*2][n],   dReg[mp*2][n]);
                    mma_m16n8k16(aReg[0] + 4, bReg[n] + 2, dReg[mp*2][n],   dReg[mp*2][n]);
                    mma_m16n8k16(aReg[1],     bReg[n],     dReg[mp*2+1][n], dReg[mp*2+1][n]);
                    mma_m16n8k16(aReg[1] + 4, bReg[n] + 2, dReg[mp*2+1][n], dReg[mp*2+1][n]);
                }
            }
        } /* end K-loop */

        /* ---- EPILOGUE: FP32 -> BF16 store to C ---- */
        int blockRowStart = tile_m * TILE_M;
        int blockColStart = tile_n * TILE_N;
        int warpColOffset = b_subtile_sel * 64 + b_warp_in_subtile * 32;

        int groupID     = laneID >> 2;
        int groupLaneID = laneID & 3;

        #pragma unroll
        for (int m = 0; m < 8; m++) {
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                int row0 = blockRowStart + warpOffsetA_rows + m * 16 + groupID;
                int col0 = blockColStart + warpColOffset + n * 8 + 2 * groupLaneID;

                __nv_bfloat162 packed0 = __floats2bfloat162_rn(dReg[m][n][0],
                                                                dReg[m][n][1]);
                *reinterpret_cast<__nv_bfloat162*>(&C[row0 * N + col0]) = packed0;

                int row1 = row0 + 8;
                __nv_bfloat162 packed1 = __floats2bfloat162_rn(dReg[m][n][2],
                                                                dReg[m][n][3]);
                *reinterpret_cast<__nv_bfloat162*>(&C[row1 * N + col0]) = packed1;
            }
        }
    } /* end persistent tile loop */
}

/* -------------------------------------------------------------------------
 * Host side: TMA descriptor creation and kernel launch
 * ------------------------------------------------------------------------- */
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
    if (!handle) {
        fprintf(stderr, "TMA init: cannot open libcuda.so\n");
        return false;
    }

    typedef CUresult (*getProc_fn)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
    getProc_fn getProc = (getProc_fn)dlsym(handle, "cuGetProcAddress_v2");
    if (!getProc) getProc = (getProc_fn)dlsym(handle, "cuGetProcAddress");
    if (!getProc) {
        fprintf(stderr, "TMA init: cannot find cuGetProcAddress\n");
        return false;
    }

    CUdriverProcAddressQueryResult status;
    CUresult res = getProc("cuTensorMapEncodeTiled",
                           (void**)&s_encodeTiled,
                           12000,
                           CU_GET_PROC_ADDRESS_DEFAULT,
                           &status);
    if (res != CUDA_SUCCESS || !s_encodeTiled) {
        fprintf(stderr, "TMA init: cuGetProcAddress failed (res=%d)\n", res);
        return false;
    }
    return true;
}

/* -------------------------------------------------------------------------
 * Shape parsing
 * ------------------------------------------------------------------------- */
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

/* -------------------------------------------------------------------------
 * kernel_run
 * ------------------------------------------------------------------------- */
extern "C" int kernel_run(__nv_bfloat16** inputs,  int num_inputs,
                          __nv_bfloat16** outputs, int num_outputs,
                          int n, cudaStream_t stream) {
    const __nv_bfloat16* A = inputs[0];
    const __nv_bfloat16* B = inputs[1];
    __nv_bfloat16*       C = outputs[0];

    ensure_shape(n);
    int M = s_M, N = s_N, K = s_K;

    if (!init_tma_encoder()) {
        fprintf(stderr, "kernel_run: TMA encoder init failed\n");
        return -1;
    }

    /* Create TMA descriptors */
    CUtensorMap tma_A, tma_B;

    /* A: row-major M x K.
     * TMA dims: fast = K (dim 0), slow = M (dim 1).
     * Box: 32 K-cols x 256 M-rows. SWIZZLE_64B (box fast = 32*2 = 64 bytes). */
    {
        cuuint64_t dims[2]    = {(cuuint64_t)K, (cuuint64_t)M};
        cuuint64_t strides[1] = {(cuuint64_t)K * 2};
        cuuint32_t box[2]     = {32, 256};
        cuuint32_t elem[2]    = {1, 1};
        CUresult res = s_encodeTiled(&tma_A,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)A,
            dims, strides, box, elem,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "kernel_run: TMA A encode failed (res=%d)\n", res);
            return -2;
        }
    }

    /* B: row-major K x N.
     * TMA dims: fast = N (dim 0), slow = K (dim 1).
     * Box: 64 N-cols x 32 K-rows. SWIZZLE_128B (box fast = 64*2 = 128 bytes). */
    {
        cuuint64_t dims[2]    = {(cuuint64_t)N, (cuuint64_t)K};
        cuuint64_t strides[1] = {(cuuint64_t)N * 2};
        cuuint32_t box[2]     = {64, 32};
        cuuint32_t elem[2]    = {1, 1};
        CUresult res = s_encodeTiled(&tma_B,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)B,
            dims, strides, box, elem,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "kernel_run: TMA B encode failed (res=%d)\n", res);
            return -3;
        }
    }

    /* Tile counts */
    int nTilesM = M / TILE_M;
    int nTilesN = N / TILE_N;
    int totalTiles = nTilesM * nTilesN;

    /* Configure dynamic SMEM */
    static bool s_smem_configured = false;
    if (!s_smem_configured) {
        cudaFuncSetAttribute(mma_matmul_tma,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)GEMM_SMEM_BYTES);
        s_smem_configured = true;
    }

    dim3 threads(256);
    int gridSize = totalTiles < 188 ? totalTiles : 188;
    dim3 grid(gridSize);

    mma_matmul_tma<<<grid, threads, GEMM_SMEM_BYTES, stream>>>(
        C, M, N, K, totalTiles, nTilesN, tma_A, tma_B);

    return 0;
}
