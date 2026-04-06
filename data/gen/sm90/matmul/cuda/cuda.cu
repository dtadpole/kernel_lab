/*
 * BF16 matmul for SM90 — WGMMA + TMA, 3-WG warp-specialized.
 *
 * 3 warpgroups (384 threads):
 *   WG0 = producer (TMA loads, 40 registers per thread)
 *   WG1 = consumer0 (WGMMA rows 0-63, 232 registers per thread)
 *   WG2 = consumer1 (WGMMA rows 64-127, 232 registers per thread)
 *
 * CTA tile 128×128, TILE_K=64, 5-stage pipeline.
 * B loaded MN-major (no transpose) via split TMA: two 64-col loads with 128B swizzle.
 * WGMMA m64n64k16 with trans-b=1.
 * Dual mbarrier sets: mbar_full (producer→consumers) + mbar_empty (consumers→producer).
 * FP32 accumulation → BF16 output.
 */
#include <cuda_bf16.h>
#include <cuda.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

#define TILE_M      128
#define TILE_N      128
#define TILE_K      64
#define TILE_N_HALF 64
#define WG_SIZE     128
#define NUM_WG      3
#define THREADS     (NUM_WG * WG_SIZE)   /* 384 */
#define STAGES      5

#define A_BYTES       (TILE_M * TILE_K * 2)       /* 16384 */
#define B_HALF_BYTES  (TILE_N_HALF * TILE_K * 2)  /* 8192 */
#define B_BYTES       (2 * B_HALF_BYTES)          /* 16384 */
#define STAGE_BYTES   (A_BYTES + B_BYTES)         /* 32768 */
#define MBAR_FULL_OFFSET  (STAGES * STAGE_BYTES)
#define MBAR_EMPTY_OFFSET (MBAR_FULL_OFFSET + 128)
#define SMEM_TOTAL        (MBAR_EMPTY_OFFSET + 128)

/* Per-stage SMEM offset macros (avoids stack-allocated pointer arrays) */
#define SA(s)  (smem + (s) * STAGE_BYTES)
#define SBL(s) (smem + (s) * STAGE_BYTES + A_BYTES)
#define SBR(s) (smem + (s) * STAGE_BYTES + A_BYTES + B_HALF_BYTES)

/* =========================================================================
 * PTX helpers
 * ========================================================================= */
__device__ __forceinline__
void mbarrier_init(uint64_t* mbar, unsigned count) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(addr), "r"(count));
}

__device__ __forceinline__
void mbarrier_arrive_expect_tx(uint64_t* mbar, unsigned tx_bytes) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
                 :: "r"(addr), "r"(tx_bytes));
}

__device__ __forceinline__
void mbarrier_arrive(uint64_t* mbar) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" :: "r"(addr));
}

__device__ __forceinline__
void mbarrier_wait_parity(uint64_t* mbar, unsigned phase) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    unsigned result;
    do {
        asm volatile(
            "{\n .reg .pred p;\n"
            " mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n"
            " selp.u32 %0, 1, 0, p;\n"
            "}\n"
            : "=r"(result) : "r"(addr), "r"(phase));
    } while (result == 0);
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

/* =========================================================================
 * WGMMA descriptor helpers
 * ========================================================================= */

/* Descriptor for 64-element-wide tiles with 128B swizzle.
 * Works for both K-major A (trans=0) and MN-major B halves (trans=1).
 * LBO=1 (16 bytes), SBO=64 (1024 bytes), swizzle=128B.
 */
__device__ __forceinline__
uint64_t make_desc(const void* smem_ptr, int tile_inner) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    int sbo_16B = (8 * tile_inner * 2) >> 4;
    uint64_t desc = 0;
    desc |= (uint64_t)((addr >> 4) & 0x3FFF);
    desc |= (uint64_t)(1) << 16;                      /* LBO = 16 bytes */
    desc |= (uint64_t)(sbo_16B & 0x3FFF) << 32;       /* SBO */
    desc |= (uint64_t)(1) << 62;                       /* swizzle = 128B */
    return desc;
}

__device__ __forceinline__
uint64_t desc_advance(uint64_t desc, int offset_16B) {
    uint32_t lo = (uint32_t)desc + (uint32_t)offset_16B;
    uint32_t hi = (uint32_t)(desc >> 32);
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_wait0() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_wait1() {
    asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
}

/* m64n64k16 — 32 output registers per call.
 * trans-a=0 (K-major A), trans-b=1 (MN-major B half). */
__device__ __forceinline__
void wgmma_m64n64k16(float (&d)[32], uint64_t da, uint64_t db, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "%32,%33,p,1,1,0,1;\n}\n"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),
          "+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
          "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),
          "+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
          "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),
          "+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
          "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),
          "+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
        : "l"(da), "l"(db), "r"(scale_d));
}

/* =========================================================================
 * Epilogue — store 32-element accumulator for m64n64k16
 * ========================================================================= */
__device__ __forceinline__
void store_acc_n64(__nv_bfloat16* C, float (&acc)[32],
                   int ctaRow, int ctaCol, int M, int N, int local_tid) {
    int warp = local_tid / 32;
    int lane = local_tid % 32;
    int row_base = ctaRow + warp * 16 + lane / 4;
    int col_base = ctaCol + (lane % 4) * 2;
    /* Pack 2 adjacent bf16 into 1 x 32-bit store (halves store count) */
    for (int p = 0; p < 8; p++) {
        int col = col_base + p * 8;
        int row0 = row_base, row8 = row_base + 8;
        if (row0 < M && col + 1 < N) {
            __nv_bfloat162 v01 = __halves2bfloat162(
                __float2bfloat16(acc[4*p]), __float2bfloat16(acc[4*p + 1]));
            *(__nv_bfloat162*)(C + row0 * N + col) = v01;
        }
        if (row8 < M && col + 1 < N) {
            __nv_bfloat162 v23 = __halves2bfloat162(
                __float2bfloat16(acc[4*p + 2]), __float2bfloat16(acc[4*p + 3]));
            *(__nv_bfloat162*)(C + row8 * N + col) = v23;
        }
    }
}

/* =========================================================================
 * Main kernel — 3-WG warp-specialized
 *
 * WG0 (producer): TMA loads A + B-left + B-right per stage.
 * WG1 (consumer0): WGMMA on M-rows 0-63, both N-halves.
 * WG2 (consumer1): WGMMA on M-rows 64-127, both N-halves.
 *
 * Synchronization:
 *   mbar_full[STAGES]:  producer→consumers (TMA data ready)
 *   mbar_empty[STAGES]: consumers→producer (SMEM stage free for reuse)
 * ========================================================================= */
__global__ void __launch_bounds__(THREADS, 1)
matmul_wgmma_tma(
    __nv_bfloat16* __restrict__ C, int M, int N, int K,
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B)
{
    extern __shared__ char smem[];

    uint64_t* mbar_full  = (uint64_t*)(smem + MBAR_FULL_OFFSET);
    uint64_t* mbar_empty = (uint64_t*)(smem + MBAR_EMPTY_OFFSET);

    int tid = threadIdx.x;
    int wg_id = tid / WG_SIZE;
    int ctaRow = blockIdx.y * TILE_M;
    int ctaCol = blockIdx.x * TILE_N;
    int numK = (K + TILE_K - 1) / TILE_K;

    /* Init mbarriers (thread 0 only) */
    if (tid == 0) {
        for (int s = 0; s < STAGES; s++) {
            mbarrier_init(&mbar_full[s], 1);  /* 1 TMA transaction set */
            mbarrier_init(&mbar_empty[s], 2); /* 2 consumer arrives */
        }
    }
    __syncthreads();

    if (wg_id == 0) {
        /* =============================================================
         * PRODUCER (WG0): TMA loads only
         *
         * Prefill stages 0..min(STAGES,numK)-1, then main loop:
         * wait mbar_empty → issue TMA → mbar_full auto-completes via TX.
         * Only thread 0 issues TMA; other producer threads idle.
         * ============================================================= */
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");

        if (tid == 0) {
            int prefill = (numK < STAGES) ? numK : STAGES;
            for (int s = 0; s < prefill; s++) {
                mbarrier_arrive_expect_tx(&mbar_full[s], A_BYTES + B_BYTES);
                tma_load_2d(SA(s), &tma_A, s * TILE_K, ctaRow, &mbar_full[s]);
                tma_load_2d(SBL(s), &tma_B, ctaCol, s * TILE_K, &mbar_full[s]);
                tma_load_2d(SBR(s), &tma_B, ctaCol + TILE_N_HALF, s * TILE_K, &mbar_full[s]);
            }

            for (int kt = STAGES; kt < numK; kt++) {
                int stage = kt % STAGES;
                /* Wait for both consumers to release this stage.
                 * try_wait.parity(P) returns true when current parity ≠ P.
                 * We pass the OLD parity so it returns when phase advances. */
                mbarrier_wait_parity(&mbar_empty[stage], ((kt / STAGES) + 1) & 1);
                mbarrier_arrive_expect_tx(&mbar_full[stage], A_BYTES + B_BYTES);
                tma_load_2d(SA(stage), &tma_A, kt * TILE_K, ctaRow, &mbar_full[stage]);
                tma_load_2d(SBL(stage), &tma_B, ctaCol, kt * TILE_K, &mbar_full[stage]);
                tma_load_2d(SBR(stage), &tma_B, ctaCol + TILE_N_HALF, kt * TILE_K, &mbar_full[stage]);
            }
        }

    } else {
        /* =============================================================
         * CONSUMER (WG1 = rows 0-63, WG2 = rows 64-127)
         *
         * Each consumer: wait mbar_full → WGMMA × 4 k-substeps on its
         * M-half (left + right N-halves) → signal mbar_empty for prev stage.
         * ============================================================= */
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");

        int consumer_id = wg_id - 1;  /* 0 or 1 */
        int local_tid = tid - wg_id * WG_SIZE;

        float accL[32], accR[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) { accL[i] = 0.0f; accR[i] = 0.0f; }

        /* A row offset: consumer_id * 64 rows * TILE_K * 2 bytes */
        int a_row_offset = consumer_id * 64 * TILE_K * 2;

        for (int kt = 0; kt < numK; kt++) {
            int stage = kt % STAGES;
            int parity = (kt / STAGES) & 1;

            /* 1. Wait for TMA load */
            mbarrier_wait_parity(&mbar_full[stage], parity);

            /* 2. Build descriptors and issue WGMMA */
            uint64_t da  = make_desc(SA(stage) + a_row_offset, TILE_K);
            uint64_t dbL = make_desc(SBL(stage), TILE_N_HALF);
            uint64_t dbR = make_desc(SBR(stage), TILE_N_HALF);

            int sd = (kt == 0) ? 0 : 1;
            wgmma_fence();
            #pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                uint64_t dak  = desc_advance(da,  ks * 2);
                uint64_t dbLk = desc_advance(dbL, ks * 128);
                uint64_t dbRk = desc_advance(dbR, ks * 128);
                int s = (ks == 0) ? sd : 1;
                wgmma_m64n64k16(accL, dak, dbLk, s);
                wgmma_m64n64k16(accR, dak, dbRk, s);
            }
            wgmma_commit();

            /* 3. Wait for previous WGMMA group, release previous stage */
            if (kt >= 1) {
                wgmma_wait1();
                if (local_tid == 0) {
                    int prev_stage = (kt - 1) % STAGES;
                    mbarrier_arrive(&mbar_empty[prev_stage]);
                }
            }
        }

        /* Drain all outstanding WGMMA */
        wgmma_wait0();

        /* Release last stage */
        if (numK >= 1 && local_tid == 0) {
            int last_stage = (numK - 1) % STAGES;
            mbarrier_arrive(&mbar_empty[last_stage]);
        }

        /* Epilogue: store results */
        int consumer_row = ctaRow + consumer_id * 64;
        store_acc_n64(C, accL, consumer_row, ctaCol,               M, N, local_tid);
        store_acc_n64(C, accR, consumer_row, ctaCol + TILE_N_HALF, M, N, local_tid);
    }
}

/* =========================================================================
 * Host: TMA encoder, kernel dispatch
 * ========================================================================= */
typedef CUresult (*cuTensorMapEncodeTiled_fn)(
    CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
    const cuuint64_t*, const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
    CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion,
    CUtensorMapFloatOOBfill);

static cuTensorMapEncodeTiled_fn s_encodeTiled = nullptr;

static bool init_tma_encoder() {
    if (s_encodeTiled) return true;
    void* handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!handle) handle = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!handle) handle = dlopen("libcuda.so", RTLD_LAZY);
    if (!handle) return false;

    typedef CUresult (*getProc_fn)(const char*, void**, int, cuuint64_t,
                                   CUdriverProcAddressQueryResult*);
    getProc_fn getProc = (getProc_fn)dlsym(handle, "cuGetProcAddress_v2");
    if (!getProc) getProc = (getProc_fn)dlsym(handle, "cuGetProcAddress");
    if (!getProc) return false;

    CUdriverProcAddressQueryResult status;
    CUresult res = getProc("cuTensorMapEncodeTiled",
                           (void**)&s_encodeTiled, 12000,
                           CU_GET_PROC_ADDRESS_DEFAULT, &status);
    return (res == CUDA_SUCCESS && s_encodeTiled);
}

extern "C" int kernel_run(
    __nv_bfloat16** inputs, int num_inputs,
    __nv_bfloat16** outputs, int num_outputs,
    int n, cudaStream_t stream)
{
    int dim = (int)sqrtf((float)n);
    if (dim * dim != n) return 1;
    if (!init_tma_encoder()) { fprintf(stderr, "TMA init failed\n"); return -1; }

    const __nv_bfloat16* A = inputs[0];
    const __nv_bfloat16* B = inputs[1];
    __nv_bfloat16* C = outputs[0];
    int M = dim, N = dim, K = dim;

    /* Create TMA descriptors */
    CUtensorMap tma_A, tma_B;
    {
        cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)M};
        cuuint64_t str[1]  = {(cuuint64_t)(K * 2)};
        cuuint32_t box[2]  = {(cuuint32_t)TILE_K, (cuuint32_t)TILE_M};
        cuuint32_t el[2]   = {1, 1};
        CUresult r = s_encodeTiled(&tma_A, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
            (void*)A, dims, str, box, el,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) return -2;
    }
    {
        /* B is K×N row-major (N contiguous). TMA dims [N, K], box [64, TILE_K].
         * 128B swizzle: 64 bf16 × 2 bytes = 128 bytes ✓
         * Two TMA loads per stage: one for left N-half, one for right N-half. */
        cuuint64_t dims[2] = {(cuuint64_t)N, (cuuint64_t)K};
        cuuint64_t str[1]  = {(cuuint64_t)(N * 2)};
        cuuint32_t box[2]  = {(cuuint32_t)TILE_N_HALF, (cuuint32_t)TILE_K};
        cuuint32_t el[2]   = {1, 1};
        CUresult r = s_encodeTiled(&tma_B, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
            (void*)B, dims, str, box, el,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) return -3;
    }

    /* Launch: 3 warpgroups, 384 threads */
    dim3 block(THREADS);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    cudaFuncSetAttribute(matmul_wgmma_tma,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
    matmul_wgmma_tma<<<grid, block, SMEM_TOTAL, stream>>>(
        C, M, N, K, tma_A, tma_B);

    return 0;
}
