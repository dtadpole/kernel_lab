/*
 * BF16 matmul for SM90 — WGMMA + TMA, 3-WG warp-specialized.
 *
 * 3 warpgroups (384 threads):
 *   WG0 = producer (TMA loads, 40 registers per thread)
 *   WG1 = consumer0 (WGMMA rows 0-63, 232 registers per thread)
 *   WG2 = consumer1 (WGMMA rows 64-127, 232 registers per thread)
 *
 * CTA tile 128×256, TILE_K=64, 4-stage pipeline.
 * B loaded MN-major (no transpose) via split TMA: four 64-col loads with 128B swizzle.
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
#define TILE_N      256
#define TILE_K      64
#define TILE_NQ     64             /* each B quarter: 64 N-elements = 128B swizzle */
#define NUM_NQ      4              /* 4 quarters of 64 = 256 */
#define WG_SIZE     128
#define NUM_WG      3
#define THREADS     (NUM_WG * WG_SIZE)   /* 384 */
#define STAGES      4

#define A_BYTES       (TILE_M * TILE_K * 2)       /* 16384 */
#define BQ_BYTES      (TILE_NQ * TILE_K * 2)      /* 8192 per quarter */
#define B_BYTES       (NUM_NQ * BQ_BYTES)         /* 32768 total */
#define STAGE_BYTES   (A_BYTES + B_BYTES)         /* 49152 */
#define MBAR_FULL_OFFSET   (STAGES * STAGE_BYTES)
#define MBAR_EMPTY_OFFSET  (MBAR_FULL_OFFSET + 128)
#define MBAR_TREADY_OFFSET (MBAR_EMPTY_OFFSET + 128)
#define TILE_INFO_OFFSET   (MBAR_TREADY_OFFSET + 16)
#define SMEM_TOTAL         (TILE_INFO_OFFSET + 16)

/* Per-stage SMEM offset macros */
#define SA(s)   (smem + (s) * STAGE_BYTES)
#define SBQ(s,q) (smem + (s) * STAGE_BYTES + A_BYTES + (q) * BQ_BYTES)

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

/* TMA store: async SMEM → GMEM bulk transfer */
__device__ __forceinline__
void tma_store_2d(const void* tma_desc, int coord0, int coord1,
                  const void* smem_src) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_src);
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];\n"
        :: "l"(tma_desc), "r"(coord0), "r"(coord1), "r"(smem_addr)
        : "memory");
}

__device__ __forceinline__
void tma_store_commit() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

__device__ __forceinline__
void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
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
 * Epilogue — store accumulator to SMEM for TMA store, or to GMEM directly
 * ========================================================================= */

/* Write 32-element m64n64k16 accumulator to SMEM in row-major BF16 layout.
 * smem_base points to a 64×64 BF16 buffer (8KB) in shared memory.
 * Tile is 64 rows × 64 cols (matching one N-quarter). */
__device__ __forceinline__
void store_acc_to_smem(__nv_bfloat16* smem_base, float (&acc)[32],
                       int local_tid) {
    int warp = local_tid / 32;
    int lane = local_tid % 32;
    int row_base = warp * 16 + lane / 4;
    int col_base = (lane % 4) * 2;
    for (int p = 0; p < 8; p++) {
        int col = col_base + p * 8;
        int row0 = row_base, row8 = row_base + 8;
        __nv_bfloat162 v01 = __halves2bfloat162(
            __float2bfloat16(acc[4*p]), __float2bfloat16(acc[4*p + 1]));
        *(__nv_bfloat162*)(smem_base + row0 * TILE_NQ + col) = v01;
        __nv_bfloat162 v23 = __halves2bfloat162(
            __float2bfloat16(acc[4*p + 2]), __float2bfloat16(acc[4*p + 3]));
        *(__nv_bfloat162*)(smem_base + row8 * TILE_NQ + col) = v23;
    }
}

/* Fallback: direct GMEM store (for correctness comparison) */
__device__ __forceinline__
void store_acc_n64(__nv_bfloat16* C, float (&acc)[32],
                   int ctaRow, int ctaCol, int M, int N, int local_tid) {
    int warp = local_tid / 32;
    int lane = local_tid % 32;
    int row_base = ctaRow + warp * 16 + lane / 4;
    int col_base = ctaCol + (lane % 4) * 2;
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
    const __grid_constant__ CUtensorMap tma_B,
    const __grid_constant__ CUtensorMap tma_C,
    int grid_m, int grid_n, int total_tiles, unsigned* tile_counter)
{
    extern __shared__ char smem[];

    uint64_t* mbar_full   = (uint64_t*)(smem + MBAR_FULL_OFFSET);
    uint64_t* mbar_empty  = (uint64_t*)(smem + MBAR_EMPTY_OFFSET);
    uint64_t* mbar_tready = (uint64_t*)(smem + MBAR_TREADY_OFFSET);
    int*      tile_info   = (int*)(smem + TILE_INFO_OFFSET);

    int tid = threadIdx.x;
    int wg_id = tid / WG_SIZE;
    int numK = (K + TILE_K - 1) / TILE_K;
    int phases_per_tile = (numK + STAGES - 1) / STAGES;

    /* Init mbarriers (thread 0 only) */
    if (tid == 0) {
        for (int s = 0; s < STAGES; s++) {
            mbarrier_init(&mbar_full[s], 1);
            mbarrier_init(&mbar_empty[s], 2);
        }
        mbarrier_init(mbar_tready, 1);
    }
    __syncthreads();

    if (wg_id == 0) {
        /* =============================================================
         * PRODUCER — persistent tile loop
         * ============================================================= */
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n");

        for (int tile = 0; ; tile++) {
            if (tid != 0) {
                /* Non-zero producer threads: wait for tile ready, check exit */
                mbarrier_wait_parity(mbar_tready, tile & 1);
                if (tile_info[2] >= total_tiles) break;
                continue;
            }

            /* tid == 0: get next tile, setup, prefill */

            /* Wait for consumers to finish previous tile's SMEM usage.
             * Only need to wait on the LAST mbar_empty (drain signal).
             * By the time consumer signals drain, all earlier stages are already free. */
            if (tile > 0) {
                int prev_phase_base = (tile - 1) * phases_per_tile;
                int last_stage = (numK - 1) % STAGES;
                int last_phase = (prev_phase_base + (numK - 1) / STAGES) & 1;
                mbarrier_wait_parity(&mbar_empty[last_stage], last_phase);
            }

            /* Get next tile via atomic counter */
            int tile_id = atomicAdd(tile_counter, 1);

            /* CTA swizzle */
            int pid_m, pid_n;
            {
                const int GROUP_M = 12;
                int group_id  = tile_id / (GROUP_M * grid_n);
                int first_m   = group_id * GROUP_M;
                int group_sz  = (first_m + GROUP_M <= grid_m) ? GROUP_M : (grid_m - first_m);
                int local_id  = tile_id % (GROUP_M * grid_n);
                pid_m = first_m + local_id % group_sz;
                pid_n = local_id / group_sz;
            }
            tile_info[0] = pid_m * TILE_M;  /* ctaRow */
            tile_info[1] = pid_n * TILE_N;  /* ctaCol */
            tile_info[2] = tile_id;

            /* Prefill TMA for this tile */
            if (tile_id < total_tiles) {
                int ctaRow = tile_info[0];
                int ctaCol = tile_info[1];
                int prefill = (numK < STAGES) ? numK : STAGES;
                for (int s = 0; s < prefill; s++) {
                    mbarrier_arrive_expect_tx(&mbar_full[s], A_BYTES + B_BYTES);
                    tma_load_2d(SA(s), &tma_A, s * TILE_K, ctaRow, &mbar_full[s]);
                    for (int q = 0; q < NUM_NQ; q++)
                        tma_load_2d(SBQ(s,q), &tma_B, ctaCol + q * TILE_NQ, s * TILE_K, &mbar_full[s]);
                }
            }

            /* Signal consumers: tile ready */
            mbarrier_arrive(mbar_tready);

            if (tile_id >= total_tiles) break;

            /* Producer K-loop */
            {
                int ctaRow = tile_info[0];
                int ctaCol = tile_info[1];
                int phase_base = tile * phases_per_tile;
                for (int kt = STAGES; kt < numK; kt++) {
                    int stage = kt % STAGES;
                    int empty_parity = (phase_base + (kt / STAGES) + 1) & 1;
                    mbarrier_wait_parity(&mbar_empty[stage], empty_parity);
                    mbarrier_arrive_expect_tx(&mbar_full[stage], A_BYTES + B_BYTES);
                    tma_load_2d(SA(stage), &tma_A, kt * TILE_K, ctaRow, &mbar_full[stage]);
                    for (int q = 0; q < NUM_NQ; q++)
                        tma_load_2d(SBQ(stage,q), &tma_B, ctaCol + q * TILE_NQ, kt * TILE_K, &mbar_full[stage]);
                }
            }
        }

    } else {
        /* =============================================================
         * CONSUMER — persistent tile loop
         * ============================================================= */
        asm volatile("setmaxnreg.inc.sync.aligned.u32 240;\n");

        int consumer_id = wg_id - 1;
        int local_tid = tid - wg_id * WG_SIZE;
        int a_row_offset = consumer_id * 64 * TILE_K * 2;

        for (int tile = 0; ; tile++) {
            /* Wait for producer to signal tile ready */
            mbarrier_wait_parity(mbar_tready, tile & 1);

            if (tile_info[2] >= total_tiles) break;

            int ctaRow = tile_info[0];
            int ctaCol = tile_info[1];
            int phase_base = tile * phases_per_tile;

            /* Zero accumulators */
            float acc0[32], acc1[32], acc2[32], acc3[32];
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                acc0[i] = 0.0f; acc1[i] = 0.0f;
                acc2[i] = 0.0f; acc3[i] = 0.0f;
            }

            /* K-loop */
            for (int kt = 0; kt < numK; kt++) {
                int stage = kt % STAGES;
                int parity = (phase_base + kt / STAGES) & 1;

                mbarrier_wait_parity(&mbar_full[stage], parity);

                uint64_t da  = make_desc(SA(stage) + a_row_offset, TILE_K);
                uint64_t db0 = make_desc(SBQ(stage,0), TILE_NQ);
                uint64_t db1 = make_desc(SBQ(stage,1), TILE_NQ);
                uint64_t db2 = make_desc(SBQ(stage,2), TILE_NQ);
                uint64_t db3 = make_desc(SBQ(stage,3), TILE_NQ);

                int sd = (kt == 0) ? 0 : 1;
                wgmma_fence();
                #pragma unroll
                for (int ks = 0; ks < 4; ks++) {
                    uint64_t dak  = desc_advance(da,  ks * 2);
                    uint64_t db0k = desc_advance(db0, ks * 128);
                    uint64_t db1k = desc_advance(db1, ks * 128);
                    uint64_t db2k = desc_advance(db2, ks * 128);
                    uint64_t db3k = desc_advance(db3, ks * 128);
                    int s = (ks == 0) ? sd : 1;
                    wgmma_m64n64k16(acc0, dak, db0k, s);
                    wgmma_m64n64k16(acc1, dak, db1k, s);
                    wgmma_m64n64k16(acc2, dak, db2k, s);
                    wgmma_m64n64k16(acc3, dak, db3k, s);
                }
                wgmma_commit();

                if (kt >= 1) {
                    wgmma_wait1();
                    if (local_tid == 0) {
                        mbarrier_arrive(&mbar_empty[(kt - 1) % STAGES]);
                    }
                }
            }

            /* Drain */
            wgmma_wait0();
            if (local_tid == 0) {
                if (numK >= 1)
                    mbarrier_arrive(&mbar_empty[(numK - 1) % STAGES]);
            }

            /* Epilogue: write accumulators to SMEM, then TMA store to GMEM.
             * Reuse stage 0's SMEM buffer (48KB) for epilogue staging.
             * Each consumer stores 4 N-quarters (4 × 64×64 × 2B = 32KB). */
            {
                int consumer_row = ctaRow + consumer_id * 64;
                /* Use stage 0 SMEM for epilogue: consumer 0 uses first 32KB,
                 * consumer 1 uses next 32KB (within B portion of stage 0+1) */
                __nv_bfloat16* epi_smem = (__nv_bfloat16*)(smem + consumer_id * 32768);

                /* Store each N-quarter: acc → SMEM → TMA store → GMEM.
                 * Each quarter: 64 rows × 64 cols × 2B = 8KB in SMEM. */
                #define EPI_STORE_Q(ACC, Q) do { \
                    __nv_bfloat16* smem_q = epi_smem + (Q) * (TILE_NQ * 64); \
                    store_acc_to_smem(smem_q, ACC, local_tid); \
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); \
                    if (local_tid == 0) \
                        tma_store_2d(&tma_C, ctaCol + (Q) * TILE_NQ, consumer_row, smem_q); \
                    tma_store_commit(); \
                } while(0)

                EPI_STORE_Q(acc0, 0);
                EPI_STORE_Q(acc1, 1);
                EPI_STORE_Q(acc2, 2);
                EPI_STORE_Q(acc3, 3);

                #undef EPI_STORE_Q

                /* Wait for all TMA stores to complete before reusing SMEM */
                tma_store_wait();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            }
        }
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
    CUtensorMap tma_A, tma_B, tma_C;
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
        /* B is K×N row-major (N contiguous). TMA dims [N, K], box [TILE_NQ=64, TILE_K].
         * 128B swizzle: 64 bf16 × 2 bytes = 128 bytes ✓
         * Four TMA loads per stage: one per N-quarter. */
        cuuint64_t dims[2] = {(cuuint64_t)N, (cuuint64_t)K};
        cuuint64_t str[1]  = {(cuuint64_t)(N * 2)};
        cuuint32_t box[2]  = {(cuuint32_t)TILE_NQ, (cuuint32_t)TILE_K};
        cuuint32_t el[2]   = {1, 1};
        CUresult r = s_encodeTiled(&tma_B, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
            (void*)B, dims, str, box, el,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) return -3;
    }
    {
        /* C is M×N row-major (N contiguous). TMA store: dims [N, M], box [64, 64].
         * No swizzle for store — SMEM is plain row-major. */
        cuuint64_t dims[2] = {(cuuint64_t)N, (cuuint64_t)M};
        cuuint64_t str[1]  = {(cuuint64_t)(N * 2)};
        cuuint32_t box[2]  = {(cuuint32_t)TILE_NQ, 64};
        cuuint32_t el[2]   = {1, 1};
        CUresult r = s_encodeTiled(&tma_C, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
            (void*)C, dims, str, box, el,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) return -4;
    }

    /* Persistent launch: 1 block per SM */
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    if (num_sms <= 0) num_sms = 132;

    int grid_m = (M + TILE_M - 1) / TILE_M;
    int grid_n = (N + TILE_N - 1) / TILE_N;
    int total_tiles = grid_m * grid_n;
    int num_blocks = (total_tiles < num_sms) ? total_tiles : num_sms;

    /* Tile counter (allocated once, reset per launch) */
    static unsigned* s_tile_counter = nullptr;
    if (!s_tile_counter) cudaMalloc(&s_tile_counter, sizeof(unsigned));
    cudaMemsetAsync(s_tile_counter, 0, sizeof(unsigned), stream);

    cudaFuncSetAttribute(matmul_wgmma_tma,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
    matmul_wgmma_tma<<<num_blocks, THREADS, SMEM_TOTAL, stream>>>(
        C, M, N, K, tma_A, tma_B, tma_C,
        grid_m, grid_n, total_tiles, s_tile_counter);

    return 0;
}
