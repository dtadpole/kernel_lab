/*
 * Sample BF16 matmul — simplified WGMMA + TMA + warp specialization.
 *
 * Based on the peak-cuda v27 architecture, stripped down for readability:
 *   - No template/dual-tile dispatch (fixed TILE_N)
 *   - No vectorized epilogue (simple per-element stores)
 *   - No GROUP_M swizzle (linear tile mapping)
 *
 * Architecture: 3 warpgroups (384 threads):
 *   WG0 = producer (TMA loads)
 *   WG1 = consumer0 (WGMMA rows 0-63)
 *   WG2 = consumer1 (WGMMA rows 64-127)
 *
 * Tunable parameters (via #ifndef / autotune.yaml):
 *   TILE_N  — N-dimension tile size (must be multiple of 64)
 *   STAGES  — pipeline depth (2-5)
 *
 * Fixed: TILE_M=128, TILE_K=64 (WGMMA constraints).
 */
#include <cuda_bf16.h>
#include <cuda.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

/* --- Tunable parameters --- */
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef STAGES
#define STAGES 4
#endif

/* --- Fixed parameters (WGMMA constraints) --- */
#define TILE_M      128
#define TILE_K      64
#define WG_SIZE     128
#define NUM_WG      3
#define THREADS     (NUM_WG * WG_SIZE)  /* 384 */

#define A_BYTES     (TILE_M * TILE_K * 2)   /* 16384 */
#define B_BYTES     (TILE_N * TILE_K * 2)
#define STAGE_BYTES (A_BYTES + B_BYTES)

/* Number of N-quarters (64-wide WGMMA units) */
#define NQ          (TILE_N / 64)

/* Epilogue SMEM: 64×72 BF16 buffer per consumer (72 = 64+8 padding for bank conflicts) */
#define EPI_STRIDE    72
#define EPI_BUF_BYTES (64 * EPI_STRIDE * 2)   /* 9216 bytes per consumer */

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
 * WGMMA helpers
 * ========================================================================= */
__device__ __forceinline__
uint64_t make_desc(const void* smem_ptr, int tile_inner) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    int sbo_16B = (8 * tile_inner * 2) >> 4;
    uint64_t desc = 0;
    desc |= (uint64_t)((addr >> 4) & 0x3FFF);
    desc |= (uint64_t)(1) << 16;
    desc |= (uint64_t)(sbo_16B & 0x3FFF) << 32;
    desc |= (uint64_t)(1) << 62;
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
 * Epilogue — SMEM-buffered vectorized coalesced stores (128-bit / uint4).
 *
 * 1. Each thread scatters its WGMMA registers into a 64×72 SMEM buffer
 * 2. bar.sync within the warpgroup
 * 3. Each thread reads a coalesced 128-bit (8 BF16) chunk from SMEM
 * 4. Writes 128-bit to global memory (fully coalesced)
 * ========================================================================= */
__device__ __forceinline__
void store_acc_n64_smem(__nv_bfloat16* C, float (&acc)[32],
                        int ctaRow, int ctaCol, int M, int N,
                        int local_tid, __nv_bfloat16* epi_buf, int bar_id,
                        int last) {
    const int warp = local_tid / 32;
    const int lane = local_tid % 32;

    /* Step 1: scatter registers → SMEM in WGMMA layout */
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        const int col = (lane % 4) * 2 + p * 8;
        const int row0 = warp * 16 + lane / 4;
        const int row8 = row0 + 8;
        *(__nv_bfloat162*)(epi_buf + row0 * EPI_STRIDE + col) = __halves2bfloat162(
            __float2bfloat16(acc[4*p + 0]), __float2bfloat16(acc[4*p + 1]));
        *(__nv_bfloat162*)(epi_buf + row8 * EPI_STRIDE + col) = __halves2bfloat162(
            __float2bfloat16(acc[4*p + 2]), __float2bfloat16(acc[4*p + 3]));
    }

    /* Step 2: sync within warpgroup */
    asm volatile("bar.sync %0, 128;\n" :: "r"(bar_id));

    /* Step 3-4: coalesced 128-bit reads from SMEM → 128-bit writes to global */
    #pragma unroll
    for (int iter = 0; iter < 4; iter++) {
        const int row = iter * 16 + warp * 4 + lane / 8;
        const int col = (lane % 8) * 8;
        const int gRow = ctaRow + row;
        const int gCol = ctaCol + col;
        if (gRow < M && gCol + 7 < N) {
            uint4 val = *(const uint4*)((const char*)epi_buf + (row * EPI_STRIDE + col) * 2);
            *(uint4*)((char*)C + ((size_t)gRow * N + gCol) * 2) = val;
        }
    }

    if (!last) {
        asm volatile("bar.sync %0, 128;\n" :: "r"(bar_id));
    }
}

/* =========================================================================
 * Main kernel
 * ========================================================================= */
/* GROUP_M tile swizzle: groups M-tiles so nearby blocks share B data in L2 */
__device__ __forceinline__
void get_tile_coords(int tile_id, int grid_m, int grid_n,
                     int& ctaRow, int& ctaCol) {
    const int GROUP_M  = 12;
    const int group_id = tile_id / (GROUP_M * grid_n);
    const int first_m  = group_id * GROUP_M;
    const int group_sz = (first_m + GROUP_M <= grid_m) ? GROUP_M : (grid_m - first_m);
    const int local_id = tile_id % (GROUP_M * grid_n);
    ctaRow = (first_m + local_id % group_sz) * TILE_M;
    ctaCol = (local_id / group_sz) * TILE_N;
}

__global__ void __launch_bounds__(THREADS, 1)
matmul_wgmma(
    __nv_bfloat16* __restrict__ C, int M, int N, int K,
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    int grid_m, int grid_n, int total_tiles)
{
    /* SMEM layout: [STAGES × (A+B)] [mbar_full] [mbar_empty] [epi_buf×2] */
    constexpr int MBAR_FULL_OFF  = STAGES * STAGE_BYTES;
    constexpr int MBAR_EMPTY_OFF = MBAR_FULL_OFF + 128;
    constexpr int EPI_OFF        = MBAR_EMPTY_OFF + 128;

    extern __shared__ char smem[];
    uint64_t* mbar_full  = (uint64_t*)(smem + MBAR_FULL_OFF);
    uint64_t* mbar_empty = (uint64_t*)(smem + MBAR_EMPTY_OFF);

    const int tid   = threadIdx.x;
    const int wg_id = tid / WG_SIZE;
    const int numK  = (K + TILE_K - 1) / TILE_K;
    const int phases_per_tile = (numK + STAGES - 1) / STAGES;
    const int last_stage = (numK - 1) % STAGES;

    /* Init barriers */
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; s++) {
            mbarrier_init(&mbar_full[s],  1);
            mbarrier_init(&mbar_empty[s], 2);  /* 2 consumer WGs */
        }
    }
    __syncthreads();

    if (wg_id == 0) {
        /* ============= PRODUCER (WG0) ============= */
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n");
        if (tid != 0) return;

        /* Persistent loop: each block processes multiple tiles */
        for (int local_k = 0; ; local_k++) {
            const int tile = (int)blockIdx.x + local_k * (int)gridDim.x;
            if (tile >= total_tiles) break;

            int ctaRow, ctaCol;
            get_tile_coords(tile, grid_m, grid_n, ctaRow, ctaCol);

            /* Wait for consumers to finish previous tile's last stage */
            if (local_k > 0) {
                const int prev_phase = ((local_k - 1) * phases_per_tile + (numK - 1) / STAGES) & 1;
                mbarrier_wait_parity(&mbar_empty[last_stage], prev_phase);
            }

            /* Prefill */
            const int prefill = (numK < STAGES) ? numK : STAGES;
            for (int s = 0; s < prefill; s++) {
                mbarrier_arrive_expect_tx(&mbar_full[s], A_BYTES + B_BYTES);
                tma_load_2d(smem + s * STAGE_BYTES, &tma_A,
                            s * TILE_K, ctaRow, &mbar_full[s]);
                for (int q = 0; q < NQ; q++)
                    tma_load_2d(smem + s * STAGE_BYTES + A_BYTES + q * 64 * TILE_K * 2,
                                &tma_B, ctaCol + q * 64, s * TILE_K, &mbar_full[s]);
            }

            /* Steady state */
            const int phase_base = local_k * phases_per_tile;
            for (int kt = STAGES; kt < numK; kt++) {
                const int stage = kt % STAGES;
                const int empty_parity = (phase_base + (kt / STAGES) + 1) & 1;
                mbarrier_wait_parity(&mbar_empty[stage], empty_parity);
                mbarrier_arrive_expect_tx(&mbar_full[stage], A_BYTES + B_BYTES);
                tma_load_2d(smem + stage * STAGE_BYTES, &tma_A,
                            kt * TILE_K, ctaRow, &mbar_full[stage]);
                for (int q = 0; q < NQ; q++)
                    tma_load_2d(smem + stage * STAGE_BYTES + A_BYTES + q * 64 * TILE_K * 2,
                                &tma_B, ctaCol + q * 64, kt * TILE_K, &mbar_full[stage]);
            }
        }

    } else {
        /* ============= CONSUMER (WG1, WG2) ============= */
        asm volatile("setmaxnreg.inc.sync.aligned.u32 240;\n");

        const int consumer_id = wg_id - 1;
        const int local_tid   = tid - wg_id * WG_SIZE;
        const int a_row_off   = consumer_id * 64 * TILE_K * 2;

        /* Persistent loop */
        for (int local_k = 0; ; local_k++) {
            const int tile = (int)blockIdx.x + local_k * (int)gridDim.x;
            if (tile >= total_tiles) break;

            int ctaRow, ctaCol;
            get_tile_coords(tile, grid_m, grid_n, ctaRow, ctaCol);

            const int phase_base = local_k * phases_per_tile;

            /* Zero accumulators */
            float acc[NQ][32];
            #pragma unroll
            for (int q = 0; q < NQ; q++)
                #pragma unroll
                for (int i = 0; i < 32; i++)
                    acc[q][i] = 0.0f;

            /* K-loop */
            for (int kt = 0; kt < numK; kt++) {
                const int stage  = kt % STAGES;
                const int parity = (phase_base + kt / STAGES) & 1;

                mbarrier_wait_parity(&mbar_full[stage], parity);

                const char* base = smem + stage * STAGE_BYTES;
                const uint64_t da = make_desc(base + a_row_off, TILE_K);

                const int sd = (kt == 0) ? 0 : 1;
                wgmma_fence();

                #pragma unroll
                for (int ks = 0; ks < 4; ks++) {
                    const uint64_t dak = desc_advance(da, ks * 2);
                    const int s = (ks == 0) ? sd : 1;
                    #pragma unroll
                    for (int q = 0; q < NQ; q++) {
                        uint64_t db = make_desc(base + A_BYTES + q * 64 * TILE_K * 2, 64);
                        wgmma_m64n64k16(acc[q], dak, desc_advance(db, ks * 128), s);
                    }
                }
                wgmma_commit();

                if (kt >= 1) {
                    wgmma_wait1();
                    if (local_tid == 0)
                        mbarrier_arrive(&mbar_empty[(kt - 1) % STAGES]);
                }
            }

            wgmma_wait0();
            if (local_tid == 0)
                mbarrier_arrive(&mbar_empty[(numK - 1) % STAGES]);

            /* Epilogue: SMEM-buffered vectorized stores */
            const int consumer_row = ctaRow + consumer_id * 64;
            __nv_bfloat16* epi_buf = (__nv_bfloat16*)(smem + EPI_OFF + consumer_id * EPI_BUF_BYTES);
            const int bar_id = consumer_id + 1;
            #pragma unroll
            for (int q = 0; q < NQ; q++)
                store_acc_n64_smem(C, acc[q], consumer_row, ctaCol + q * 64,
                                   M, N, local_tid, epi_buf, bar_id, q == NQ - 1);
        }
    }
}

/* =========================================================================
 * Host: TMA encoder + kernel dispatch
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

    /* Encode TMA descriptors */
    CUtensorMap tma_A, tma_B;
    {
        cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)M};
        cuuint64_t str[1]  = {(cuuint64_t)(K * 2)};
        cuuint32_t box[2]  = {(cuuint32_t)TILE_K, (cuuint32_t)TILE_M};
        cuuint32_t el[2]   = {1, 1};
        if (s_encodeTiled(&tma_A, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
            (void*)A, dims, str, box, el,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) return -2;
    }
    {
        cuuint64_t dims[2] = {(cuuint64_t)N, (cuuint64_t)K};
        cuuint64_t str[1]  = {(cuuint64_t)(N * 2)};
        cuuint32_t box[2]  = {64, (cuuint32_t)TILE_K};  /* TMA loads 64-wide B tiles */
        cuuint32_t el[2]   = {1, 1};
        if (s_encodeTiled(&tma_B, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
            (void*)B, dims, str, box, el,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) return -3;
    }

    /* Launch */
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    if (num_sms <= 0) num_sms = 132;

    const int grid_m = (M + TILE_M - 1) / TILE_M;
    const int grid_n = (N + TILE_N - 1) / TILE_N;
    const int total_tiles = grid_m * grid_n;
    const int num_blocks = (total_tiles < num_sms) ? total_tiles : num_sms;
    constexpr int SMEM = STAGES * STAGE_BYTES + 256 + 2 * EPI_BUF_BYTES;

    cudaFuncSetAttribute(matmul_wgmma,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    matmul_wgmma<<<num_blocks, THREADS, SMEM, stream>>>(
        C, M, N, K, tma_A, tma_B, grid_m, grid_n, total_tiles);

    return 0;
}
