/*
 * BF16 matmul for SM90 — WGMMA + TMA loads.
 *
 * 1 warpgroup (128 threads), CTA tile 128×128, TILE_K=64.
 * TMA cp.async.bulk.tensor.2d for global→shared (zero instruction overhead).
 * mbarrier for TMA completion tracking.
 * 128B swizzle SMEM (handled by TMA hardware).
 * Pre-transposed B (N×K layout).
 * FP32 accumulation → BF16 output.
 */
#include <cuda_bf16.h>
#include <cuda.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

#define TILE_M    128
#define TILE_N    128
#define TILE_K    64
#define THREADS   128
#define STAGES    3

#define A_BYTES   (TILE_M * TILE_K * 2)   /* 16384 */
#define B_BYTES   (TILE_N * TILE_K * 2)   /* 16384 */
#define STAGE_BYTES (A_BYTES + B_BYTES)    /* 32768 */
/* SMEM: 2 stages + 2 mbarriers (16 bytes each, 128-byte aligned) */
#define MBAR_OFFSET (STAGES * STAGE_BYTES)
#define SMEM_TOTAL  (MBAR_OFFSET + 256)   /* 256 for aligned mbarriers */

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

/* =========================================================================
 * WGMMA helpers (same as before)
 * ========================================================================= */
__device__ __forceinline__
uint64_t make_desc(const void* smem_ptr, int tile_k) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    int stride_16B = (8 * tile_k * 2) >> 4;
    uint64_t desc = 0;
    desc |= (uint64_t)((addr >> 4) & 0x3FFF);
    desc |= (uint64_t)(1) << 16;
    desc |= (uint64_t)(stride_16B & 0x3FFF) << 32;
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

__device__ __forceinline__
void wgmma_m64n128k16(float (&d)[64], uint64_t da, uint64_t db, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
        "%64,%65,p,1,1,0,0;\n}\n"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),
          "+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
          "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),
          "+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
          "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),
          "+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
          "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),
          "+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),
          "+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),
          "+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),
          "+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),
          "+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),
          "+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),
          "+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),
          "+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),
          "+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])
        : "l"(da), "l"(db), "r"(scale_d));
}

/* =========================================================================
 * Epilogue
 * ========================================================================= */
__device__ __forceinline__
void store_acc(__nv_bfloat16* C, float (&acc)[64],
               int ctaRow, int ctaCol, int M, int N) {
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row_base = ctaRow + warp * 16 + lane / 4;
    int col_base = ctaCol + (lane % 4) * 2;
    for (int p = 0; p < 16; p++) {
        int col = col_base + p * 8;
        int row0 = row_base, row8 = row_base + 8;
        if (row0 < M && col < N)
            C[row0 * N + col] = __float2bfloat16(acc[4*p]);
        if (row0 < M && col + 1 < N)
            C[row0 * N + col + 1] = __float2bfloat16(acc[4*p + 1]);
        if (row8 < M && col < N)
            C[row8 * N + col] = __float2bfloat16(acc[4*p + 2]);
        if (row8 < M && col + 1 < N)
            C[row8 * N + col + 1] = __float2bfloat16(acc[4*p + 3]);
    }
}

/* =========================================================================
 * Main kernel — TMA loads + WGMMA compute
 * ========================================================================= */
__global__ void __launch_bounds__(THREADS, 1)
matmul_wgmma_tma(
    __nv_bfloat16* __restrict__ C, int M, int N, int K,
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B)
{
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");

    extern __shared__ char smem[];
    char* sA[STAGES], *sB[STAGES];
    for (int s = 0; s < STAGES; s++) {
        sA[s] = smem + s * STAGE_BYTES;
        sB[s] = smem + s * STAGE_BYTES + A_BYTES;
    }
    uint64_t* mbar = (uint64_t*)(smem + MBAR_OFFSET);

    int ctaRow = blockIdx.y * TILE_M;
    int ctaCol = blockIdx.x * TILE_N;
    int tid = threadIdx.x;
    int numK = (K + TILE_K - 1) / TILE_K;

    /* Init mbarriers (only 1 thread) */
    if (tid == 0) {
        for (int s = 0; s < STAGES; s++)
            mbarrier_init(&mbar[s], 1);  /* 1 arrive for TMA */
    }
    __syncthreads();

    float acc0[64], acc1[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) { acc0[i] = 0.0f; acc1[i] = 0.0f; }

    /* Prefill: load first STAGES tiles via TMA */
    int prefill = (numK < STAGES) ? numK : STAGES;
    for (int s = 0; s < prefill; s++) {
        if (tid == 0) {
            mbarrier_arrive_expect_tx(&mbar[s], A_BYTES + B_BYTES);
            /* TMA load A: dims are [K, M], coord = (k_offset, m_offset) */
            tma_load_2d(sA[s], &tma_A, s * TILE_K, ctaRow, &mbar[s]);
            /* TMA load B: dims are [K, N], coord = (k_offset, n_offset) */
            tma_load_2d(sB[s], &tma_B, s * TILE_K, ctaCol, &mbar[s]);
        }
    }

    /* Software-pipelined K-loop with wgmma.wait_group 1.
     *
     * With 3 stages and wait_group 1, we allow 1 outstanding WGMMA group.
     * At iteration kt, we:
     *   1. Wait for TMA load of stage cur to complete
     *   2. Issue WGMMA on stage cur (commit as group)
     *   3. Wait for group from kt-2 to complete (wait_group 1)
     *      → This means stage from kt-2 is now free for TMA reuse
     *   4. Issue TMA load for future K tile into the now-free stage
     *
     * Timeline (3 stages, s0/s1/s2):
     *   kt=0: TMA wait s0, WGMMA s0 (g0), commit
     *   kt=1: TMA wait s1, WGMMA s1 (g1), commit, wait_group 1 (nothing), TMA→s2
     *   kt=2: TMA wait s2, WGMMA s2 (g0), commit, wait_group 1 (g0 done→s0 free), TMA→s0
     *   kt=3: TMA wait s0, WGMMA s0 (g1), commit, wait_group 1 (g1 done→s1 free), TMA→s1
     *   ...
     * Epilogue: wait_group 0 (drain all)
     */
    for (int kt = 0; kt < numK; kt++) {
        int cur = kt % STAGES;

        /* 1. Wait for TMA load to complete */
        mbarrier_wait_parity(&mbar[cur], (kt / STAGES) & 1);

        /* 2. Issue WGMMA compute on this stage */
        uint64_t da0 = make_desc(sA[cur], TILE_K);
        uint64_t da1 = make_desc(sA[cur] + 64 * TILE_K * 2, TILE_K);
        uint64_t db = make_desc(sB[cur], TILE_K);

        int sd = (kt == 0) ? 0 : 1;
        wgmma_fence();
        for (int ks = 0; ks < 4; ks++) {
            uint64_t dak0 = desc_advance(da0, ks * 2);
            uint64_t dak1 = desc_advance(da1, ks * 2);
            uint64_t dbk  = desc_advance(db,  ks * 2);
            wgmma_m64n128k16(acc0, dak0, dbk, (ks == 0) ? sd : 1);
            wgmma_m64n128k16(acc1, dak1, dbk, (ks == 0) ? sd : 1);
        }
        wgmma_commit();

        /* 3. Wait for the WGMMA group from 2 iterations ago.
         * This ensures the stage that will be reused for TMA is no longer
         * being read by WGMMA. With 3 stages, the stage freed is (kt-2)%3.
         * wait_group 1 = wait until at most 1 group is outstanding. */
        if (kt >= 1) {
            asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        }

        /* 4. Issue TMA load for a future K tile.
         * After wait_group 1, WGMMA from kt-1 has completed — stage is safe.
         * No __syncthreads needed: 1 warpgroup = wgmma.wait syncs all threads. */
        if (kt >= 1) {
            int free_stage = (kt - 1) % STAGES;
            int futK = (kt - 1) + STAGES;
            if (futK < numK && tid == 0) {
                mbarrier_arrive_expect_tx(&mbar[free_stage], A_BYTES + B_BYTES);
                tma_load_2d(sA[free_stage], &tma_A, futK * TILE_K, ctaRow, &mbar[free_stage]);
                tma_load_2d(sB[free_stage], &tma_B, futK * TILE_K, ctaCol, &mbar[free_stage]);
            }
        }
    }

    /* Drain: wait for ALL outstanding WGMMA groups to complete.
     * Also issue the last pending TMA if needed. */
    wgmma_wait0();
    /* The last iteration's freed stage gets a TMA load here if needed */
    if (numK >= 1) {
        int free_stage = (numK - 1) % STAGES;
        int futK = (numK - 1) + STAGES;
        /* futK >= numK always, so no more loads needed. Just drain. */
    }

    /* Epilogue */
    store_acc(C, acc0, ctaRow, ctaCol, M, N);
    store_acc(C, acc1, ctaRow + 64, ctaCol, M, N);

    /* Invalidate barriers */
    if (tid == 0) {
        for (int s = 0; s < STAGES; s++)
            mbarrier_inval(&mbar[s]);
    }
}

/* =========================================================================
 * B transpose kernel
 * ========================================================================= */
__global__ void transpose_bf16(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int K, int N)
{
    __shared__ __nv_bfloat16 tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (y < K && x < N)
        tile[threadIdx.y][threadIdx.x] = src[y * N + x];
    else
        tile[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
    __syncthreads();
    int ox = blockIdx.y * 32 + threadIdx.x;
    int oy = blockIdx.x * 32 + threadIdx.y;
    if (oy < N && ox < K)
        dst[oy * K + ox] = tile[threadIdx.x][threadIdx.y];
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

    /* Pre-transpose B: K×N → Bt: N×K */
    __nv_bfloat16* Bt;
    cudaMalloc(&Bt, (size_t)K * N * sizeof(__nv_bfloat16));
    {
        dim3 tb(32, 32);
        dim3 tg((N + 31) / 32, (K + 31) / 32);
        transpose_bf16<<<tg, tb, 0, stream>>>(B, Bt, K, N);
    }

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
        cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)N};
        cuuint64_t str[1]  = {(cuuint64_t)(K * 2)};
        cuuint32_t box[2]  = {(cuuint32_t)TILE_K, (cuuint32_t)TILE_N};
        cuuint32_t el[2]   = {1, 1};
        CUresult r = s_encodeTiled(&tma_B, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
            (void*)Bt, dims, str, box, el,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) return -3;
    }

    /* Launch */
    dim3 block(THREADS);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    cudaFuncSetAttribute(matmul_wgmma_tma,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
    matmul_wgmma_tma<<<grid, block, SMEM_TOTAL, stream>>>(
        C, M, N, K, tma_A, tma_B);

    cudaFree(Bt);
    return 0;
}
