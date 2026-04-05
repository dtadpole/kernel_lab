/*
 * Flash Attention forward pass — BF16, SM90 WGMMA kernel (2-consumer-WG).
 *
 * Architecture: 384 threads = 3 warp groups (1 producer + 2 consumer).
 *   WG0 (warps 0-3): Producer — TMA loads for Q, K, V
 *   WG1 (warps 4-7): Consumer 1 — MMA for Q rows 0-63
 *   WG2 (warps 8-11): Consumer 2 — MMA for Q rows 64-127
 *
 * TMA with SWIZZLE_128B, split-DIM (cols 0-63, 64-127).
 * BLOCK_Q=128, BLOCK_KV=128, DIM=128.
 *
 * SMEM layout (160KB):
 *   Q_lo(16KB) Q_hi(16KB)
 *   K0_lo(16KB) K0_hi(16KB) K1_lo(16KB) K1_hi(16KB)
 *   V0_lo(16KB) V0_hi(16KB) V1_lo(16KB) V1_hi(16KB)
 *   mbarriers: K_full[2] K_empty[2] V_full[2] V_empty[2] Q_full Q_empty
 *
 * QK GEMM: m64n128k16 SS mode per consumer WG (64×128 output).
 * PV GEMM: m64n64k16 RS mode, 8 k-steps per half (V_lo → O_lo, V_hi → O_hi).
 *
 * Intra-WG overlap (matches CuTe DSL FA4 pattern):
 *   Prologue: QK[0] → softmax → pack P
 *   Mainloop: issue QK[n] + PV[n-1] concurrently (2 WGMMA groups in flight)
 *             wait_group<1> (QK done) → softmax overlaps PV execution
 *             wait_group<0> (PV done) → rescale O → pack P
 *   Epilogue: PV[last] → finalize → store O
 *
 * Scheduler barriers: bar 2 (WG1), bar 3 (WG2), 256 threads each.
 * Register redistribution: producer 24, consumer 240.
 *
 * Target: NVIDIA H100 (SM90a). Compile with -gencode arch=compute_90a,code=sm_90a.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cfloat>

__device__ __host__ constexpr
int cdiv(int a, int b) { return (a + b - 1) / b; }

/* --- Fast math --------------------------------------------------------- */

__device__ __forceinline__
float fast_exp2f(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__
float fast_rcp(float x) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

/* --- mbarrier helpers -------------------------------------------------- */

__device__ __forceinline__
void mbarrier_init(uint64_t* mbar, unsigned count) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(addr), "r"(count));
}

__device__ __forceinline__
void mbarrier_arrive_expect_tx(uint64_t* mbar, unsigned tx_bytes) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                 :: "r"(addr), "r"(tx_bytes));
}

__device__ __forceinline__
void mbarrier_arrive(uint64_t* mbar) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(addr));
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

/* --- Named barrier helpers (for warp scheduler coordination) ----------- */

__device__ __forceinline__
void named_barrier_arrive(int bar_id, int num_threads) {
    asm volatile("bar.arrive %0, %1;" :: "r"(bar_id), "r"(num_threads));
}

__device__ __forceinline__
void named_barrier_sync(int bar_id, int num_threads) {
    asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(num_threads));
}

/* --- TMA load (G2S) --------------------------------------------------- */

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

/* --- TMA store (S2G) -------------------------------------------------- */

__device__ __forceinline__
void tma_store_2d(const void* tma_desc, uint32_t smem_addr,
                  int coord0, int coord1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];\n"
        :: "l"(tma_desc), "r"(coord0), "r"(coord1), "r"(smem_addr)
        : "memory");
}

__device__ __forceinline__
void cp_async_bulk_commit_group() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

__device__ __forceinline__
void cp_async_bulk_wait_group() {
    asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__
void fence_view_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

/* --- WGMMA synchronization --------------------------------------------- */

__device__ __forceinline__
void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__
void wgmma_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__
void wgmma_wait_group() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

/* --- WGMMA descriptor helpers ------------------------------------------ */

__device__ __forceinline__
uint64_t make_wgmma_desc(uint32_t smem_addr, int stride_bytes) {
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_addr >> 4) & 0x3FFF);
    desc |= (uint64_t)(1) << 16;                               /* LBO = 1 */
    desc |= (uint64_t)(((stride_bytes >> 4) & 0x3FFF)) << 32;
    desc |= (uint64_t)(1) << 62;                                /* SWIZZLE_128B */
    return desc;
}

__device__ __forceinline__
uint64_t make_wgmma_desc_lbo(uint32_t smem_addr, int lbo_bytes, int stride_bytes) {
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_addr >> 4) & 0x3FFF);
    desc |= (uint64_t)(((lbo_bytes >> 4) & 0x3FFF)) << 16;     /* custom LBO */
    desc |= (uint64_t)(((stride_bytes >> 4) & 0x3FFF)) << 32;
    desc |= (uint64_t)(1) << 62;                                /* SWIZZLE_128B */
    return desc;
}

__device__ __forceinline__
uint64_t gmma_desc_advance(uint64_t desc, int offset_16B) {
    uint32_t lo = (uint32_t)desc + (uint32_t)offset_16B;
    uint32_t hi = (uint32_t)(desc >> 32);
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

/* --- WGMMA m64n128k16 SS mode (QK GEMM) ------------------------------- */

__device__ __forceinline__
void wgmma_m64n128k16_f32_bf16(float acc[64], uint64_t desc_a, uint64_t desc_b,
                                int scale_D) {
    asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, %66, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
    "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
    " %8,  %9,  %10, %11, %12, %13, %14, %15, "
    " %16, %17, %18, %19, %20, %21, %22, %23, "
    " %24, %25, %26, %27, %28, %29, %30, %31, "
    " %32, %33, %34, %35, %36, %37, %38, %39, "
    " %40, %41, %42, %43, %44, %45, %46, %47, "
    " %48, %49, %50, %51, %52, %53, %54, %55, "
    " %56, %57, %58, %59, %60, %61, %62, %63},"
    " %64, %65, p, 1, 1, 0, 0;\n"
    "}\n"
    : "+f"(acc[0]),  "+f"(acc[1]),  "+f"(acc[2]),  "+f"(acc[3]),
      "+f"(acc[4]),  "+f"(acc[5]),  "+f"(acc[6]),  "+f"(acc[7]),
      "+f"(acc[8]),  "+f"(acc[9]),  "+f"(acc[10]), "+f"(acc[11]),
      "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]),
      "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]),
      "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]),
      "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]),
      "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31]),
      "+f"(acc[32]), "+f"(acc[33]), "+f"(acc[34]), "+f"(acc[35]),
      "+f"(acc[36]), "+f"(acc[37]), "+f"(acc[38]), "+f"(acc[39]),
      "+f"(acc[40]), "+f"(acc[41]), "+f"(acc[42]), "+f"(acc[43]),
      "+f"(acc[44]), "+f"(acc[45]), "+f"(acc[46]), "+f"(acc[47]),
      "+f"(acc[48]), "+f"(acc[49]), "+f"(acc[50]), "+f"(acc[51]),
      "+f"(acc[52]), "+f"(acc[53]), "+f"(acc[54]), "+f"(acc[55]),
      "+f"(acc[56]), "+f"(acc[57]), "+f"(acc[58]), "+f"(acc[59]),
      "+f"(acc[60]), "+f"(acc[61]), "+f"(acc[62]), "+f"(acc[63])
    : "l"(desc_a), "l"(desc_b), "r"(scale_D));
}

/* --- WGMMA m64n64k16 RS mode (PV GEMM) -------------------------------- */

__device__ __forceinline__
void wgmma_m64n64k16_f32_bf16_RS(float acc[32],
                                  uint32_t a0, uint32_t a1,
                                  uint32_t a2, uint32_t a3,
                                  uint64_t desc_b,
                                  int scale_D) {
    asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, %37, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
    "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
    " %8,  %9,  %10, %11, %12, %13, %14, %15, "
    " %16, %17, %18, %19, %20, %21, %22, %23, "
    " %24, %25, %26, %27, %28, %29, %30, %31},"
    " {%32, %33, %34, %35},"
    " %36, p, 1, 1, 1;\n"
    "}\n"
    : "+f"(acc[0]),  "+f"(acc[1]),  "+f"(acc[2]),  "+f"(acc[3]),
      "+f"(acc[4]),  "+f"(acc[5]),  "+f"(acc[6]),  "+f"(acc[7]),
      "+f"(acc[8]),  "+f"(acc[9]),  "+f"(acc[10]), "+f"(acc[11]),
      "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]),
      "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]),
      "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]),
      "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]),
      "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "l"(desc_b), "r"(scale_D));
}

/* --- WGMMA m64n128k16 RS mode (PV GEMM, full DIM=128) ------------------- */

__device__ __forceinline__
void wgmma_m64n128k16_f32_bf16_RS(float acc[64],
                                   uint32_t a0, uint32_t a1,
                                   uint32_t a2, uint32_t a3,
                                   uint64_t desc_b,
                                   int scale_D) {
    asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, %69, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
    "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
    " %8,  %9,  %10, %11, %12, %13, %14, %15, "
    " %16, %17, %18, %19, %20, %21, %22, %23, "
    " %24, %25, %26, %27, %28, %29, %30, %31, "
    " %32, %33, %34, %35, %36, %37, %38, %39, "
    " %40, %41, %42, %43, %44, %45, %46, %47, "
    " %48, %49, %50, %51, %52, %53, %54, %55, "
    " %56, %57, %58, %59, %60, %61, %62, %63},"
    " {%64, %65, %66, %67},"
    " %68, p, 1, 1, 1;\n"
    "}\n"
    : "+f"(acc[0]),  "+f"(acc[1]),  "+f"(acc[2]),  "+f"(acc[3]),
      "+f"(acc[4]),  "+f"(acc[5]),  "+f"(acc[6]),  "+f"(acc[7]),
      "+f"(acc[8]),  "+f"(acc[9]),  "+f"(acc[10]), "+f"(acc[11]),
      "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]),
      "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]),
      "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]),
      "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]),
      "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31]),
      "+f"(acc[32]), "+f"(acc[33]), "+f"(acc[34]), "+f"(acc[35]),
      "+f"(acc[36]), "+f"(acc[37]), "+f"(acc[38]), "+f"(acc[39]),
      "+f"(acc[40]), "+f"(acc[41]), "+f"(acc[42]), "+f"(acc[43]),
      "+f"(acc[44]), "+f"(acc[45]), "+f"(acc[46]), "+f"(acc[47]),
      "+f"(acc[48]), "+f"(acc[49]), "+f"(acc[50]), "+f"(acc[51]),
      "+f"(acc[52]), "+f"(acc[53]), "+f"(acc[54]), "+f"(acc[55]),
      "+f"(acc[56]), "+f"(acc[57]), "+f"(acc[58]), "+f"(acc[59]),
      "+f"(acc[60]), "+f"(acc[61]), "+f"(acc[62]), "+f"(acc[63])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "l"(desc_b), "r"(scale_D));
}

/* --- Pack BF16x2 ------------------------------------------------------- */

__device__ __forceinline__
uint32_t pack_bf16(float a, float b) {
    uint32_t result;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(result) : "f"(a), "f"(b));
    return result;
}

/* ======================================================================
 *  2-Consumer-WG Flash Attention Kernel
 *
 *  WG0 (warps 0-3): Producer — TMA loads Q, K, V
 *  WG1 (warps 4-7): Consumer 1 — processes Q rows 0-63
 *  WG2 (warps 8-11): Consumer 2 — processes Q rows 64-127
 *
 *  Pipeline: K and V are double-buffered with full+empty mbarrier protocol.
 *  Scheduler barriers: bar 2 (WG1), bar 3 (WG2), 256 threads.
 * ====================================================================== */

template<int BLOCK_Q, int BLOCK_KV, int DIM>
__launch_bounds__(384, 1)
__global__
void flash_attention_2wg(
    const nv_bfloat16 *Q_unused,
    const nv_bfloat16 *K_unused,
    const nv_bfloat16 *V_unused,
    nv_bfloat16 *O,
    int B, int S, int H,
    int len_q, int len_kv,
    int is_causal,
    const CUtensorMap *tma_Q_arr,
    const CUtensorMap *tma_K_arr,
    const CUtensorMap *tma_V_arr,
    const CUtensorMap *tma_O_arr)
{
    constexpr int WG_SIZE    = 128;
    constexpr int HALF_DIM   = DIM / 2;          /* 64 */
    constexpr int HALF_Q     = BLOCK_Q * HALF_DIM * (int)sizeof(nv_bfloat16);  /* 16KB */
    constexpr int HALF_KV    = BLOCK_KV * HALF_DIM * (int)sizeof(nv_bfloat16); /* 16KB */
    constexpr int STRIDE     = 8 * HALF_DIM * (int)sizeof(nv_bfloat16);        /* 1024B */
    constexpr int KV_TILE_BYTES = 2 * HALF_KV;    /* 32KB per full K/V tile */
    constexpr int Q_TILE_BYTES  = 2 * HALF_Q;     /* 32KB */
    constexpr int V_KS_ADVANCE  = 2 * STRIDE / 16; /* 128 in 16B units */

    /* SMEM layout: Q(32KB) + K×2(64KB) + V×2(64KB) + mbarriers = 160KB + 80B */
    extern __shared__ nv_bfloat16 smem[];
    const uint32_t sb = __cvta_generic_to_shared(smem);

    const uint32_t Q_lo   = sb;
    const uint32_t Q_hi   = sb + HALF_Q;
    const uint32_t K0_lo  = sb + 2*HALF_Q;
    const uint32_t K0_hi  = sb + 2*HALF_Q + HALF_KV;
    const uint32_t K1_lo  = sb + 2*HALF_Q + 2*HALF_KV;
    const uint32_t K1_hi  = sb + 2*HALF_Q + 3*HALF_KV;
    const uint32_t V0_lo  = sb + 2*HALF_Q + 4*HALF_KV;
    const uint32_t V0_hi  = sb + 2*HALF_Q + 5*HALF_KV;
    const uint32_t V1_lo  = sb + 2*HALF_Q + 6*HALF_KV;
    const uint32_t V1_hi  = sb + 2*HALF_Q + 7*HALF_KV;

    constexpr int DATA_BYTES = 2*HALF_Q + 8*HALF_KV;  /* 160KB */
    uint64_t *mbar = (uint64_t*)((char*)smem + DATA_BYTES);
    /* mbar[0..1] = K_full[2], mbar[2..3] = K_empty[2],
     * mbar[4..5] = V_full[2], mbar[6..7] = V_empty[2],
     * mbar[8] = Q_full, mbar[9] = Q_empty */
    uint64_t *K_full  = &mbar[0];
    uint64_t *K_empty = &mbar[2];
    uint64_t *V_full  = &mbar[4];
    uint64_t *V_empty = &mbar[6];
    uint64_t *Q_full_mbar  = &mbar[8];

    const int tid     = threadIdx.x;
    const int wg_id   = tid / WG_SIZE;       /* 0=producer, 1,2=consumer */
    const int wg_tid  = tid % WG_SIZE;       /* 0-127 within WG */
    const int warp_id = tid / 32;            /* 0-11 */
    const int lane_id = tid % 32;

    /* Block mapping (BLOCK_Q=128) */
    const int bid = blockIdx.x;
    const int num_q_blocks = cdiv(len_q, BLOCK_Q);
    const int bs_id = bid / num_q_blocks;
    const int q_block_id = bid % num_q_blocks;
    const int batch_id = bs_id / H;
    const int head_id  = bs_id % H;
    const int bh_id = batch_id * H + head_id;

    const CUtensorMap *tma_Q = &tma_Q_arr[bh_id];
    const CUtensorMap *tma_K = &tma_K_arr[bh_id];
    const CUtensorMap *tma_V = &tma_V_arr[bh_id];
    const CUtensorMap *tma_O = &tma_O_arr[bh_id];

    (void)O;  /* O is written via TMA S2G using tma_O descriptor */

    const int max_kv_iter = is_causal
        ? min(cdiv(len_kv, BLOCK_KV), cdiv((q_block_id + 1) * BLOCK_Q, BLOCK_KV))
        : cdiv(len_kv, BLOCK_KV);

    const float softmax_scale_log2 =
        rsqrtf(static_cast<float>(DIM)) * 1.4426950408889634f;

    /* ---- Initialize mbarriers (thread 0) ---- */
    if (tid == 0) {
        for (int s = 0; s < 2; s++) {
            mbarrier_init(&K_full[s],  1);  /* 1 TMA thread arrives */
            mbarrier_init(&K_empty[s], 2);  /* 2 consumer WGs release */
            mbarrier_init(&V_full[s],  1);
            mbarrier_init(&V_empty[s], 2);
        }
        mbarrier_init(Q_full_mbar, 1);
        /* Pre-arrive on empty barriers: mark all stages as initially free.
         * arrive_count=2, so we need 2 pre-arrives to flip phase. */
        for (int s = 0; s < 2; s++) {
            mbarrier_arrive(&K_empty[s]);
            mbarrier_arrive(&K_empty[s]);
            mbarrier_arrive(&V_empty[s]);
            mbarrier_arrive(&V_empty[s]);
        }
    }
    __syncthreads();

    /* ---- Register redistribution ---- */
    if (wg_id == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n");
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 240;\n");
    }

    /* ================================================================
     *  PRODUCER PATH (WG0, warps 0-3)
     * ================================================================ */
    if (wg_id == 0) {
        if (wg_tid == 0) {
            /* Load Q (single stage) */
            mbarrier_arrive_expect_tx(Q_full_mbar, Q_TILE_BYTES);
            tma_load_2d((void*)(Q_lo), tma_Q, 0,  q_block_id * BLOCK_Q, Q_full_mbar);
            tma_load_2d((void*)(Q_hi), tma_Q, 64, q_block_id * BLOCK_Q, Q_full_mbar);

            /* Pipeline: load K and V tiles for consumers.
             * K leads V by one pipeline slot (K[n] loaded with V[n-1]).
             * Stage alternates 0, 1, 0, 1... */
            int k_stage = 0;
            int v_stage = 0;
            int k_empty_phase = 0;
            int v_empty_phase = 0;

            for (int n = 0; n < max_kv_iter; n++) {
                /* Load K[n] */
                mbarrier_wait_parity(&K_empty[k_stage], k_empty_phase);
                mbarrier_arrive_expect_tx(&K_full[k_stage], KV_TILE_BYTES);
                uint32_t ks_lo = (k_stage == 0) ? K0_lo : K1_lo;
                uint32_t ks_hi = (k_stage == 0) ? K0_hi : K1_hi;
                tma_load_2d((void*)ks_lo, tma_K, 0,  n * BLOCK_KV, &K_full[k_stage]);
                tma_load_2d((void*)ks_hi, tma_K, 64, n * BLOCK_KV, &K_full[k_stage]);
                k_stage ^= 1;
                if (k_stage == 0) k_empty_phase ^= 1;

                /* Load V[n] */
                mbarrier_wait_parity(&V_empty[v_stage], v_empty_phase);
                mbarrier_arrive_expect_tx(&V_full[v_stage], KV_TILE_BYTES);
                uint32_t vs_lo = (v_stage == 0) ? V0_lo : V1_lo;
                uint32_t vs_hi = (v_stage == 0) ? V0_hi : V1_hi;
                tma_load_2d((void*)vs_lo, tma_V, 0,  n * BLOCK_KV, &V_full[v_stage]);
                tma_load_2d((void*)vs_hi, tma_V, 64, n * BLOCK_KV, &V_full[v_stage]);
                v_stage ^= 1;
                if (v_stage == 0) v_empty_phase ^= 1;
            }
        }
        /* Producer warps 1-3: idle (TMA only needs 1 thread) */
        return;
    }

    /* ================================================================
     *  CONSUMER PATH (WG1 or WG2, warps 4-11)
     * ================================================================ */
    const int cwg = wg_id - 1;       /* 0 = consumer WG1, 1 = consumer WG2 */
    const int cwarp = warp_id - 4;   /* 0-7 within consumer warps */
    const int mywarp = cwarp & 3;    /* 0-3 within this consumer WG */

    /* Scheduler barriers: bar 2 for WG1, bar 3 for WG2 */
    const int my_bar    = 2 + cwg;       /* WG1=2, WG2=3 */
    const int other_bar = 2 + (1 - cwg); /* WG1→3, WG2→2 */

    /* Wait for Q load */
    mbarrier_wait_parity(Q_full_mbar, 0);

    /* Q descriptor for this consumer WG (each WG handles 64 rows of Q).
     * WG1: rows 0-63 (groups 0-7), WG2: rows 64-127 (groups 8-15). */
    const uint32_t my_Q_lo = Q_lo + cwg * 8 * STRIDE;  /* +0 or +8KB */
    const uint32_t my_Q_hi = Q_hi + cwg * 8 * STRIDE;
    const uint64_t desc_q_lo = make_wgmma_desc(my_Q_lo, STRIDE);
    const uint64_t desc_q_hi = make_wgmma_desc(my_Q_hi, STRIDE);

    /* Accumulators */
    float O_acc[64];  /* unified: [0..31]=DIM lo, [32..63]=DIM hi */
    #pragma unroll
    for (int i = 0; i < 64; i++) O_acc[i] = 0.0f;
    float rowmax[2] = {-FLT_MAX, -FLT_MAX};
    float rowsumexp[2] = {0.0f, 0.0f};

    /* Pre-packed P values (bf16x2) for RS WGMMA PV GEMM.
     * Stored separately from S_acc so QK[n+1] can overwrite S_acc
     * while PV[n] reads from P_packed concurrently. */
    uint32_t P_packed[32];

    /* Pipeline state — K and V tracked independently for overlap */
    int k_stage = 0, v_stage = 0;
    int k_full_phase = 0, v_full_phase = 0;

    /* mma_init: WG1 (cwg=0) primes its scheduler barrier so it can proceed first */
    if (cwg == 0) {
        named_barrier_arrive(2, 256);  /* arrive on bar 2 (WG1's barrier) */
    }

    /* ---- Prologue: wait for first K tile ---- */
    mbarrier_wait_parity(&K_full[k_stage], k_full_phase);

    /* ================================================================
     *  PROLOGUE: QK[0] only — no PV to pair with
     *
     *  CuTe DSL first_half_block_overlap: NO scheduler barriers.
     *  Both consumer WGs do QK[0] independently.
     * ================================================================ */
    {
        const uint32_t cur_K_lo = (k_stage == 0) ? K0_lo : K1_lo;
        const uint32_t cur_K_hi = (k_stage == 0) ? K0_hi : K1_hi;

        float S_acc[64];
        #pragma unroll
        for (int i = 0; i < 64; i++) S_acc[i] = 0.0f;

        const uint64_t dk_lo = make_wgmma_desc(cur_K_lo, STRIDE);
        const uint64_t dk_hi = make_wgmma_desc(cur_K_hi, STRIDE);

        wgmma_fence();
        #pragma unroll
        for (int ks = 0; ks < DIM / 16; ks++) {
            uint64_t dq = (ks < 4) ? gmma_desc_advance(desc_q_lo, ks * 2)
                                   : gmma_desc_advance(desc_q_hi, (ks - 4) * 2);
            uint64_t dk = (ks < 4) ? gmma_desc_advance(dk_lo, ks * 2)
                                   : gmma_desc_advance(dk_hi, (ks - 4) * 2);
            wgmma_m64n128k16_f32_bf16(S_acc, dq, dk, (ks == 0) ? 0 : 1);
        }
        wgmma_commit_group();
        wgmma_wait_group<0>();
        wgmma_fence();

        /* Release K[0] */
        if (lane_id == 0 && mywarp == 0) {
            mbarrier_arrive(&K_empty[k_stage]);
        }
        k_stage ^= 1;
        if (k_stage == 0) k_full_phase ^= 1;

        /* Softmax (first block — no prior O to rescale) */
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            float rv[32];
            #pragma unroll
            for (int p4 = 0; p4 < 16; p4++) {
                rv[p4*2+0] = S_acc[(p4<<2)|(half<<1)|0];
                rv[p4*2+1] = S_acc[(p4<<2)|(half<<1)|1];
            }
            /* rv[] is in UNSCALED space — scale is fused into exp2 via FMA.
             * This eliminates 32 FMULs per half per iteration. */

            if (is_causal) {
                const int row = mywarp * 16 + half * 8 + (lane_id / 4);
                const int q_pos = q_block_id * BLOCK_Q + cwg * 64 + row;
                #pragma unroll
                for (int p4 = 0; p4 < 16; p4++) {
                    #pragma unroll
                    for (int s2 = 0; s2 < 2; s2++) {
                        const int col = p4 * 8 + s2 + (lane_id % 4) * 2;
                        const int kv_pos = 0 * BLOCK_KV + col;
                        if (kv_pos > q_pos)
                            rv[p4*2+s2] = -FLT_MAX;
                    }
                }
            }

            /* Tree max reduction (unscaled rv) */
            float tmax[16];
            #pragma unroll
            for (int i = 0; i < 16; i++) tmax[i] = fmaxf(rv[i], rv[i + 16]);
            #pragma unroll
            for (int i = 0; i < 8; i++) tmax[i] = fmaxf(tmax[i], tmax[i + 8]);
            #pragma unroll
            for (int i = 0; i < 4; i++) tmax[i] = fmaxf(tmax[i], tmax[i + 4]);
            tmax[0] = fmaxf(tmax[0], tmax[2]); tmax[1] = fmaxf(tmax[1], tmax[3]);
            float lmax = fmaxf(tmax[0], tmax[1]);
            lmax = fmaxf(lmax, __shfl_xor_sync(0xFFFFFFFF, lmax, 1));
            lmax = fmaxf(lmax, __shfl_xor_sync(0xFFFFFFFF, lmax, 2));

            rowmax[half] = lmax;

            /* Exp2 via FMA: exp2(rv * scale - max * scale) = exp2(fma(rv, scale, -max*scale))
             * Fuses scale multiply into the exp2 argument, saving 32 FMULs. */
            float neg_max_scaled = -lmax * softmax_scale_log2;
            float tsum[16];
            #pragma unroll
            for (int c = 0; c < 32; c++) {
                rv[c] = fast_exp2f(fmaf(rv[c], softmax_scale_log2, neg_max_scaled));
            }
            #pragma unroll
            for (int i = 0; i < 16; i++) tsum[i] = rv[i] + rv[i + 16];
            #pragma unroll
            for (int i = 0; i < 8; i++) tsum[i] += tsum[i + 8];
            #pragma unroll
            for (int i = 0; i < 4; i++) tsum[i] += tsum[i + 4];
            tsum[0] += tsum[2]; tsum[1] += tsum[3];
            float lsum = tsum[0] + tsum[1];
            /* Deferred quad shuffle to finalize (CuTe DSL pattern) */
            rowsumexp[half] = lsum;

            #pragma unroll
            for (int p4 = 0; p4 < 16; p4++) {
                S_acc[(p4<<2)|(half<<1)|0] = rv[p4*2+0];
                S_acc[(p4<<2)|(half<<1)|1] = rv[p4*2+1];
            }
        }

        /* Pre-pack P to bf16x2 for PV RS WGMMA */
        #pragma unroll
        for (int ks = 0; ks < BLOCK_KV / 16; ks++) {
            P_packed[ks*4+0] = pack_bf16(S_acc[ks*8+0], S_acc[ks*8+1]);
            P_packed[ks*4+1] = pack_bf16(S_acc[ks*8+2], S_acc[ks*8+3]);
            P_packed[ks*4+2] = pack_bf16(S_acc[ks*8+4], S_acc[ks*8+5]);
            P_packed[ks*4+3] = pack_bf16(S_acc[ks*8+6], S_acc[ks*8+7]);
        }
    }

    /* ================================================================
     *  MAINLOOP: QK[n] + PV[n-1] overlap (n = 1..max_kv_iter-1)
     *
     *  CuTe DSL mma_one_n_block_intrawg_overlap pattern:
     *  Issue QK[n] (wg_wait=-1) → Issue PV[n-1] (wg_wait=-1)
     *  → wait_group<1> (QK done) → softmax (overlaps PV execution!)
     *  → wait_group<0> (PV done) → rescale O → pack P
     *
     *  Key: softmax scalar work runs while PV tensor cores execute.
     * ================================================================ */
    if (max_kv_iter > 1) {
        /* Prefetch K[1] for first mainloop iteration */
        mbarrier_wait_parity(&K_full[k_stage], k_full_phase);
    }

    for (int kv_id = 1; kv_id < max_kv_iter; kv_id++) {
        /* K[kv_id] already waited (from above or previous iteration) */

        named_barrier_sync(my_bar, 256);

        /* == Issue QK[kv_id] — commit but don't wait == */
        float S_acc[64];
        #pragma unroll
        for (int i = 0; i < 64; i++) S_acc[i] = 0.0f;

        const uint32_t cur_K_lo = (k_stage == 0) ? K0_lo : K1_lo;
        const uint32_t cur_K_hi = (k_stage == 0) ? K0_hi : K1_hi;
        const uint64_t dk_lo = make_wgmma_desc(cur_K_lo, STRIDE);
        const uint64_t dk_hi = make_wgmma_desc(cur_K_hi, STRIDE);

        wgmma_fence();
        #pragma unroll
        for (int ks = 0; ks < DIM / 16; ks++) {
            uint64_t dq = (ks < 4) ? gmma_desc_advance(desc_q_lo, ks * 2)
                                   : gmma_desc_advance(desc_q_hi, (ks - 4) * 2);
            uint64_t dk = (ks < 4) ? gmma_desc_advance(dk_lo, ks * 2)
                                   : gmma_desc_advance(dk_hi, (ks - 4) * 2);
            wgmma_m64n128k16_f32_bf16(S_acc, dq, dk, (ks == 0) ? 0 : 1);
        }
        wgmma_commit_group();
        /* QK is now in flight — don't wait yet */

        /* == Issue PV[kv_id-1] — overlaps with QK on tensor cores == */
        /* Wait for V[kv_id-1] while QK tensor cores are executing */
        mbarrier_wait_parity(&V_full[v_stage], v_full_phase);

        /* V descriptor: LBO spans V_lo→V_hi (16KB gap) for m64n128k16 */
        const uint32_t cur_V_lo = (v_stage == 0) ? V0_lo : V1_lo;
        const uint64_t dv = make_wgmma_desc_lbo(cur_V_lo, HALF_KV, STRIDE);

        wgmma_fence();
        #pragma unroll
        for (int ks = 0; ks < BLOCK_KV / 16; ks++) {
            uint64_t dv_k = gmma_desc_advance(dv, ks * V_KS_ADVANCE);
            wgmma_m64n128k16_f32_bf16_RS(O_acc,
                P_packed[ks*4+0], P_packed[ks*4+1],
                P_packed[ks*4+2], P_packed[ks*4+3], dv_k, 1);
        }
        wgmma_commit_group();
        /* PV is now in flight — 2 WGMMA groups pending (QK + PV) */

        /* Signal other WG AFTER PV issue (CuTe DSL pattern) */
        named_barrier_arrive(other_bar, 256);

        /* == Wait for QK to complete (PV still running) == */
        wgmma_wait_group<1>();
        wgmma_fence();

        /* Release K[kv_id] */
        if (lane_id == 0 && mywarp == 0) {
            mbarrier_arrive(&K_empty[k_stage]);
        }
        k_stage ^= 1;
        if (k_stage == 0) k_full_phase ^= 1;

        /* == Softmax — runs while PV tensor cores are still executing! ==
         * O rescaling is deferred until after PV completes (CuTe DSL pattern
         * for hdim≤128: rescale_O_before_gemm=False). */
        float o_rescale[2];
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            float rv[32];
            #pragma unroll
            for (int p4 = 0; p4 < 16; p4++) {
                rv[p4*2+0] = S_acc[(p4<<2)|(half<<1)|0];
                rv[p4*2+1] = S_acc[(p4<<2)|(half<<1)|1];
            }
            /* rv[] is UNSCALED — scale fused into exp2 via FMA */

            if (is_causal) {
                const int row = mywarp * 16 + half * 8 + (lane_id / 4);
                const int q_pos = q_block_id * BLOCK_Q + cwg * 64 + row;
                #pragma unroll
                for (int p4 = 0; p4 < 16; p4++) {
                    #pragma unroll
                    for (int s2 = 0; s2 < 2; s2++) {
                        const int col = p4 * 8 + s2 + (lane_id % 4) * 2;
                        const int kv_pos = kv_id * BLOCK_KV + col;
                        if (kv_pos > q_pos)
                            rv[p4*2+s2] = -FLT_MAX;
                    }
                }
            }

            /* Tree max reduction (unscaled rv) */
            float tmax[16];
            #pragma unroll
            for (int i = 0; i < 16; i++) tmax[i] = fmaxf(rv[i], rv[i + 16]);
            #pragma unroll
            for (int i = 0; i < 8; i++) tmax[i] = fmaxf(tmax[i], tmax[i + 8]);
            #pragma unroll
            for (int i = 0; i < 4; i++) tmax[i] = fmaxf(tmax[i], tmax[i + 4]);
            tmax[0] = fmaxf(tmax[0], tmax[2]); tmax[1] = fmaxf(tmax[1], tmax[3]);
            float lmax = fmaxf(tmax[0], tmax[1]);
            lmax = fmaxf(lmax, __shfl_xor_sync(0xFFFFFFFF, lmax, 1));
            lmax = fmaxf(lmax, __shfl_xor_sync(0xFFFFFFFF, lmax, 2));

            float new_max = fmaxf(lmax, rowmax[half]);
            /* Rescale uses unscaled max difference, scaled via FMA */
            float rescale = fast_exp2f((rowmax[half] - new_max) * softmax_scale_log2);
            rowmax[half] = new_max;
            o_rescale[half] = rescale;

            /* Exp2 via FMA: exp2(fma(rv, scale, -max*scale)) */
            float neg_max_scaled = -new_max * softmax_scale_log2;
            float tsum[16];
            #pragma unroll
            for (int c = 0; c < 32; c++) {
                rv[c] = fast_exp2f(fmaf(rv[c], softmax_scale_log2, neg_max_scaled));
            }
            #pragma unroll
            for (int i = 0; i < 16; i++) tsum[i] = rv[i] + rv[i + 16];
            #pragma unroll
            for (int i = 0; i < 8; i++) tsum[i] += tsum[i + 8];
            #pragma unroll
            for (int i = 0; i < 4; i++) tsum[i] += tsum[i + 4];
            tsum[0] += tsum[2]; tsum[1] += tsum[3];
            float lsum = tsum[0] + tsum[1];
            /* Deferred: no quad sum shuffle here */
            rowsumexp[half] = rowsumexp[half] * rescale + lsum;

            #pragma unroll
            for (int p4 = 0; p4 < 16; p4++) {
                S_acc[(p4<<2)|(half<<1)|0] = rv[p4*2+0];
                S_acc[(p4<<2)|(half<<1)|1] = rv[p4*2+1];
            }
        }

        /* == Wait for PV[n-1] to complete == */
        wgmma_wait_group<0>();

        /* Release V[kv_id-1] */
        if (lane_id == 0 && mywarp == 0) {
            mbarrier_arrive(&V_empty[v_stage]);
        }
        v_stage ^= 1;
        if (v_stage == 0) v_full_phase ^= 1;

        /* Deferred O rescaling (CuTe DSL: rescale AFTER PV for hdim≤128).
         * Mathematically correct: PV[n-1] added P[n-1]@V[n-1] using P values
         * computed with the same running max. Rescaling the entire O (including
         * the just-added PV contribution) adjusts all terms to the new max. */
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            #pragma unroll
            for (int p4 = 0; p4 < 8; p4++) {
                O_acc[(p4<<2)|(half<<1)|0] *= o_rescale[half];
                O_acc[(p4<<2)|(half<<1)|1] *= o_rescale[half];
                O_acc[32+(p4<<2)|(half<<1)|0] *= o_rescale[half];
                O_acc[32+(p4<<2)|(half<<1)|1] *= o_rescale[half];
            }
        }

        /* Pack P for next PV iteration */
        #pragma unroll
        for (int ks = 0; ks < BLOCK_KV / 16; ks++) {
            P_packed[ks*4+0] = pack_bf16(S_acc[ks*8+0], S_acc[ks*8+1]);
            P_packed[ks*4+1] = pack_bf16(S_acc[ks*8+2], S_acc[ks*8+3]);
            P_packed[ks*4+2] = pack_bf16(S_acc[ks*8+4], S_acc[ks*8+5]);
            P_packed[ks*4+3] = pack_bf16(S_acc[ks*8+6], S_acc[ks*8+7]);
        }

        /* Prefetch next K */
        if (kv_id + 1 < max_kv_iter) {
            mbarrier_wait_parity(&K_full[k_stage], k_full_phase);
        }
    } /* end mainloop */

    /* ================================================================
     *  EPILOGUE: PV[last] — no QK to pair with
     *
     *  CuTe DSL last_half_block_overlap pattern.
     * ================================================================ */
    {
        /* Wait for V[max_kv_iter-1] */
        mbarrier_wait_parity(&V_full[v_stage], v_full_phase);

        const uint32_t cur_V_lo = (v_stage == 0) ? V0_lo : V1_lo;
        const uint64_t dv = make_wgmma_desc_lbo(cur_V_lo, HALF_KV, STRIDE);

        wgmma_fence();
        #pragma unroll
        for (int ks = 0; ks < BLOCK_KV / 16; ks++) {
            uint64_t dv_k = gmma_desc_advance(dv, ks * V_KS_ADVANCE);
            wgmma_m64n128k16_f32_bf16_RS(O_acc,
                P_packed[ks*4+0], P_packed[ks*4+1],
                P_packed[ks*4+2], P_packed[ks*4+3], dv_k, 1);
        }
        wgmma_commit_group();
        wgmma_wait_group<0>();

        /* Release V[last] */
        if (lane_id == 0 && mywarp == 0) {
            mbarrier_arrive(&V_empty[v_stage]);
        }
        v_stage ^= 1;
        if (v_stage == 0) v_full_phase ^= 1;
    }

    /* ---- Epilogue: finalize and store O via TMA S2G ---- */
    wgmma_fence();

    /* Deferred quad sum reduction (CuTe DSL finalize pattern).
     * rowsumexp was accumulated per-thread without warp shuffles in the mainloop.
     * Now do the quad shuffle once, saving 4 shuffles × max_kv_iter iterations. */
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        rowsumexp[half] += __shfl_xor_sync(0xFFFFFFFF, rowsumexp[half], 1);
        rowsumexp[half] += __shfl_xor_sync(0xFFFFFFFF, rowsumexp[half], 2);
    }

    /* Normalize by softmax sum */
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        float inv = fast_rcp(rowsumexp[half]);
        #pragma unroll
        for (int p4 = 0; p4 < 8; p4++) {
            O_acc[(p4<<2)|(half<<1)|0] *= inv;
            O_acc[(p4<<2)|(half<<1)|1] *= inv;
            O_acc[32+(p4<<2)|(half<<1)|0] *= inv;
            O_acc[32+(p4<<2)|(half<<1)|1] *= inv;
        }
    }

    /* Step 1: Write O from registers → Q SMEM (reuse Q region, swizzled layout).
     * Both consumer WGs write their 64 rows. WG1 → rows 0-63, WG2 → rows 64-127.
     *
     * SMEM address for each (row, col) with SWIZZLE_128B:
     *   byte_off = full_row * 128 + col_base * 2
     *   swizzled = byte_off ^ (((byte_off >> 7) & 7) << 4)
     */
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        const int local_row = mywarp * 16 + half * 8 + (lane_id / 4);
        const int full_row = cwg * 64 + local_row;
        #pragma unroll
        for (int p4 = 0; p4 < 8; p4++) {
            const int col_base = p4 * 8 + (lane_id % 4) * 2;
            const int idx0 = (p4 << 2) | (half << 1) | 0;
            const int idx1 = (p4 << 2) | (half << 1) | 1;

            uint32_t lo_packed = pack_bf16(O_acc[idx0], O_acc[idx1]);
            uint32_t hi_packed = pack_bf16(O_acc[32+idx0], O_acc[32+idx1]);

            /* Compute swizzled SMEM offset (SWIZZLE_128B: XOR bits[6:4] with bits[9:7]) */
            uint32_t byte_off = full_row * 128 + col_base * 2;
            uint32_t swz = byte_off ^ (((byte_off >> 7) & 7) << 4);
            asm volatile("st.shared.u32 [%0], %1;" :: "r"(Q_lo + swz), "r"(lo_packed));
            asm volatile("st.shared.u32 [%0], %1;" :: "r"(Q_hi + swz), "r"(hi_packed));
        }
    }

    /* Step 2: Fence + 2-phase barrier (CuTe DSL epilogue pattern).
     * Phase 1: All 256 consumer threads ARRIVE (mark SMEM writes done).
     * Phase 2: Warp 4 (32 threads) SYNC (waits for all 288 arrivals, then does TMA). */
    constexpr int EPILOGUE_BAR = 1;
    constexpr int EPILOGUE_THREADS = 256 + 32;  /* 256 consumers + 32 TMA warp */

    fence_view_async_shared();
    named_barrier_arrive(EPILOGUE_BAR, EPILOGUE_THREADS);

    /* Step 3: Warp 4 does TMA S2G (one warp copies entire O tile from SMEM → GMEM). */
    if (warp_id == 4) {
        named_barrier_sync(EPILOGUE_BAR, EPILOGUE_THREADS);

        /* TMA S2G: store Q_lo → O[0:63], Q_hi → O[64:127] */
        tma_store_2d(tma_O, Q_lo, 0,  q_block_id * BLOCK_Q);
        tma_store_2d(tma_O, Q_hi, 64, q_block_id * BLOCK_Q);
        cp_async_bulk_commit_group();
        cp_async_bulk_wait_group();
    }
}

/* ======================================================================
 *  Helpers + kernel_run
 * ====================================================================== */

static int parse_int_env(const char *name, int fallback) {
    const char *v = getenv(name);
    if (v && *v) return atoi(v);
    return fallback;
}

static bool parse_bool_env(const char *name, bool fallback) {
    const char *v = getenv(name);
    if (!v) return fallback;
    return (strcmp(v, "true") == 0 || strcmp(v, "1") == 0 ||
            strcmp(v, "True") == 0 || strcmp(v, "TRUE") == 0);
}

static int json_int(const char *json, const char *key, int fallback) {
    if (!json) return fallback;
    const char *p = strstr(json, key);
    if (!p) return fallback;
    p += strlen(key);
    while (*p && (*p == '"' || *p == ':' || *p == ' ' || *p == '\t')) ++p;
    if (*p < '0' || *p > '9') return fallback;
    int val = 0;
    while (*p >= '0' && *p <= '9') { val = val * 10 + (*p - '0'); ++p; }
    return val;
}

static bool json_bool(const char *json, const char *key, bool fallback) {
    if (!json) return fallback;
    const char *p = strstr(json, key);
    if (!p) return fallback;
    p += strlen(key);
    while (*p && (*p == '"' || *p == ':' || *p == ' ' || *p == '\t')) ++p;
    if (strncmp(p, "true", 4) == 0) return true;
    if (strncmp(p, "false", 5) == 0) return false;
    return fallback;
}

/* --- TMA descriptor encoder -------------------------------------------- */

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

static CUtensorMap* d_tma_Q = nullptr;
static CUtensorMap* d_tma_K = nullptr;
static CUtensorMap* d_tma_V = nullptr;
static CUtensorMap* d_tma_O = nullptr;
static int d_tma_bh_count = 0;

extern "C" int kernel_run(
    __nv_bfloat16 **inputs,  int num_inputs,
    __nv_bfloat16 **outputs, int num_outputs,
    int n, cudaStream_t stream)
{
    const char *config_json = getenv("CUDA_EXEC_CONFIG_JSON");

    int B = parse_int_env("CUDA_EXEC_PARAM_BATCH_SIZE", 0);
    int S = parse_int_env("CUDA_EXEC_PARAM_SEQ_LEN",    0);
    int H = parse_int_env("CUDA_EXEC_PARAM_NUM_HEADS",  0);
    int D = parse_int_env("CUDA_EXEC_PARAM_HEAD_DIM",   0);

    if (B == 0) B = json_int(config_json, "batch_size", 0);
    if (S == 0) S = json_int(config_json, "seq_len",    0);
    if (H == 0) H = json_int(config_json, "num_heads",  0);
    if (D == 0) D = json_int(config_json, "head_dim",   0);

    if (D == 0) D = 128;
    if (H == 0) H = 16;
    if (S == 0 && B == 0 && n > 0) {
        int total_tokens = n / (H * D);
        B = 1;
        S = total_tokens / B;
    }
    if (B == 0 || S == 0) return -1;

    bool causal = parse_bool_env("CUDA_EXEC_PARAM_CAUSAL", false);
    if (!causal && config_json)
        causal = json_bool(config_json, "causal", false);

    if (D != 128) return -2;
    if (!init_tma_encoder()) return -3;

    {
        const int BLOCK_Q   = 128;   /* doubled: 2 consumer WGs × m64 */
        const int BLOCK_KV  = 128;
        const int DIM_CONST = 128;
        const int TB_SIZE   = 384;   /* 3 warp groups × 128 */

        int num_blocks = B * H * cdiv(S, BLOCK_Q);

        /* SMEM: Q(32KB) + K×2(64KB) + V×2(64KB) + mbarriers = 160KB + 128B */
        int smem_size = 2 * (BLOCK_Q * (DIM_CONST / 2) * (int)sizeof(nv_bfloat16))
                      + 8 * (BLOCK_KV * (DIM_CONST / 2) * (int)sizeof(nv_bfloat16))
                      + 256;  /* mbarrier + alignment */

        int bh = B * H;
        if (bh > d_tma_bh_count) {
            if (d_tma_Q) cudaFree(d_tma_Q);
            if (d_tma_K) cudaFree(d_tma_K);
            if (d_tma_V) cudaFree(d_tma_V);
            if (d_tma_O) cudaFree(d_tma_O);
            cudaMalloc(&d_tma_Q, bh * sizeof(CUtensorMap));
            cudaMalloc(&d_tma_K, bh * sizeof(CUtensorMap));
            cudaMalloc(&d_tma_V, bh * sizeof(CUtensorMap));
            cudaMalloc(&d_tma_O, bh * sizeof(CUtensorMap));
            d_tma_bh_count = bh;
        }

        CUtensorMap* h_tma_Q = (CUtensorMap*)malloc(bh * sizeof(CUtensorMap));
        CUtensorMap* h_tma_K = (CUtensorMap*)malloc(bh * sizeof(CUtensorMap));
        CUtensorMap* h_tma_V = (CUtensorMap*)malloc(bh * sizeof(CUtensorMap));
        CUtensorMap* h_tma_O = (CUtensorMap*)malloc(bh * sizeof(CUtensorMap));

        const __nv_bfloat16 *Q_ptr = inputs[0];
        const __nv_bfloat16 *K_ptr = inputs[1];
        const __nv_bfloat16 *V_ptr = inputs[2];
        __nv_bfloat16 *O_ptr = outputs[0];
        int seq_stride = H * D;

        for (int b = 0; b < B; b++) {
            for (int h = 0; h < H; h++) {
                int idx = b * H + h;
                void* q_base = (void*)(Q_ptr + b * S * seq_stride + h * D);
                void* k_base = (void*)(K_ptr + b * S * seq_stride + h * D);
                void* v_base = (void*)(V_ptr + b * S * seq_stride + h * D);
                void* o_base = (void*)(O_ptr + b * S * seq_stride + h * D);

                cuuint64_t dims[2] = {(cuuint64_t)D, (cuuint64_t)S};
                cuuint64_t strides[1] = {(cuuint64_t)(seq_stride * sizeof(nv_bfloat16))};
                cuuint32_t box_q[2]  = {64, (cuuint32_t)BLOCK_Q};
                cuuint32_t box_kv[2] = {64, (cuuint32_t)BLOCK_KV};
                cuuint32_t elem[2] = {1, 1};

                s_encodeTiled(&h_tma_Q[idx], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                    2, q_base, dims, strides, box_q, elem,
                    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                    CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
                s_encodeTiled(&h_tma_K[idx], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                    2, k_base, dims, strides, box_kv, elem,
                    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                    CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
                s_encodeTiled(&h_tma_V[idx], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                    2, v_base, dims, strides, box_kv, elem,
                    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                    CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
                /* O uses same layout as Q (same dims, strides, box, swizzle) */
                s_encodeTiled(&h_tma_O[idx], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                    2, o_base, dims, strides, box_q, elem,
                    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                    CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            }
        }

        cudaMemcpyAsync(d_tma_Q, h_tma_Q, bh * sizeof(CUtensorMap),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_tma_K, h_tma_K, bh * sizeof(CUtensorMap),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_tma_V, h_tma_V, bh * sizeof(CUtensorMap),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_tma_O, h_tma_O, bh * sizeof(CUtensorMap),
                        cudaMemcpyHostToDevice, stream);

        auto kernel = flash_attention_2wg<BLOCK_Q, BLOCK_KV, DIM_CONST>;
        cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        kernel<<<num_blocks, TB_SIZE, smem_size, stream>>>(
            nullptr, nullptr, nullptr, outputs[0],
            B, S, H, S, S, causal ? 1 : 0,
            d_tma_Q, d_tma_K, d_tma_V, d_tma_O);

        free(h_tma_Q);
        free(h_tma_K);
        free(h_tma_V);
        free(h_tma_O);
    }

    return 0;
}
