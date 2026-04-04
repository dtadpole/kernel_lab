/*
 * Flash Attention forward pass — BF16, SM90 TMA + WGMMA kernel.
 * Unified 128 threads (1 warp group) + TMA loads + double-buffered K.
 *
 * Architecture: 128 threads, thread 0 issues TMA loads, all threads
 * participate in WGMMA compute. TMA produces SWIZZLE_128B layout
 * natively — no software address swizzling needed.
 *
 * Pipeline: V[i] and K[i+1] TMA loads overlap with QK[i] + softmax.
 * PV GEMM uses WGMMA RS mode: P stays in registers after softmax.
 *
 * DIM=128 split into 2×64 for SWIZZLE_128B (boxDim[0]*2B = 128B).
 *
 * Constants: BLOCK_Q=64, BLOCK_KV=64, DIM=128, 128 threads.
 * SMEM: Q_lo(8KB) + Q_hi(8KB) + K0_lo(8KB) + K0_hi(8KB)
 *     + K1_lo(8KB) + K1_hi(8KB) + V_lo(8KB)  + V_hi(8KB)
 *     + mbarriers(32B) = 64KB + 32B
 *
 * Target: NVIDIA H100 (SM90a, GH100). Compile with -arch=sm_90a.
 *
 * kernel_run contract:
 *   extern "C" int kernel_run(__nv_bfloat16** inputs, int num_inputs,
 *                             __nv_bfloat16** outputs, int num_outputs,
 *                             int n, cudaStream_t stream);
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <dlfcn.h>
#include <cuda.h>

__device__ __host__ constexpr
int cdiv(int a, int b) { return (a + b - 1) / b; }

/* --- Fast math: inline PTX to avoid FCHK + slowpath overhead --------- */

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

/* --- mbarrier helpers ------------------------------------------------ */

__device__ __forceinline__
void mbarrier_init(uint64_t* mbar, unsigned count) {
    uint32_t addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                 :: "r"(addr), "r"(count));
}

__device__ __forceinline__
void mbarrier_arrive_expect_tx(uint64_t* mbar, unsigned tx_bytes) {
    uint32_t addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                 :: "r"(addr), "r"(tx_bytes));
}

__device__ __forceinline__
void mbarrier_wait_parity(uint64_t* mbar, unsigned phase) {
    uint32_t addr = __cvta_generic_to_shared(mbar);
    uint32_t result;
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n"
        "  selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(result) : "r"(addr), "r"(phase));
    while (result == 0) {
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n"
            "  selp.u32 %0, 1, 0, p;\n"
            "}\n"
            : "=r"(result) : "r"(addr), "r"(phase));
    }
}

__device__ __forceinline__
void mbarrier_inval(uint64_t* mbar) {
    uint32_t addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(addr));
}

/* --- TMA load: cp.async.bulk.tensor.2d ------------------------------- */

__device__ __forceinline__
void tma_load_2d(void* smem_dst, const void* tma_desc,
                 int coord0, int coord1, uint64_t* mbar) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_dst);
    uint32_t mbar_addr = __cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(smem_addr), "l"(tma_desc),
           "r"(coord0), "r"(coord1), "r"(mbar_addr)
        : "memory");
}

/* --- WGMMA synchronization ------------------------------------------ */

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

/* --- WGMMA descriptor construction ---------------------------------- */

__device__ __forceinline__
uint64_t make_wgmma_desc(uint32_t smem_addr, int stride_bytes) {
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_addr >> 4) & 0x3FFF);            // [0:14)  start_address
    desc |= (uint64_t)(1) << 16;                               // [16:30) leading_byte_off = 1
    desc |= (uint64_t)(((stride_bytes >> 4) & 0x3FFF)) << 32;  // [32:46) stride_byte_off
    desc |= (uint64_t)(1) << 62;                                // [62:64) SWIZZLE_128B
    return desc;
}

__device__ __forceinline__
uint64_t gmma_desc_advance(uint64_t desc, int offset_16B) {
    uint32_t lo = (uint32_t)desc + (uint32_t)offset_16B;
    uint32_t hi = (uint32_t)(desc >> 32);
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

/* --- WGMMA m64n64k16 (QK GEMM) -------------------------------------- */

__device__ __forceinline__
void wgmma_m64n64k16_f32_bf16(float acc[32], uint64_t desc_a, uint64_t desc_b,
                               int scale_D) {
    asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, %34, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
    "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
    " %8,  %9,  %10, %11, %12, %13, %14, %15, "
    " %16, %17, %18, %19, %20, %21, %22, %23, "
    " %24, %25, %26, %27, %28, %29, %30, %31},"
    " %32, %33, p, 1, 1, 0, 0;\n"
    "}\n"
    : "+f"(acc[0]),  "+f"(acc[1]),  "+f"(acc[2]),  "+f"(acc[3]),
      "+f"(acc[4]),  "+f"(acc[5]),  "+f"(acc[6]),  "+f"(acc[7]),
      "+f"(acc[8]),  "+f"(acc[9]),  "+f"(acc[10]), "+f"(acc[11]),
      "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]),
      "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]),
      "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]),
      "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]),
      "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31])
    : "l"(desc_a), "l"(desc_b), "r"(scale_D));
}

/* --- WGMMA m64n64k16 RS mode (A from registers, B from SMEM) -------- */

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

/* --- WGMMA m64n128k16 (PV GEMM) ------------------------------------- */

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
    " %64, %65, p, 1, 1, 0, 1;\n"
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

/* --- Pack two F32 values into BF16x2 -------------------------------- */

__device__ __forceinline__
uint32_t pack_bf16(float a, float b) {
    uint32_t result;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(result) : "f"(a), "f"(b));
    return result;
}

/* --- WGMMA m64n128k16 RS mode (A from registers, B from SMEM) ------- */

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

/* --- fence_view_async_shared ----------------------------------------- */

__device__ __forceinline__
void fence_view_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

/* ======================================================================
 *  TMA + WGMMA Flash Attention kernel
 *
 *  128 threads = 1 warp group (warps 0-3).
 *  Thread 0 issues TMA loads; all threads do WGMMA compute.
 *  Double-buffered K: TMA loads K[i+1] during QK[i] + softmax.
 *  Dedicated V buffer: V[i] TMA loads overlap with QK[i] + softmax.
 *  PV GEMM: RS mode — P stays in registers, no SMEM roundtrip.
 *
 *  SMEM layout (split-DIM: each tile split into lo[0:63] + hi[64:127]):
 *    Q_lo(8KB) + Q_hi(8KB) = 16KB  (persistent)
 *    K0_lo(8KB) + K0_hi(8KB) = 16KB (stage 0)
 *    K1_lo(8KB) + K1_hi(8KB) = 16KB (stage 1)
 *    V_lo(8KB) + V_hi(8KB) = 16KB   (dedicated)
 *    mbarriers: 4 × 8B = 32B
 *    Total: 64KB + 32B
 * ====================================================================== */

template<int BLOCK_Q, int BLOCK_KV, int DIM>
__launch_bounds__(128, 1)
__global__
void flash_attention_tma_wgmma(
    const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    nv_bfloat16 *O,
    int B, int S, int H,
    int len_q, int len_kv,
    int is_causal,
    const CUtensorMap *tma_Q,
    const CUtensorMap *tma_K,
    const CUtensorMap *tma_V)
{
    constexpr int HALF_DIM = DIM / 2;                        /* 64 */
    constexpr int HALF_TILE_BYTES = BLOCK_Q * HALF_DIM * sizeof(nv_bfloat16);  /* 8KB */
    constexpr int KV_HALF_BYTES = BLOCK_KV * HALF_DIM * sizeof(nv_bfloat16);   /* 8KB */
    constexpr int FULL_Q_BYTES = 2 * HALF_TILE_BYTES;        /* 16KB */
    constexpr int FULL_KV_BYTES = 2 * KV_HALF_BYTES;         /* 16KB */

    /* SMEM layout: split-DIM with 8KB sections */
    extern __shared__ char smem_raw[];
    nv_bfloat16 *smem = reinterpret_cast<nv_bfloat16*>(smem_raw);
    const uint32_t smem_base = __cvta_generic_to_shared(smem);

    const uint32_t Q_lo_smem  = smem_base;                              /* 0KB */
    const uint32_t Q_hi_smem  = smem_base + HALF_TILE_BYTES;            /* 8KB */
    const uint32_t K0_lo_smem = smem_base + 2 * HALF_TILE_BYTES;        /* 16KB */
    const uint32_t K0_hi_smem = smem_base + 3 * HALF_TILE_BYTES;        /* 24KB */
    const uint32_t K1_lo_smem = smem_base + 4 * HALF_TILE_BYTES;        /* 32KB */
    const uint32_t K1_hi_smem = smem_base + 5 * HALF_TILE_BYTES;        /* 40KB */
    const uint32_t V_lo_smem  = smem_base + 6 * HALF_TILE_BYTES;        /* 48KB */
    const uint32_t V_hi_smem  = smem_base + 7 * HALF_TILE_BYTES;        /* 56KB */

    /* mbarriers at the end of SMEM */
    uint64_t *mbar_Q  = reinterpret_cast<uint64_t*>(smem_raw + 8 * HALF_TILE_BYTES);
    uint64_t *mbar_K0 = mbar_Q + 1;
    uint64_t *mbar_K1 = mbar_Q + 2;
    uint64_t *mbar_V  = mbar_Q + 3;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    /* Block -> (batch, head, q_block) mapping */
    const int bid = blockIdx.x;
    const int num_q_blocks = cdiv(len_q, BLOCK_Q);
    const int bs_id       = bid / num_q_blocks;
    const int q_block_id  = bid % num_q_blocks;
    const int batch_id    = bs_id / H;
    const int head_id     = bs_id % H;
    const int bh_id       = batch_id * H + head_id;
    const int seq_stride  = H * DIM;

    /* Select per-batch/head TMA descriptors */
    const CUtensorMap *tma_Q_bh = &tma_Q[bh_id];
    const CUtensorMap *tma_K_bh = &tma_K[bh_id];
    const CUtensorMap *tma_V_bh = &tma_V[bh_id];

    /* O output base (still uses strided global stores) */
    nv_bfloat16 *O_base = const_cast<nv_bfloat16*>(Q) - Q + O
                          + batch_id * S * seq_stride + head_id * DIM
                          + q_block_id * BLOCK_Q * seq_stride;
    /* Fix: compute O_base directly */
    O_base = O + batch_id * S * seq_stride + head_id * DIM
               + q_block_id * BLOCK_Q * seq_stride;

    /* KV iteration bounds */
    const int num_kv_iter = cdiv(len_kv, BLOCK_KV);
    const int max_kv_iter = is_causal
        ? min(num_kv_iter, cdiv((q_block_id + 1) * BLOCK_Q, BLOCK_KV))
        : num_kv_iter;

    /* ---- Initialize mbarriers (thread 0 only) ---- */
    if (tid == 0) {
        mbarrier_init(mbar_Q,  1);
        mbarrier_init(mbar_K0, 1);
        mbarrier_init(mbar_K1, 1);
        mbarrier_init(mbar_V,  1);
    }
    __syncthreads();

    /* ---- Load Q via TMA (thread 0 only) ---- */
    if (tid == 0) {
        mbarrier_arrive_expect_tx(mbar_Q, FULL_Q_BYTES);
        tma_load_2d(reinterpret_cast<void*>(Q_lo_smem), tma_Q_bh,
                    0,  q_block_id * BLOCK_Q, mbar_Q);
        tma_load_2d(reinterpret_cast<void*>(Q_hi_smem), tma_Q_bh,
                    64, q_block_id * BLOCK_Q, mbar_Q);
    }
    mbarrier_wait_parity(mbar_Q, 0);

    /* WGMMA descriptor stride: for split-DIM with SWIZZLE_128B,
     * each 8-row group = 8 × 64 × 2B = 1024B.
     * The stride for wgmma descriptor is this group stride. */
    constexpr int SPLIT_STRIDE_BYTES = 8 * HALF_DIM * sizeof(nv_bfloat16);  /* 1024 */

    const float softmax_scale_log2 =
        rsqrtf(static_cast<float>(DIM)) * 1.4426950408889634f;

    /* Initialize accumulators */
    float O_acc[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) O_acc[i] = 0.0f;

    float rowmax[2] = {-FLT_MAX, -FLT_MAX};
    float rowsumexp[2] = {0.0f, 0.0f};

    /* Precompute Q descriptors (split: lo at Q_lo_smem, hi at Q_hi_smem) */
    const uint64_t desc_q_lo = make_wgmma_desc(Q_lo_smem, SPLIT_STRIDE_BYTES);
    const uint64_t desc_q_hi = make_wgmma_desc(Q_hi_smem, SPLIT_STRIDE_BYTES);

    /* ---- Prelude: TMA load K[0] ---- */
    if (max_kv_iter > 0 && tid == 0) {
        mbarrier_arrive_expect_tx(mbar_K0, FULL_KV_BYTES);
        tma_load_2d(reinterpret_cast<void*>(K0_lo_smem), tma_K_bh,
                    0,  0, mbar_K0);
        tma_load_2d(reinterpret_cast<void*>(K0_hi_smem), tma_K_bh,
                    64, 0, mbar_K0);
    }
    if (max_kv_iter > 0) {
        mbarrier_wait_parity(mbar_K0, 0);
    }

    /* V descriptor base (V SMEM location is fixed) */
    const uint64_t desc_v_lo = make_wgmma_desc(V_lo_smem, SPLIT_STRIDE_BYTES);
    const uint64_t desc_v_hi = make_wgmma_desc(V_hi_smem, SPLIT_STRIDE_BYTES);

    /* Phase tracking for double-buffered K mbarriers */
    unsigned k0_phase = 1;  /* K[0] loaded at phase 0, next use is phase 1 */
    unsigned k1_phase = 0;  /* K[1] not yet loaded */
    unsigned v_phase = 0;   /* V first loads at phase 0 */

    /* ---- Main KV loop ---- */
    for (int kv_id = 0; kv_id < max_kv_iter; kv_id++) {
        const int cur_stage  = kv_id & 1;
        const uint32_t cur_K_lo = cur_stage ? K1_lo_smem : K0_lo_smem;
        const uint32_t cur_K_hi = cur_stage ? K1_hi_smem : K0_hi_smem;

        /* == Step 1a: TMA load V[kv_id] (thread 0 only, non-blocking) == */
        if (tid == 0) {
            mbarrier_arrive_expect_tx(mbar_V, FULL_KV_BYTES);
            tma_load_2d(reinterpret_cast<void*>(V_lo_smem), tma_V_bh,
                        0,  kv_id * BLOCK_KV, mbar_V);
            tma_load_2d(reinterpret_cast<void*>(V_hi_smem), tma_V_bh,
                        64, kv_id * BLOCK_KV, mbar_V);
        }

        /* == Step 1b: TMA load K[kv_id+1] (thread 0 only, non-blocking) == */
        const bool has_next_k = (kv_id + 1 < max_kv_iter);
        if (has_next_k && tid == 0) {
            const uint32_t next_K_lo = cur_stage ? K0_lo_smem : K1_lo_smem;
            const uint32_t next_K_hi = cur_stage ? K0_hi_smem : K1_hi_smem;
            uint64_t *mbar_k_next = cur_stage ? mbar_K0 : mbar_K1;
            mbarrier_arrive_expect_tx(mbar_k_next, FULL_KV_BYTES);
            tma_load_2d(reinterpret_cast<void*>(next_K_lo), tma_K_bh,
                        0,  (kv_id + 1) * BLOCK_KV, mbar_k_next);
            tma_load_2d(reinterpret_cast<void*>(next_K_hi), tma_K_bh,
                        64, (kv_id + 1) * BLOCK_KV, mbar_k_next);
        }

        /* == Step 2: QK GEMM using current K from SMEM ==
         * K was loaded by TMA into split-DIM layout (lo + hi).
         * Q is also in split-DIM layout.
         * For m64n64k16: 8 k-steps total.
         *   k-steps 0-3 use lo halves (cols 0-63)
         *   k-steps 4-7 use hi halves (cols 64-127)
         */
        float S_acc[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) S_acc[i] = 0.0f;

        const uint64_t desc_k_lo = make_wgmma_desc(cur_K_lo, SPLIT_STRIDE_BYTES);
        const uint64_t desc_k_hi = make_wgmma_desc(cur_K_hi, SPLIT_STRIDE_BYTES);

        wgmma_fence();

        /* K-stepping: within each half, 4 k-steps of 16 columns.
         * Each k-step advances 2 units of 16B = 32B = 16 bf16 elements.
         * In split-DIM with SWIZZLE_128B:
         *   - Within a half-tile (64 cols), 8-row groups are 1024B apart
         *   - Each k-step is 2 × 16B = 32B within the group */
        #pragma unroll
        for (int ks = 0; ks < DIM / 16; ks++) {
            uint64_t desc_q, desc_k;
            if (ks < 4) {
                /* lo half: cols 0-63 */
                desc_q = gmma_desc_advance(desc_q_lo, ks * 2);
                desc_k = gmma_desc_advance(desc_k_lo, ks * 2);
            } else {
                /* hi half: cols 64-127 */
                desc_q = gmma_desc_advance(desc_q_hi, (ks - 4) * 2);
                desc_k = gmma_desc_advance(desc_k_hi, (ks - 4) * 2);
            }
            wgmma_m64n64k16_f32_bf16(S_acc, desc_q, desc_k,
                                      (ks == 0) ? 0 : 1);
        }

        wgmma_commit_group();
        wgmma_wait_group<0>();
        wgmma_fence();

        /* == Step 3: Softmax on S_acc (in-place, P stays in registers) == */
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            float row_vals[16];
            #pragma unroll
            for (int p4 = 0; p4 < 8; p4++) {
                row_vals[p4 * 2 + 0] = S_acc[(p4 << 2) | (half << 1) | 0];
                row_vals[p4 * 2 + 1] = S_acc[(p4 << 2) | (half << 1) | 1];
            }

            /* Scale */
            #pragma unroll
            for (int c = 0; c < 16; c++)
                row_vals[c] *= softmax_scale_log2;

            /* Causal mask */
            if (is_causal) {
                const int row = warp_id * 16 + half * 8 + (lane_id / 4);
                const int q_pos = q_block_id * BLOCK_Q + row;
                #pragma unroll
                for (int p4 = 0; p4 < 8; p4++) {
                    #pragma unroll
                    for (int s2 = 0; s2 < 2; s2++) {
                        const int col = p4 * 8 + s2 + (lane_id % 4) * 2;
                        const int kv_pos = kv_id * BLOCK_KV + col;
                        if (kv_pos > q_pos)
                            row_vals[p4 * 2 + s2] = -FLT_MAX;
                    }
                }
            }

            /* Local max across 16 values */
            float local_max = row_vals[0];
            #pragma unroll
            for (int c = 1; c < 16; c++)
                local_max = fmaxf(local_max, row_vals[c]);

            /* Warp reduce max across 4 threads (lane%4 groups -> full 64-col) */
            local_max = fmaxf(local_max,
                __shfl_xor_sync(0xFFFFFFFF, local_max, 1));
            local_max = fmaxf(local_max,
                __shfl_xor_sync(0xFFFFFFFF, local_max, 2));

            /* Update running max */
            float new_max = fmaxf(local_max, rowmax[half]);
            float rescale = fast_exp2f(rowmax[half] - new_max);
            rowmax[half] = new_max;

            /* Rescale O accumulator for this row */
            #pragma unroll
            for (int p4 = 0; p4 < 16; p4++) {
                O_acc[(p4 << 2) | (half << 1) | 0] *= rescale;
                O_acc[(p4 << 2) | (half << 1) | 1] *= rescale;
            }

            /* Compute exp2(x - max) and sum */
            float local_sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < 16; c++) {
                row_vals[c] = fast_exp2f(row_vals[c] - new_max);
                local_sum += row_vals[c];
            }

            /* Warp reduce sum */
            local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, 1);
            local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, 2);

            /* Update running sum */
            rowsumexp[half] = rowsumexp[half] * rescale + local_sum;

            /* Write softmax output back to S_acc */
            #pragma unroll
            for (int p4 = 0; p4 < 8; p4++) {
                S_acc[(p4 << 2) | (half << 1) | 0] = row_vals[p4 * 2 + 0];
                S_acc[(p4 << 2) | (half << 1) | 1] = row_vals[p4 * 2 + 1];
            }
        }

        /* == Step 4: Wait for V[kv_id] TMA load == */
        mbarrier_wait_parity(mbar_V, v_phase);
        v_phase ^= 1;

        /* == Step 5: PV GEMM using RS mode (P from registers, V from SMEM) ==
         * V is in split-DIM layout: V_lo (cols 0-63) + V_hi (cols 64-127).
         * PV is m64n128k16: n=128 = DIM, so we need the full 128 columns.
         * For tnspB=1 (V transposed): k-steps advance through rows of V.
         * Each k-step processes 16 rows of V.
         * BLOCK_KV=64 → 4 k-steps.
         *
         * For split-DIM V: the n=128 output dimension spans both lo and hi.
         * We need wgmma m64n128k16 where B descriptor covers 128 columns.
         * But our V is split into two 64-col halves.
         *
         * Solution: use wgmma m64n64k16 for each half separately.
         * PV_lo = P × V_lo  (output cols 0-63)
         * PV_hi = P × V_hi  (output cols 64-127)
         *
         * Wait — this doesn't work because m64n64k16 with RS would need
         * different O_acc layout. Instead, we should use the full m64n128k16
         * with a descriptor that spans both halves.
         *
         * For m64n128k16 with SWIZZLE_128B and tnspB=1:
         * The B descriptor needs to cover 128 output columns and 16 input rows.
         * With split-DIM, V_lo and V_hi are contiguous in SMEM (V_lo at 48KB,
         * V_hi at 56KB). Each half is 8KB.
         * For tnspB=1, the descriptor stride advances through rows.
         *
         * Actually, for PV with tnspB=1 (transposed V):
         * - N dimension = DIM = 128 (output width)
         * - K dimension = BLOCK_KV = 64 (reduction over KV positions)
         * - The B matrix is V^T: each k-step reads 16 "rows" of V
         *   (which are 16 KV positions), producing 128 output columns.
         *
         * With SWIZZLE_128B and split-DIM layout:
         * V_lo has 64 rows × 64 cols = 64 values per row.
         * For tnspB=1, wgmma reads 128 columns but we only have 64 per half.
         *
         * We must use TWO m64n64k16 RS GEMMs instead:
         *   PV_lo = P × V_lo^T → O_acc[0:31]  (cols 0-63)
         *   PV_hi = P × V_hi^T → O_acc[32:63] (cols 64-127)
         * Each produces 64 output columns, mapping to the lo/hi halves.
         */

        wgmma_fence();

        /* V k-stepping with tnspB=1 and split-DIM:
         * Each k-step processes 16 rows of V. With split-DIM SWIZZLE_128B:
         * 16 rows = 2 groups of 8 rows. Group stride = 1024B.
         * k-step advance = 2 × 1024B / 16 = 128 units of 16B. */
        constexpr int V_KS_ADVANCE = 2 * SPLIT_STRIDE_BYTES / 16;  /* 128 */

        /* PV_lo: P × V_lo^T → output cols 0-63 (O_acc[0:31]) */
        #pragma unroll
        for (int ks = 0; ks < BLOCK_KV / 16; ks++) {
            uint32_t a0 = pack_bf16(S_acc[ks*8 + 0], S_acc[ks*8 + 1]);
            uint32_t a1 = pack_bf16(S_acc[ks*8 + 2], S_acc[ks*8 + 3]);
            uint32_t a2 = pack_bf16(S_acc[ks*8 + 4], S_acc[ks*8 + 5]);
            uint32_t a3 = pack_bf16(S_acc[ks*8 + 6], S_acc[ks*8 + 7]);
            uint64_t desc_v = gmma_desc_advance(desc_v_lo, ks * V_KS_ADVANCE);
            wgmma_m64n64k16_f32_bf16_RS(O_acc, a0, a1, a2, a3, desc_v, 1);
        }

        /* PV_hi: P × V_hi^T → output cols 64-127 (O_acc[32:63]) */
        #pragma unroll
        for (int ks = 0; ks < BLOCK_KV / 16; ks++) {
            uint32_t a0 = pack_bf16(S_acc[ks*8 + 0], S_acc[ks*8 + 1]);
            uint32_t a1 = pack_bf16(S_acc[ks*8 + 2], S_acc[ks*8 + 3]);
            uint32_t a2 = pack_bf16(S_acc[ks*8 + 4], S_acc[ks*8 + 5]);
            uint32_t a3 = pack_bf16(S_acc[ks*8 + 6], S_acc[ks*8 + 7]);
            uint64_t desc_v = gmma_desc_advance(desc_v_hi, ks * V_KS_ADVANCE);
            wgmma_m64n64k16_f32_bf16_RS(O_acc + 32, a0, a1, a2, a3, desc_v, 1);
        }

        wgmma_commit_group();
        wgmma_wait_group<0>();

        /* == Step 6: Wait for K[kv_id+1] TMA load == */
        if (has_next_k) {
            unsigned *phase_ptr = cur_stage ? &k0_phase : &k1_phase;
            uint64_t *mbar_k_next = cur_stage ? mbar_K0 : mbar_K1;
            mbarrier_wait_parity(mbar_k_next, *phase_ptr);
            *phase_ptr ^= 1;
        }

    } /* end kv_id loop */

    /* ---- Epilogue: finalize O and store to gmem ---- */
    wgmma_fence();

    #pragma unroll
    for (int half = 0; half < 2; half++) {
        float inv_sum = fast_rcp(rowsumexp[half]);
        #pragma unroll
        for (int p4 = 0; p4 < 16; p4++) {
            O_acc[(p4 << 2) | (half << 1) | 0] *= inv_sum;
            O_acc[(p4 << 2) | (half << 1) | 1] *= inv_sum;
        }
    }

    /* Write O to global memory */
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        const int row = warp_id * 16 + half * 8 + (lane_id / 4);
        if (row < len_q - q_block_id * BLOCK_Q) {
            #pragma unroll
            for (int p4 = 0; p4 < 16; p4++) {
                #pragma unroll
                for (int s2 = 0; s2 < 2; s2++) {
                    const int col = p4 * 8 + s2 + (lane_id % 4) * 2;
                    const int idx = (p4 << 2) | (half << 1) | s2;
                    nv_bfloat16 val = __float2bfloat16(O_acc[idx]);
                    O_base[row * seq_stride + col] = val;
                }
            }
        }
    }

    /* Invalidate mbarriers before exit */
    __syncthreads();
    if (tid == 0) {
        mbarrier_inval(mbar_Q);
        mbarrier_inval(mbar_K0);
        mbarrier_inval(mbar_K1);
        mbarrier_inval(mbar_V);
    }
}

/* ======================================================================
 *  TMA descriptor creation + kernel_run
 * ====================================================================== */

/* Dynamic loader for cuTensorMapEncodeTiled (CUDA 12.0+ driver API) */
typedef CUresult (*cuTensorMapEncodeTiled_fn)(
    CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
    const cuuint64_t*, const cuuint64_t*,
    const cuuint32_t*, const cuuint32_t*,
    CUtensorMapInterleave, CUtensorMapSwizzle,
    CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static cuTensorMapEncodeTiled_fn s_encodeTiled = nullptr;

static bool init_tma_encoder() {
    if (s_encodeTiled) return true;
    void *handle = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!handle) return false;

    /* Try direct symbol first */
    s_encodeTiled = reinterpret_cast<cuTensorMapEncodeTiled_fn>(
        dlsym(handle, "cuTensorMapEncodeTiled"));
    if (s_encodeTiled) return true;

    /* Fallback: cuGetProcAddress */
    typedef CUresult (*getProc_fn)(const char*, void**, int,
                                   cuuint64_t, CUdriverProcAddressQueryResult*);
    getProc_fn getProc = reinterpret_cast<getProc_fn>(
        dlsym(handle, "cuGetProcAddress_v2"));
    if (!getProc) return false;

    CUdriverProcAddressQueryResult status;
    CUresult res = getProc("cuTensorMapEncodeTiled",
                           reinterpret_cast<void**>(&s_encodeTiled),
                           12000, CU_GET_PROC_ADDRESS_DEFAULT, &status);
    return (res == CUDA_SUCCESS && s_encodeTiled);
}

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

    /* Initialize TMA encoder */
    if (!init_tma_encoder()) return -3;

    const int BLOCK_Q   = 64;
    const int BLOCK_KV  = 64;
    const int DIM_CONST = 128;
    const int HALF_DIM  = 64;
    const int TB_SIZE   = 128;

    int BH = B * H;

    /* Create TMA descriptors on host, one per (batch, head) */
    CUtensorMap *h_tma_Q = new CUtensorMap[BH];
    CUtensorMap *h_tma_K = new CUtensorMap[BH];
    CUtensorMap *h_tma_V = new CUtensorMap[BH];

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int idx = b * H + h;
            /* TMA sees a 2D tensor: [S rows, DIM cols] for each (batch, head).
             * Global dims: dim[0] = DIM (fast axis), dim[1] = S (slow axis).
             * Global stride: from row i to row i+1 = H * DIM * sizeof(bf16).
             * Box: [HALF_DIM cols, BLOCK_Q or BLOCK_KV rows].
             *
             * Base pointer: &Q[b * S * H * D + h * D]  (start of this head). */
            void *q_base = static_cast<void*>(
                const_cast<nv_bfloat16*>(inputs[0]) + b * S * H * D + h * D);
            void *k_base = static_cast<void*>(
                const_cast<nv_bfloat16*>(inputs[1]) + b * S * H * D + h * D);
            void *v_base = static_cast<void*>(
                const_cast<nv_bfloat16*>(inputs[2]) + b * S * H * D + h * D);

            cuuint64_t globalDim[2]    = {(cuuint64_t)DIM_CONST, (cuuint64_t)S};
            cuuint64_t globalStride[1] = {(cuuint64_t)(H * DIM_CONST * sizeof(nv_bfloat16))};
            cuuint32_t boxDim_Q[2]     = {(cuuint32_t)HALF_DIM, (cuuint32_t)BLOCK_Q};
            cuuint32_t boxDim_KV[2]    = {(cuuint32_t)HALF_DIM, (cuuint32_t)BLOCK_KV};
            cuuint32_t elemStrides[2]  = {1, 1};

            CUresult res;
            res = s_encodeTiled(&h_tma_Q[idx],
                CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, q_base,
                globalDim, globalStride, boxDim_Q, elemStrides,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            if (res != CUDA_SUCCESS) { delete[] h_tma_Q; delete[] h_tma_K; delete[] h_tma_V; return -4; }

            res = s_encodeTiled(&h_tma_K[idx],
                CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, k_base,
                globalDim, globalStride, boxDim_KV, elemStrides,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            if (res != CUDA_SUCCESS) { delete[] h_tma_Q; delete[] h_tma_K; delete[] h_tma_V; return -5; }

            res = s_encodeTiled(&h_tma_V[idx],
                CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, v_base,
                globalDim, globalStride, boxDim_KV, elemStrides,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            if (res != CUDA_SUCCESS) { delete[] h_tma_Q; delete[] h_tma_K; delete[] h_tma_V; return -6; }
        }
    }

    /* Copy TMA descriptors to device */
    CUtensorMap *d_tma_Q, *d_tma_K, *d_tma_V;
    cudaMalloc(&d_tma_Q, BH * sizeof(CUtensorMap));
    cudaMalloc(&d_tma_K, BH * sizeof(CUtensorMap));
    cudaMalloc(&d_tma_V, BH * sizeof(CUtensorMap));
    cudaMemcpyAsync(d_tma_Q, h_tma_Q, BH * sizeof(CUtensorMap),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_tma_K, h_tma_K, BH * sizeof(CUtensorMap),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_tma_V, h_tma_V, BH * sizeof(CUtensorMap),
                    cudaMemcpyHostToDevice, stream);

    int num_blocks = B * H * cdiv(S, BLOCK_Q);

    /* SMEM: 8 × 8KB tiles + 32B mbarriers = 65568B */
    int smem_size = 8 * BLOCK_Q * HALF_DIM * (int)sizeof(nv_bfloat16)  /* 64KB */
                  + 4 * (int)sizeof(uint64_t);                          /* 32B */

    auto kernel = flash_attention_tma_wgmma<BLOCK_Q, BLOCK_KV, DIM_CONST>;
    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    kernel<<<num_blocks, TB_SIZE, smem_size, stream>>>(
        inputs[0], inputs[1], inputs[2], outputs[0],
        B, S, H, S, S, causal ? 1 : 0,
        d_tma_Q, d_tma_K, d_tma_V);

    /* Cleanup: sync and free TMA descriptors */
    cudaStreamSynchronize(stream);
    cudaFree(d_tma_Q);
    cudaFree(d_tma_K);
    cudaFree(d_tma_V);
    delete[] h_tma_Q;
    delete[] h_tma_K;
    delete[] h_tma_V;

    return 0;
}
