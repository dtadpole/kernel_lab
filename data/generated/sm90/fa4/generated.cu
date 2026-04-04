/*
 * Flash Attention forward pass — BF16, SM90 WGMMA kernel (split-DIM TMA layout).
 * Unified 128 threads (1 warp group) + double-buffered K + dedicated V.
 *
 * Architecture: 128 threads, all threads cooperate on WGMMA compute.
 * TMA loads with SWIZZLE_128B: boxDim[0]=64 (128B / 2B per BF16).
 * DIM=128 requires TWO TMA loads per tile (cols 0-63 and cols 64-127).
 *
 * Split-DIM SMEM layout: lo (cols 0-63) and hi (cols 64-127) halves
 * stored in SEPARATE contiguous 8KB sections:
 *   Q_lo:    8KB at SMEM + 0
 *   Q_hi:    8KB at SMEM + 8KB
 *   K[0]_lo: 8KB at SMEM + 16KB
 *   K[0]_hi: 8KB at SMEM + 24KB
 *   K[1]_lo: 8KB at SMEM + 32KB
 *   K[1]_hi: 8KB at SMEM + 40KB
 *   V_lo:    8KB at SMEM + 48KB
 *   V_hi:    8KB at SMEM + 56KB
 *   mbarriers: 32B at SMEM + 64KB
 *   Total: 64KB + 32B
 *
 * Each 8KB section: 8 groups x (8 rows x 128 bytes) with Swizzle<3,4,3>.
 * WGMMA stride = 1024B (one atom per group, not two).
 *
 * Pipeline: V[i] and K[i+1] loads overlap with QK[i] + softmax compute.
 * PV GEMM uses RS mode with TWO m64n64k16 calls per K-step
 * (one for V_lo -> O_lo, one for V_hi -> O_hi).
 *
 * Constants: BLOCK_Q=64, BLOCK_KV=64, DIM=128, 128 threads.
 * Target: NVIDIA H100 (SM90a, GH100). Compile with -arch=sm_90a.
 *
 * kernel_run contract:
 *   extern "C" int kernel_run(__nv_bfloat16** inputs, int num_inputs,
 *                             __nv_bfloat16** outputs, int num_outputs,
 *                             int n, cudaStream_t stream);
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

/* --- mbarrier helpers (for TMA synchronization) ---------------------- */

__device__ __forceinline__
void mbarrier_init(uint64_t* mbar, unsigned count) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(addr), "r"(count));
}

__device__ __forceinline__
void mbarrier_inval(uint64_t* mbar) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(addr));
}

__device__ __forceinline__
void mbarrier_arrive_expect_tx(uint64_t* mbar, unsigned tx_bytes) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
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

/* --- TMA load (cp.async.bulk.tensor.2d) ------------------------------ */

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

/* --- WGMMA descriptor construction and advancement ------------------- */

/* K-major descriptor (tnspA=0 / tnspB=0): LBO = 1 (per CUTLASS convention) */
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

/* --- WGMMA m64n64k16 SS mode (QK GEMM) ------------------------------ */

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

/* --- WGMMA m64n64k16 RS mode (PV GEMM: A from registers, B from SMEM) --- */

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

/* --- Pack two F32 values into BF16x2 -------------------------------- */

__device__ __forceinline__
uint32_t pack_bf16(float a, float b) {
    uint32_t result;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(result) : "f"(a), "f"(b));
    return result;
}

/* --- fence_view_async_shared ----------------------------------------- */

__device__ __forceinline__
void fence_view_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

/* ======================================================================
 *  WGMMA Flash Attention kernel — Split-DIM TMA layout
 *
 *  128 threads = 1 warp group (warps 0-3).
 *  Double-buffered K: loads K[i+1] during QK[i] + softmax.
 *  Dedicated V buffer: V[i] loads overlap with QK[i] + softmax.
 *  PV GEMM: RS mode with TWO m64n64k16 per K-step (V_lo->O_lo, V_hi->O_hi).
 *
 *  Split-DIM SMEM layout (8 sections x 8KB = 64KB):
 *    Q_lo, Q_hi, K0_lo, K0_hi, K1_lo, K1_hi, V_lo, V_hi
 * ====================================================================== */

template<int BLOCK_Q, int BLOCK_KV, int DIM>
__launch_bounds__(128, 1)   /* 128 threads, 1 block/SM */
__global__
void flash_attention_wgmma(
    const nv_bfloat16 *Q,  /* unused with TMA */
    const nv_bfloat16 *K,  /* unused with TMA */
    const nv_bfloat16 *V,  /* unused with TMA */
    nv_bfloat16 *O,
    int B, int S, int H,
    int len_q, int len_kv,
    int is_causal,
    const CUtensorMap *tma_Q_arr,
    const CUtensorMap *tma_K_arr,
    const CUtensorMap *tma_V_arr)
{
    /* Constants */
    constexpr int TB_SIZE = 128;
    constexpr int HALF_DIM = DIM / 2;          /* 64 */
    constexpr int HALF_BYTES = BLOCK_Q * HALF_DIM * (int)sizeof(nv_bfloat16);  /* 8192 = 8KB */

    /* Split-DIM SMEM layout:
     *   Section 0: Q_lo    (8KB) at offset 0
     *   Section 1: Q_hi    (8KB) at offset 8KB
     *   Section 2: K0_lo   (8KB) at offset 16KB
     *   Section 3: K0_hi   (8KB) at offset 24KB
     *   Section 4: K1_lo   (8KB) at offset 32KB
     *   Section 5: K1_hi   (8KB) at offset 40KB
     *   Section 6: V_lo    (8KB) at offset 48KB
     *   Section 7: V_hi    (8KB) at offset 56KB
     *   mbarriers: 4x8B at offset 64KB
     *   Total: 64KB + 32B */
    extern __shared__ nv_bfloat16 smem[];
    const uint32_t smem_base = __cvta_generic_to_shared(smem);

    /* Section offsets (each 8KB) */
    const uint32_t Q_lo_smem   = smem_base;                         /* offset 0 */
    const uint32_t Q_hi_smem   = smem_base + HALF_BYTES;            /* offset 8KB */
    const uint32_t K0_lo_smem  = smem_base + 2 * HALF_BYTES;        /* offset 16KB */
    const uint32_t K0_hi_smem  = smem_base + 3 * HALF_BYTES;        /* offset 24KB */
    const uint32_t K1_lo_smem  = smem_base + 4 * HALF_BYTES;        /* offset 32KB */
    const uint32_t K1_hi_smem  = smem_base + 5 * HALF_BYTES;        /* offset 40KB */
    const uint32_t V_lo_smem   = smem_base + 6 * HALF_BYTES;        /* offset 48KB */
    const uint32_t V_hi_smem   = smem_base + 7 * HALF_BYTES;        /* offset 56KB */

    /* mbarrier storage at end of SMEM (aligned to 8 bytes) */
    constexpr int TILE_DATA_BYTES = 8 * HALF_BYTES;  /* 64KB */
    uint64_t *mbar_Q  = (uint64_t*)((char*)smem + TILE_DATA_BYTES);
    uint64_t *mbar_K0 = mbar_Q + 1;
    uint64_t *mbar_K1 = mbar_Q + 2;
    uint64_t *mbar_V  = mbar_Q + 3;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;         /* 0-3 */
    const int lane_id = tid % 32;

    /* Block -> (batch, head, q_block) mapping */
    const int bid = blockIdx.x;
    const int num_q_blocks = cdiv(len_q, BLOCK_Q);
    const int bs_id       = bid / num_q_blocks;
    const int q_block_id  = bid % num_q_blocks;
    const int batch_id    = bs_id / H;
    const int head_id     = bs_id % H;
    const int seq_stride  = H * DIM;
    const int bh_id       = batch_id * H + head_id;

    /* TMA descriptors for this (batch, head) pair */
    const CUtensorMap *tma_Q = &tma_Q_arr[bh_id];
    const CUtensorMap *tma_K = &tma_K_arr[bh_id];
    const CUtensorMap *tma_V = &tma_V_arr[bh_id];

    /* Output pointer (still direct global store) */
    nv_bfloat16 *O_base = O + batch_id * S * seq_stride + head_id * DIM
                             + q_block_id * BLOCK_Q * seq_stride;

    /* Each TMA load transfers one half (64 cols x BLOCK rows x 2B = 8KB).
     * A full tile = 2 halves = 16KB. */
    constexpr int FULL_TILE_BYTES = 2 * HALF_BYTES;  /* 16384 = 16KB */

    /* KV iteration bounds */
    const int num_kv_iter = cdiv(len_kv, BLOCK_KV);
    const int max_kv_iter = is_causal
        ? min(num_kv_iter, cdiv((q_block_id + 1) * BLOCK_Q, BLOCK_KV))
        : num_kv_iter;

    /* ---- Initialize mbarriers (thread 0 only) ---- */
    if (tid == 0) {
        mbarrier_init(mbar_Q,  1);  /* 1 TMA thread arrives */
        mbarrier_init(mbar_K0, 1);
        mbarrier_init(mbar_K1, 1);
        mbarrier_init(mbar_V,  1);
    }
    __syncthreads();

    /* ---- TMA load Q to shared memory (thread 0 only) ---- */
    /* Two TMA loads: lo half (cols 0-63) -> Q_lo_smem, hi half (cols 64-127) -> Q_hi_smem */
    if (tid == 0) {
        mbarrier_arrive_expect_tx(mbar_Q, FULL_TILE_BYTES);
        tma_load_2d((void*)(smem_base),                tma_Q, 0,  q_block_id * BLOCK_Q, mbar_Q);   /* lo */
        tma_load_2d((void*)(smem_base + HALF_BYTES),   tma_Q, 64, q_block_id * BLOCK_Q, mbar_Q);   /* hi */
    }
    mbarrier_wait_parity(mbar_Q, 0);

    /* WGMMA descriptor stride for split layout:
     * Each section has 1 atom (64 cols) per 8-row group.
     * Group stride = 8 rows x 64 cols x 2B = 1024 bytes. */
    constexpr int SPLIT_STRIDE_BYTES = 8 * HALF_DIM * (int)sizeof(nv_bfloat16);  /* 1024 */

    const float softmax_scale_log2 =
        rsqrtf(static_cast<float>(DIM)) * 1.4426950408889634f;

    /* Initialize accumulators: O_lo[32] and O_hi[32] for split output */
    float O_lo[32], O_hi[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) { O_lo[i] = 0.0f; O_hi[i] = 0.0f; }

    float rowmax[2] = {-FLT_MAX, -FLT_MAX};
    float rowsumexp[2] = {0.0f, 0.0f};

    /* Precompute base descriptors for Q (invariant across iterations) */
    const uint64_t desc_q_lo = make_wgmma_desc(Q_lo_smem, SPLIT_STRIDE_BYTES);
    const uint64_t desc_q_hi = make_wgmma_desc(Q_hi_smem, SPLIT_STRIDE_BYTES);

    /* ---- Prelude: TMA load K[0] into stage 0 (thread 0) ---- */
    if (max_kv_iter > 0 && tid == 0) {
        mbarrier_arrive_expect_tx(mbar_K0, FULL_TILE_BYTES);
        tma_load_2d((void*)(K0_lo_smem), tma_K, 0,  0, mbar_K0);   /* lo */
        tma_load_2d((void*)(K0_hi_smem), tma_K, 64, 0, mbar_K0);   /* hi */
    }
    if (max_kv_iter > 0)
        mbarrier_wait_parity(mbar_K0, 0);

    /* Precompute V descriptor bases (will be set per iteration after V load) */
    /* V_lo and V_hi share the same stride (1024B) and LBO=1 */

    /* K mbarrier phase tracking (alternates 0/1 for double-buffer) */
    int k_phase[2] = {1, 0};  /* K[0] already consumed phase 0 */

    /* ---- Main KV loop ---- */
    for (int kv_id = 0; kv_id < max_kv_iter; kv_id++) {
        const int cur_stage  = kv_id & 1;

        /* Current K smem addresses based on stage */
        const uint32_t cur_K_lo = cur_stage ? K1_lo_smem : K0_lo_smem;
        const uint32_t cur_K_hi = cur_stage ? K1_hi_smem : K0_hi_smem;

        /* == Step 1a: TMA load V[kv_id] into V_smem (thread 0, non-blocking) == */
        if (tid == 0) {
            mbarrier_arrive_expect_tx(mbar_V, FULL_TILE_BYTES);
            tma_load_2d((void*)(V_lo_smem), tma_V, 0,  kv_id * BLOCK_KV, mbar_V);   /* lo */
            tma_load_2d((void*)(V_hi_smem), tma_V, 64, kv_id * BLOCK_KV, mbar_V);   /* hi */
        }

        /* == Step 1b: TMA load K[kv_id+1] (thread 0, non-blocking) == */
        const bool has_next_k = (kv_id + 1 < max_kv_iter);
        if (has_next_k && tid == 0) {
            uint64_t *mbar_k_next = (cur_stage == 0) ? mbar_K1 : mbar_K0;
            const uint32_t next_K_lo = cur_stage ? K0_lo_smem : K1_lo_smem;
            const uint32_t next_K_hi = cur_stage ? K0_hi_smem : K1_hi_smem;
            mbarrier_arrive_expect_tx(mbar_k_next, FULL_TILE_BYTES);
            tma_load_2d((void*)(next_K_lo), tma_K, 0,  (kv_id + 1) * BLOCK_KV, mbar_k_next);   /* lo */
            tma_load_2d((void*)(next_K_hi), tma_K, 64, (kv_id + 1) * BLOCK_KV, mbar_k_next);   /* hi */
        }

        /* == Step 2: QK GEMM == */
        /* Split-DIM: ks 0-3 use lo descriptors, ks 4-7 use hi descriptors.
         * Each half has 4 k-steps of 16 cols = 64 cols. */
        float S_acc[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) S_acc[i] = 0.0f;

        const uint64_t desc_k_lo = make_wgmma_desc(cur_K_lo, SPLIT_STRIDE_BYTES);
        const uint64_t desc_k_hi = make_wgmma_desc(cur_K_hi, SPLIT_STRIDE_BYTES);

        wgmma_fence();

        #pragma unroll
        for (int ks = 0; ks < DIM / 16; ks++) {
            uint64_t desc_q, desc_k;
            if (ks < 4) {
                desc_q = gmma_desc_advance(desc_q_lo, ks * 2);
                desc_k = gmma_desc_advance(desc_k_lo, ks * 2);
            } else {
                desc_q = gmma_desc_advance(desc_q_hi, (ks - 4) * 2);
                desc_k = gmma_desc_advance(desc_k_hi, (ks - 4) * 2);
            }
            wgmma_m64n64k16_f32_bf16(S_acc, desc_q, desc_k,
                                      (ks == 0) ? 0 : 1);
        }

        wgmma_commit_group();
        wgmma_wait_group<0>();
        wgmma_fence();  /* fence before reading accumulators */

        /* == Step 3: Softmax on S_acc == */
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

            /* Rescale O accumulators for this row (both lo and hi halves) */
            #pragma unroll
            for (int p4 = 0; p4 < 8; p4++) {
                O_lo[(p4 << 2) | (half << 1) | 0] *= rescale;
                O_lo[(p4 << 2) | (half << 1) | 1] *= rescale;
                O_hi[(p4 << 2) | (half << 1) | 0] *= rescale;
                O_hi[(p4 << 2) | (half << 1) | 1] *= rescale;
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

            /* Write softmax output back to S_acc (P stays in registers) */
            #pragma unroll
            for (int p4 = 0; p4 < 8; p4++) {
                S_acc[(p4 << 2) | (half << 1) | 0] = row_vals[p4 * 2 + 0];
                S_acc[(p4 << 2) | (half << 1) | 1] = row_vals[p4 * 2 + 1];
            }
        }

        /* == Step 4: Wait for V[kv_id] TMA load to complete == */
        mbarrier_wait_parity(mbar_V, kv_id & 1);

        /* == Step 5: PV GEMM using RS mode ==
         * TWO m64n64k16 calls per K-step: V_lo -> O_lo, V_hi -> O_hi.
         *
         * V is in split layout: V_lo (64 rows x 64 cols), V_hi (64 rows x 64 cols).
         * Each section: 8 groups x 1024B. WGMMA stride = 1024B.
         * With tnspB=1, each K-step processes 16 rows of V.
         * 16 rows = 2 groups. Advance per K-step = 2 x 1024B / 16 = 128 (in 16B units).
         */
        const uint64_t desc_v_lo = make_wgmma_desc(V_lo_smem, SPLIT_STRIDE_BYTES);
        const uint64_t desc_v_hi = make_wgmma_desc(V_hi_smem, SPLIT_STRIDE_BYTES);

        /* V K-step advance: 2 groups x 1024B = 2048B per step. In 16B units: 128 */
        constexpr int V_KS_ADVANCE = 2 * SPLIT_STRIDE_BYTES / 16;  /* 128 */

        wgmma_fence();

        #pragma unroll
        for (int ks = 0; ks < BLOCK_KV / 16; ks++) {
            uint32_t a0 = pack_bf16(S_acc[ks*8 + 0], S_acc[ks*8 + 1]);
            uint32_t a1 = pack_bf16(S_acc[ks*8 + 2], S_acc[ks*8 + 3]);
            uint32_t a2 = pack_bf16(S_acc[ks*8 + 4], S_acc[ks*8 + 5]);
            uint32_t a3 = pack_bf16(S_acc[ks*8 + 6], S_acc[ks*8 + 7]);

            /* V_lo GEMM -> O_lo */
            uint64_t dv_lo = gmma_desc_advance(desc_v_lo, ks * V_KS_ADVANCE);
            wgmma_m64n64k16_f32_bf16_RS(O_lo, a0, a1, a2, a3, dv_lo, 1);

            /* V_hi GEMM -> O_hi */
            uint64_t dv_hi = gmma_desc_advance(desc_v_hi, ks * V_KS_ADVANCE);
            wgmma_m64n64k16_f32_bf16_RS(O_hi, a0, a1, a2, a3, dv_hi, 1);
        }

        wgmma_commit_group();
        wgmma_wait_group<0>();

        /* == Step 6: Wait for K[kv_id+1] TMA load if needed == */
        if (has_next_k) {
            uint64_t *mbar_k_next = (cur_stage == 0) ? mbar_K1 : mbar_K0;
            int next_phase = k_phase[1 - cur_stage];
            mbarrier_wait_parity(mbar_k_next, next_phase);
            k_phase[1 - cur_stage] ^= 1;  /* flip phase for next use */
        }

    } /* end kv_id loop */

    /* ---- Epilogue: finalize O and store to gmem ---- */
    wgmma_fence();  /* fence before reading O_lo, O_hi */

    #pragma unroll
    for (int half = 0; half < 2; half++) {
        float inv_sum = fast_rcp(rowsumexp[half]);
        #pragma unroll
        for (int p4 = 0; p4 < 8; p4++) {
            O_lo[(p4 << 2) | (half << 1) | 0] *= inv_sum;
            O_lo[(p4 << 2) | (half << 1) | 1] *= inv_sum;
            O_hi[(p4 << 2) | (half << 1) | 0] *= inv_sum;
            O_hi[(p4 << 2) | (half << 1) | 1] *= inv_sum;
        }
    }

    /* Write O to global memory.
     * m64n64k16 register layout:
     *   half = (i >> 1) & 1
     *   pair4 = i >> 2
     *   sub2 = i & 1
     *   row = warp*16 + half*8 + lane/4
     *   col = pair4*8 + sub2 + (lane%4)*2   (0-63 within each half)
     * O_lo covers output cols 0-63, O_hi covers output cols 64-127. */
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        const int row = warp_id * 16 + half * 8 + (lane_id / 4);
        if (row < len_q - q_block_id * BLOCK_Q) {
            #pragma unroll
            for (int p4 = 0; p4 < 8; p4++) {
                #pragma unroll
                for (int s2 = 0; s2 < 2; s2++) {
                    const int col_in_half = p4 * 8 + s2 + (lane_id % 4) * 2;
                    const int idx = (p4 << 2) | (half << 1) | s2;

                    /* O_lo: cols 0-63 */
                    nv_bfloat16 val_lo = __float2bfloat16(O_lo[idx]);
                    O_base[row * seq_stride + col_in_half] = val_lo;

                    /* O_hi: cols 64-127 */
                    nv_bfloat16 val_hi = __float2bfloat16(O_hi[idx]);
                    O_base[row * seq_stride + 64 + col_in_half] = val_hi;
                }
            }
        }
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

/* --- TMA descriptor encoder (loaded via dlopen) ---------------------- */

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

/* Device-side TMA descriptor arrays (allocated once, reused) */
static CUtensorMap* d_tma_Q = nullptr;
static CUtensorMap* d_tma_K = nullptr;
static CUtensorMap* d_tma_V = nullptr;
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
        const int BLOCK_Q   = 64;
        const int BLOCK_KV  = 64;
        const int DIM_CONST = 128;
        const int TB_SIZE   = 128;

        int num_blocks = B * H * cdiv(S, BLOCK_Q);

        /* SMEM: 8 sections x 8KB + mbarrier storage = 64KB + 64B */
        int smem_size = 8 * (BLOCK_Q * (DIM_CONST / 2) * (int)sizeof(nv_bfloat16))
                      + 128;  /* mbarrier storage (aligned) */

        /* --- Create TMA descriptors for all (batch, head) pairs --- */
        int bh = B * H;
        if (bh > d_tma_bh_count) {
            if (d_tma_Q) cudaFree(d_tma_Q);
            if (d_tma_K) cudaFree(d_tma_K);
            if (d_tma_V) cudaFree(d_tma_V);
            cudaMalloc(&d_tma_Q, bh * sizeof(CUtensorMap));
            cudaMalloc(&d_tma_K, bh * sizeof(CUtensorMap));
            cudaMalloc(&d_tma_V, bh * sizeof(CUtensorMap));
            d_tma_bh_count = bh;
        }

        /* Host-side TMA descriptor creation */
        CUtensorMap* h_tma_Q = (CUtensorMap*)malloc(bh * sizeof(CUtensorMap));
        CUtensorMap* h_tma_K = (CUtensorMap*)malloc(bh * sizeof(CUtensorMap));
        CUtensorMap* h_tma_V = (CUtensorMap*)malloc(bh * sizeof(CUtensorMap));

        const __nv_bfloat16 *Q_ptr = inputs[0];
        const __nv_bfloat16 *K_ptr = inputs[1];
        const __nv_bfloat16 *V_ptr = inputs[2];
        int seq_stride = H * D;

        for (int b = 0; b < B; b++) {
            for (int h = 0; h < H; h++) {
                int idx = b * H + h;
                void* q_base = (void*)(Q_ptr + b * S * seq_stride + h * D);
                void* k_base = (void*)(K_ptr + b * S * seq_stride + h * D);
                void* v_base = (void*)(V_ptr + b * S * seq_stride + h * D);

                /* TMA with SWIZZLE_128B: boxDim[0] = 64 for BF16 (128B / 2B).
                 * DIM=128 needs two TMA loads per tile: cols 0-63 and cols 64-127.
                 * Use ONE descriptor with boxDim={64, BLOCK} and coord0=0 or 64. */
                cuuint64_t dims[2] = {(cuuint64_t)D, (cuuint64_t)S};
                cuuint64_t strides[1] = {(cuuint64_t)(seq_stride * sizeof(nv_bfloat16))};
                cuuint32_t box_q[2] = {64, (cuuint32_t)BLOCK_Q};
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
            }
        }

        cudaMemcpyAsync(d_tma_Q, h_tma_Q, bh * sizeof(CUtensorMap),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_tma_K, h_tma_K, bh * sizeof(CUtensorMap),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_tma_V, h_tma_V, bh * sizeof(CUtensorMap),
                        cudaMemcpyHostToDevice, stream);

        auto kernel = flash_attention_wgmma<BLOCK_Q, BLOCK_KV, DIM_CONST>;
        cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        kernel<<<num_blocks, TB_SIZE, smem_size, stream>>>(
            nullptr, nullptr, nullptr, outputs[0],  /* Q/K/V now via TMA */
            B, S, H, S, S, causal ? 1 : 0,
            d_tma_Q, d_tma_K, d_tma_V);

        free(h_tma_Q);
        free(h_tma_K);
        free(h_tma_V);
    }

    return 0;
}
