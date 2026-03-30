/*
 * Flash Attention forward pass — BF16, SM80+ tensor-core MMA kernel.
 *
 * Adapted from attention_v5 (github.com/tspeterkim/flash-attention-minimal
 * style) with multi-head support and optional causal masking.
 *
 * Layout:
 *   Q, K, V, O  : [B, S, H, D]  (row-major __nv_bfloat16)
 *
 * Strategy: transpose to [B, H, S, D] so each (batch, head) pair is a
 * contiguous [S, D] attention instance — identical to the single-head v5
 * kernel. After attention, transpose O back to [B, S, H, D].
 *
 * Constants: BLOCK_Q=128, BLOCK_KV=64, DIM=128, NUM_WARPS=4.
 *
 * Register strategy: on-the-fly SMEM→register loading for K and V fragments
 * to stay within 255 registers at BLOCK_Q=128. K_rmem and V_rmem are
 * eliminated; instead, 2-register fragments are loaded and consumed
 * immediately inside the MMA inner loops.
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

/* ======================================================================
 *  Helper constants & device functions (inlined from common.h)
 * ====================================================================== */

static constexpr int WARP_SIZE = 32;

__device__ __host__ constexpr
int cdiv(int a, int b) { return (a + b - 1) / b; }

/* --- Swizzle --------------------------------------------------------- */

template <int STRIDE>
__device__
uint32_t swizzle(uint32_t index) {
    if constexpr (STRIDE == 16)
        return index;
    uint32_t row_idx = (index / STRIDE) % 8;
    uint32_t bits_to_xor = row_idx / max(64 / STRIDE, 1);
    return index ^ (bits_to_xor << 4);
}

/* --- cp.async global -> shared (swizzled) ----------------------------- */

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared_swizzle(uint32_t dst, const nv_bfloat16 *src,
                              int src_stride, int tid) {
    constexpr int num_elems = 16 / sizeof(nv_bfloat16);          /* 8 */
    constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

    for (int iter = 0; iter < num_iters; iter++) {
        const int idx = (iter * TB_SIZE + tid) * num_elems;
        const int row = idx / WIDTH;
        const int col = idx % WIDTH;

        const uint32_t dst_addr =
            swizzle<WIDTH * sizeof(nv_bfloat16)>(
                dst + (row * WIDTH + col) * sizeof(nv_bfloat16));
        const nv_bfloat16 *src_addr = src + (row * src_stride + col);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :: "r"(dst_addr), "l"(src_addr));
    }
}

/* --- ldmatrix helpers ------------------------------------------------- */

__device__ inline
void ldmatrix_x2(uint32_t regs[2], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(regs[0]), "=r"(regs[1])
        : "r"(addr));
}

__device__ inline
void ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "r"(addr));
}

__device__ inline
void ldmatrix_x2_trans(uint32_t regs[2], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(regs[0]), "=r"(regs[1])
        : "r"(addr));
}

__device__ inline
void ldmatrix_x4_trans(uint32_t regs[4], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "r"(addr));
}

/* --- mma.m16n8k16 ---------------------------------------------------- */

__device__ inline
void mma_m16n8k16(uint32_t A[4], uint32_t B[2], float D[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]),
          "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

/* ======================================================================
 *  Transpose kernel:  [B, S, H, D] <-> [B, H, S, D]
 *
 *  forward  = true  → BSHD -> BHSD
 *  forward  = false → BHSD -> BSHD
 * ====================================================================== */

__global__ void transpose_bshd_bhsd(
    const __nv_bfloat16 * __restrict__ in,
    __nv_bfloat16 * __restrict__ out,
    int B, int S, int H, int D, int forward)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * H * D;
    if (idx >= total) return;

    /* Decompose linear index into (b, s, h, d) or (b, h, s, d) */
    int d = idx % D;
    int tmp = idx / D;

    if (forward) {
        /* in  layout: [B, S, H, D]  → out layout: [B, H, S, D] */
        int h = tmp % H;   tmp /= H;
        int s = tmp % S;   tmp /= S;
        int b = tmp;
        out[((b * H + h) * S + s) * D + d] = in[idx];
    } else {
        /* in  layout: [B, H, S, D]  → out layout: [B, S, H, D] */
        int s = tmp % S;   tmp /= S;
        int h = tmp % H;   tmp /= H;
        int b = tmp;
        out[((b * S + s) * H + h) * D + d] = in[idx];
    }
}

/* ======================================================================
 *  Flash Attention v5 kernel (adapted with causal mask support)
 *
 *  Q, K, V, O are already in [effective_bs, S, D] contiguous layout
 *  (after BSHD -> BHSD transpose, effective_bs = B * H).
 * ====================================================================== */

template<int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS>
__launch_bounds__(NUM_WARPS * WARP_SIZE)
__global__
void flash_attention_kernel(
    const nv_bfloat16 *Q,   /* [effective_bs, len_q, DIM]  */
    const nv_bfloat16 *K,   /* [effective_bs, len_kv, DIM] */
    const nv_bfloat16 *V,   /* [effective_bs, len_kv, DIM] */
    nv_bfloat16 *O,         /* [effective_bs, len_q, DIM]  */
    int effective_bs,
    int len_q,
    int len_kv,
    int is_causal)
{
    constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    /* Each thread-block handles one BLOCK_Q chunk of one (batch, head). */
    const int num_q_blocks = cdiv(len_q, BLOCK_Q);
    const int bs_id       = bid / num_q_blocks;
    const int q_block_id  = bid % num_q_blocks;

    Q += bs_id * len_q * DIM + q_block_id * BLOCK_Q * DIM;
    K += bs_id * len_kv * DIM;
    V += bs_id * len_kv * DIM;
    O += bs_id * len_q * DIM + q_block_id * BLOCK_Q * DIM;

    /* Shared memory layout:
     *   Q region:  [BLOCK_Q, DIM] — persistent, Q loaded on-the-fly from SMEM
     *   KV region: K double-buffer [2, BLOCK_KV, DIM] + V single-buffer [BLOCK_KV, DIM]
     * The two regions do NOT overlap so Q stays resident while K/V rotate.
     * Total SMEM: max(Q_bytes, KV_bytes) when laid out sequentially:
     *   Q: 128*128*2 = 32KB,  KV: 3*64*128*2 = 48KB  → 80KB total
     */
    extern __shared__ nv_bfloat16 smem[];
    const uint32_t smem_base = __cvta_generic_to_shared(smem);
    const uint32_t Q_smem = smem_base;
    const uint32_t KV_base = smem_base + BLOCK_Q * DIM * sizeof(nv_bfloat16);
    const uint32_t K_smem = KV_base;
    const uint32_t V_smem = KV_base + 2 * BLOCK_KV * DIM * sizeof(nv_bfloat16);

    /* FA2-style: shard BLOCK_Q rows among warps */
    constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;

    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;

    /* Register tiles — Q, K and V are ALL loaded on-the-fly from SMEM.
     * Only O accumulator lives persistently in registers.
     * S and P are scoped per mma_id_q inside the main loop. */
    float    O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};

    /* Pre-compute swizzled SMEM addresses for ldmatrix */
    uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
    {
        /* A tile (Q) — ldmatrix x4 for m16n8k16 A operand */
        const int row_off = warp_id * WARP_Q + (lane_id % 16);
        const int col_off = lane_id / 16 * 8;
        Q_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(
            Q_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
    {
        /* B tile (K — non-transposed ldmatrix) */
        const int row_off = lane_id % 8;
        const int col_off = lane_id / 8 * 8;
        K_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(
            K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
    {
        /* B tile trans (V — transposed ldmatrix) */
        const int row_off = lane_id % 16;
        const int col_off = lane_id / 16 * 8;
        V_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(
            V_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }

    const float softmax_scale = rsqrtf(static_cast<float>(DIM));

    float rowmax[WARP_Q / MMA_M][2];
    float rowsumexp[WARP_Q / MMA_M][2] = {};
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
        rowmax[mma_id_q][0] = -FLT_MAX;
        rowmax[mma_id_q][1] = -FLT_MAX;
    }

    /* Load Q [BLOCK_Q, DIM] from global -> shared (persistent for all KV iters) */
    global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    const int num_kv_iter = cdiv(len_kv, BLOCK_KV);

    /* Causal early-exit bound: skip KV blocks entirely past the diagonal. */
    const int max_kv_iter = is_causal
        ? min(num_kv_iter, cdiv((q_block_id + 1) * BLOCK_Q, BLOCK_KV))
        : num_kv_iter;

    /* Local copy of K/V pointers (advanced by load lambdas) */
    const nv_bfloat16 *K_ptr = K;
    const nv_bfloat16 *V_ptr = V;

    /* Prefetch first K block */
    if (0 < max_kv_iter) {
        global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(
            K_smem, K_ptr, DIM, tid);
        K_ptr += BLOCK_KV * DIM;
    }
    asm volatile("cp.async.commit_group;");

    for (int kv_id = 0; kv_id < max_kv_iter; kv_id++) {

        /* Prefetch V (need sync to ensure previous V_smem is consumed) */
        __syncthreads();
        global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(
            V_smem, V_ptr, DIM, tid);
        V_ptr += BLOCK_KV * DIM;
        asm volatile("cp.async.commit_group;");

        /* Wait for K to arrive in SMEM */
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();

        const uint32_t K_base = K_smem_thread +
            (kv_id % 2) * (BLOCK_KV * DIM * sizeof(nv_bfloat16));

        /* Process each mma_id_q slice independently: QK → scale → mask →
         * softmax → pack P.  S_local is [8][4] = 32 regs instead of
         * [2][8][4] = 64, halving peak pressure during softmax. */
        uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];

        for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {

            /* ---- QK: S = Q[mma_id_q] @ K^T ---- */
            float S_local[BLOCK_KV / MMA_N][4] = {};

            /* Iterate over d-dimension first, loading Q once per d-step
             * and reusing it across all mma_id_kv. This reduces Q ldmatrix
             * count from (BLOCK_KV/MMA_N * DIM/MMA_K) to (DIM/MMA_K). */
            #pragma unroll
            for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
                uint32_t Q_frag[4];
                {
                    uint32_t qaddr = Q_smem_thread;
                    qaddr += mma_id_q * MMA_M * DIM * sizeof(nv_bfloat16);
                    qaddr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
                    ldmatrix_x4(Q_frag, qaddr);
                }
                #pragma unroll
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                    uint32_t K_frag[2];
                    {
                        uint32_t kaddr = K_base;
                        kaddr += mma_id_kv * MMA_N * DIM * sizeof(nv_bfloat16);
                        kaddr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
                        ldmatrix_x2(K_frag, kaddr);
                    }
                    mma_m16n8k16(Q_frag, K_frag, S_local[mma_id_kv]);
                }
            }

            /* ---- Softmax scale ---- */
            #pragma unroll
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
                #pragma unroll
                for (int reg_id = 0; reg_id < 4; reg_id++)
                    S_local[mma_id_kv][reg_id] *= softmax_scale;

            /* ---- Causal mask ---- */
            if (is_causal) {
                const int q_start = q_block_id * BLOCK_Q;
                const int kv_start = kv_id * BLOCK_KV;

                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                    int q_pos_0 = q_start + warp_id * WARP_Q +
                                  mma_id_q * MMA_M + (lane_id / 4);
                    int q_pos_1 = q_pos_0 + 8;
                    int kv_pos_0 = kv_start +
                                   mma_id_kv * MMA_N + (lane_id % 4) * 2;
                    int kv_pos_1 = kv_pos_0 + 1;

                    if (kv_pos_0 > q_pos_0)
                        S_local[mma_id_kv][0] = -FLT_MAX;
                    if (kv_pos_1 > q_pos_0)
                        S_local[mma_id_kv][1] = -FLT_MAX;
                    if (kv_pos_0 > q_pos_1)
                        S_local[mma_id_kv][2] = -FLT_MAX;
                    if (kv_pos_1 > q_pos_1)
                        S_local[mma_id_kv][3] = -FLT_MAX;
                }
            }

            /* ---- Online softmax + rescaling ---- */
            float this_rowmax[2];
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                float *regs = S_local[mma_id_kv];
                if (mma_id_kv == 0) {
                    this_rowmax[0] = max(regs[0], regs[1]);
                    this_rowmax[1] = max(regs[2], regs[3]);
                } else {
                    this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));
                    this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));
                }
            }

            /* Butterfly reduction within 4 threads */
            this_rowmax[0] = max(this_rowmax[0],
                __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 1));
            this_rowmax[0] = max(this_rowmax[0],
                __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 2));
            this_rowmax[1] = max(this_rowmax[1],
                __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 1));
            this_rowmax[1] = max(this_rowmax[1],
                __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 2));

            this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
            this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);

            /* Rescale previous O accumulator */
            float rescale[2];
            rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
            rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);
            #pragma unroll
            for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
                O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
                O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
                O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
                O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
            }

            rowmax[mma_id_q][0] = this_rowmax[0];
            rowmax[mma_id_q][1] = this_rowmax[1];

            /* Row sum-exp + pack S -> P */
            float this_rowsumexp[2];
            #pragma unroll
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                float *regs = S_local[mma_id_kv];
                regs[0] = __expf(regs[0] - rowmax[mma_id_q][0]);
                regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]);
                regs[2] = __expf(regs[2] - rowmax[mma_id_q][1]);
                regs[3] = __expf(regs[3] - rowmax[mma_id_q][1]);

                if (mma_id_kv == 0) {
                    this_rowsumexp[0] = regs[0] + regs[1];
                    this_rowsumexp[1] = regs[2] + regs[3];
                } else {
                    this_rowsumexp[0] += regs[0] + regs[1];
                    this_rowsumexp[1] += regs[2] + regs[3];
                }

                /* Pack to P registers (m16n8 -> m16k16 layout) */
                nv_bfloat162 *this_P =
                    reinterpret_cast<nv_bfloat162 *>(
                        P_rmem[mma_id_q][mma_id_kv / 2]);
                this_P[(mma_id_kv % 2) * 2] =
                    __float22bfloat162_rn({regs[0], regs[1]});
                this_P[(mma_id_kv % 2) * 2 + 1] =
                    __float22bfloat162_rn({regs[2], regs[3]});
            }

            /* Butterfly reduction for sumexp */
            this_rowsumexp[0] +=
                __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 1);
            this_rowsumexp[0] +=
                __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 2);
            this_rowsumexp[1] +=
                __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 1);
            this_rowsumexp[1] +=
                __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 2);

            rowsumexp[mma_id_q][0] =
                rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
            rowsumexp[mma_id_q][1] =
                rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];

        } /* end per-mma_id_q QK+softmax */

        /* Prefetch next K block (after QK is done, K_smem[kv_id%2] is free) */
        if (kv_id + 1 < max_kv_iter) {
            const uint32_t K_dst = K_smem +
                ((kv_id + 1) % 2) * (BLOCK_KV * DIM * sizeof(nv_bfloat16));
            global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(
                K_dst, K_ptr, DIM, tid);
            K_ptr += BLOCK_KV * DIM;
        }
        asm volatile("cp.async.commit_group;");

        /* Wait for V to arrive in SMEM */
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();

        /* O += P @ V  [BLOCK_Q, DIM]
         * Load V once per (mma_kv, mma_d) step, reuse across mma_id_q.
         * Loop order: mma_kv outer (P changes), mma_d middle (V changes),
         * mma_id_q inner (reuses both V_frag and P_rmem[mma_id_q]). */
        #pragma unroll
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
            #pragma unroll
            for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
                uint32_t V_frag[2];
                uint32_t addr = V_smem_thread;
                addr += mma_id_kv * MMA_K * DIM * sizeof(nv_bfloat16);
                addr ^= mma_id_d * MMA_N * sizeof(nv_bfloat16);
                ldmatrix_x2_trans(V_frag, addr);
                #pragma unroll
                for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
                    mma_m16n8k16(P_rmem[mma_id_q][mma_id_kv],
                                 V_frag,
                                 O_rmem[mma_id_q][mma_id_d]);
            }
    }

    /* ---- Write O to global memory (divide by softmax denominator) ---- */
    #pragma unroll
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        #pragma unroll
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
            const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

            float *regs = O_rmem[mma_id_q][mma_id_d];
            regs[0] /= rowsumexp[mma_id_q][0];
            regs[1] /= rowsumexp[mma_id_q][0];
            regs[2] /= rowsumexp[mma_id_q][1];
            regs[3] /= rowsumexp[mma_id_q][1];

            reinterpret_cast<nv_bfloat162 *>(O + (row + 0) * DIM + col)[0] =
                __float22bfloat162_rn({regs[0], regs[1]});
            reinterpret_cast<nv_bfloat162 *>(O + (row + 8) * DIM + col)[0] =
                __float22bfloat162_rn({regs[2], regs[3]});
        }
}

/* ======================================================================
 *  Simple integer parser helpers (no stdlib dep beyond getenv/atoi)
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

/* Tiny JSON integer extractor: find "key": <int> in a JSON string. */
static int json_int(const char *json, const char *key, int fallback) {
    if (!json) return fallback;
    const char *p = strstr(json, key);
    if (!p) return fallback;
    p += strlen(key);
    /* skip ": and whitespace */
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

/* ======================================================================
 *  kernel_run — entry point called by the cuda_exec harness
 *
 *   inputs[0] = Q [B, S, H, D]
 *   inputs[1] = K [B, S, H, D]
 *   inputs[2] = V [B, S, H, D]
 *   outputs[0] = O [B, S, H, D]
 *   n = total elements per buffer = B * S * H * D
 * ====================================================================== */

extern "C" int kernel_run(
    __nv_bfloat16 **inputs,  int num_inputs,
    __nv_bfloat16 **outputs, int num_outputs,
    int n, cudaStream_t stream)
{
    /* ---- Read dimensions from env ---- */
    const char *config_json = getenv("CUDA_EXEC_CONFIG_JSON");

    int B = parse_int_env("CUDA_EXEC_PARAM_BATCH_SIZE", 0);
    int S = parse_int_env("CUDA_EXEC_PARAM_SEQ_LEN",    0);
    int H = parse_int_env("CUDA_EXEC_PARAM_NUM_HEADS",  0);
    int D = parse_int_env("CUDA_EXEC_PARAM_HEAD_DIM",   0);

    /* Fallback: try CUDA_EXEC_CONFIG_JSON */
    if (B == 0) B = json_int(config_json, "batch_size", 0);
    if (S == 0) S = json_int(config_json, "seq_len",    0);
    if (H == 0) H = json_int(config_json, "num_heads",  0);
    if (D == 0) D = json_int(config_json, "head_dim",   0);

    /* Last resort: infer from n assuming D=128, H=16 */
    if (D == 0) D = 128;
    if (H == 0) H = 16;
    if (S == 0 && B == 0 && n > 0) {
        int total_tokens = n / (H * D);       /* B * S */
        /* Assume B=1 if we cannot determine */
        B = 1;
        S = total_tokens / B;
    }
    if (B == 0 || S == 0) return -1;

    /* Causal flag */
    bool causal = parse_bool_env("CUDA_EXEC_PARAM_CAUSAL", false);
    if (!causal && config_json) {
        causal = json_bool(config_json, "causal", false);
    }

    /* Sanity check */
    if (D != 128) return -2;    /* kernel only supports DIM=128 */

    /* ---- Allocate transposed buffers [B, H, S, D] ---- */
    size_t buf_bytes = (size_t)n * sizeof(__nv_bfloat16);
    __nv_bfloat16 *Qt = nullptr, *Kt = nullptr, *Vt = nullptr, *Ot = nullptr;
    cudaMalloc(&Qt, buf_bytes);
    cudaMalloc(&Kt, buf_bytes);
    cudaMalloc(&Vt, buf_bytes);
    cudaMalloc(&Ot, buf_bytes);

    /* ---- Transpose Q, K, V: [B, S, H, D] -> [B, H, S, D] ---- */
    {
        int tpb = 256;
        int blks = (n + tpb - 1) / tpb;
        transpose_bshd_bhsd<<<blks, tpb, 0, stream>>>(
            inputs[0], Qt, B, S, H, D, /*forward=*/1);
        transpose_bshd_bhsd<<<blks, tpb, 0, stream>>>(
            inputs[1], Kt, B, S, H, D, /*forward=*/1);
        transpose_bshd_bhsd<<<blks, tpb, 0, stream>>>(
            inputs[2], Vt, B, S, H, D, /*forward=*/1);
    }

    /* ---- Launch Flash Attention ---- */
    {
        const int BLOCK_Q   = 128;
        const int BLOCK_KV  = 64;
        const int DIM_CONST = 128;
        const int NUM_WARPS = 4;
        const int TB_SIZE   = NUM_WARPS * WARP_SIZE;

        int effective_bs = B * H;
        int num_blocks = effective_bs * cdiv(S, BLOCK_Q);

        /* SMEM budget: Q region (persistent) + KV region (rotated)
         *   Q:  BLOCK_Q * DIM * 2 = 128*128*2 = 32768
         *   KV: (2*BLOCK_KV + BLOCK_KV)*DIM*2 = 3*64*128*2 = 49152
         *   Total: 32768 + 49152 = 81920 (80KB, within 99KB limit)
         */
        int smem_q   = BLOCK_Q * DIM_CONST * (int)sizeof(nv_bfloat16);
        int smem_kv  = 3 * BLOCK_KV * DIM_CONST * (int)sizeof(nv_bfloat16);
        int smem_size = smem_q + smem_kv;

        auto kernel = flash_attention_kernel<BLOCK_Q, BLOCK_KV,
                                             DIM_CONST, NUM_WARPS>;
        if (smem_size > 48000)
            cudaFuncSetAttribute(kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        kernel<<<num_blocks, TB_SIZE, smem_size, stream>>>(
            Qt, Kt, Vt, Ot, effective_bs, S, S, causal ? 1 : 0);
    }

    /* ---- Transpose O back: [B, H, S, D] -> [B, S, H, D] ---- */
    {
        int tpb = 256;
        int blks = (n + tpb - 1) / tpb;
        transpose_bshd_bhsd<<<blks, tpb, 0, stream>>>(
            Ot, outputs[0], B, S, H, D, /*forward=*/0);
    }

    /* ---- Cleanup ---- */
    cudaStreamSynchronize(stream);
    cudaFree(Qt);
    cudaFree(Kt);
    cudaFree(Vt);
    cudaFree(Ot);

    return 0;
}
