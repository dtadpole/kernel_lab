/*
 * Flash Attention forward pass — BF16, SM120 tensor-core MMA kernel.
 * ALL 128 threads (4 warps) do both cp.async loading AND mma.sync
 * computation. No dedicated DMA warp.
 *
 * Layout:
 *   Q, K, V, O  : [B, S, H, D]  (row-major __nv_bfloat16)
 *
 * Constants: BLOCK_Q=128, BLOCK_KV=64, DIM=128, 4 warps (128 threads).
 *
 * Pipeline: double-buffered K/V with cp.async. Each iteration issues
 * loads for K[next]+V[next] BEFORE compute, so memory transfers overlap
 * with QK and PV GEMM.
 *
 * SMEM layout (double-buffered K/V):
 *   Q:  128 x 128 x 2 = 32,768 bytes (persistent)
 *   K:  2 x 64 x 128 x 2 = 32,768 bytes
 *   V:  2 x 64 x 128 x 2 = 32,768 bytes
 *   Total: 98,304 bytes (96KB)
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
 *  Helper constants & device functions
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

/* --- cp.async global -> shared (swizzled) — inlined version ---------- */

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

/* --- cp.async load + commit (noinline to isolate register scope) ----- */

template <int HEIGHT, int WIDTH, int TB_SIZE>
__noinline__ __device__
void load_tile_async(uint32_t smem_dst, const nv_bfloat16 *gmem_src,
                     int src_stride, int tid) {
    global_to_shared_swizzle<HEIGHT, WIDTH, TB_SIZE>(smem_dst, gmem_src, src_stride, tid);
    asm volatile("cp.async.commit_group;");
}

/* Load two KV tiles (K+V) with a single noinline call to minimize
 * function-call ABI overhead while isolating register scope. */
template <int HEIGHT, int WIDTH, int TB_SIZE>
__noinline__ __device__
void load_kv_pair_async(uint32_t k_smem_dst, const nv_bfloat16 *k_gmem_src,
                        uint32_t v_smem_dst, const nv_bfloat16 *v_gmem_src,
                        int src_stride, int tid) {
    global_to_shared_swizzle<HEIGHT, WIDTH, TB_SIZE>(k_smem_dst, k_gmem_src, src_stride, tid);
    global_to_shared_swizzle<HEIGHT, WIDTH, TB_SIZE>(v_smem_dst, v_gmem_src, src_stride, tid);
    asm volatile("cp.async.commit_group;");
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
 *  Flash Attention kernel — 128 threads, all compute+load
 * ====================================================================== */

template<int BLOCK_Q, int BLOCK_KV, int DIM>
__launch_bounds__(128, 1)
__global__
void flash_attention_kernel(
    const nv_bfloat16 *Q,   /* [B, S, H, D] */
    const nv_bfloat16 *K,   /* [B, S, H, D] */
    const nv_bfloat16 *V,   /* [B, S, H, D] */
    nv_bfloat16 *O,         /* [B, S, H, D] */
    int B, int S, int H,
    int len_q,
    int len_kv,
    int is_causal)
{
    constexpr int NUM_WARPS = 4;
    constexpr int TB_SIZE   = 128;
    constexpr int WARP_Q    = BLOCK_Q / NUM_WARPS;  /* 32 */
    constexpr int MMA_M     = 16;
    constexpr int MMA_N     = 8;
    constexpr int MMA_K     = 16;
    constexpr int KV_SLOT_BYTES = BLOCK_KV * DIM * (int)sizeof(nv_bfloat16);

    const int bid     = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int num_q_blocks = cdiv(len_q, BLOCK_Q);
    const int bs_id        = bid / num_q_blocks;
    const int q_block_id   = bid % num_q_blocks;

    const int batch_id = bs_id / H;
    const int head_id  = bs_id % H;

    const int seq_stride = H * DIM;

    const nv_bfloat16 *Q_base = Q + batch_id * S * seq_stride + head_id * DIM
                                   + q_block_id * BLOCK_Q * seq_stride;
    const nv_bfloat16 *K_base_ptr = K + batch_id * S * seq_stride + head_id * DIM;
    const nv_bfloat16 *V_base_ptr = V + batch_id * S * seq_stride + head_id * DIM;
    nv_bfloat16       *O_base = O + batch_id * S * seq_stride + head_id * DIM
                                  + q_block_id * BLOCK_Q * seq_stride;

    extern __shared__ nv_bfloat16 smem[];
    const uint32_t smem_base = __cvta_generic_to_shared(smem);
    const uint32_t Q_smem  = smem_base;
    const uint32_t KV_base = smem_base + BLOCK_Q * DIM * sizeof(nv_bfloat16);
    const uint32_t K_smem  = KV_base;
    const uint32_t V_smem  = KV_base + 2 * KV_SLOT_BYTES;

    /* ================================================================
     *  Prologue: Load Q, K[0], V[0]
     * ================================================================ */

    load_tile_async<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q_base, seq_stride, tid);
    load_tile_async<BLOCK_KV, DIM, TB_SIZE>(K_smem, K_base_ptr, seq_stride, tid);
    load_tile_async<BLOCK_KV, DIM, TB_SIZE>(V_smem, V_base_ptr, seq_stride, tid);

    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    /* ================================================================
     *  Pre-compute SMEM addresses for ldmatrix
     * ================================================================ */

    uint32_t Q_smem_thread;
    {
        const int row_off = warp_id * WARP_Q + (lane_id % 16);
        const int col_off = lane_id / 16 * 8;
        Q_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(
            Q_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }

    uint32_t K_smem_thread;
    {
        const int row_off = lane_id % 8;
        const int col_off = lane_id / 8 * 8;
        K_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(
            K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }

    uint32_t V_smem_thread;
    {
        const int row_off = lane_id % 16;
        const int col_off = lane_id / 16 * 8;
        V_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(
            V_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }

    /* ================================================================
     *  Initialize accumulators
     * ================================================================ */

    const float softmax_scale = rsqrtf(static_cast<float>(DIM));
    const float softmax_scale_log2 = softmax_scale * 1.4426950408889634f;

    float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};

    float rowmax[WARP_Q / MMA_M][2];
    float rowsumexp[WARP_Q / MMA_M][2] = {};
    #pragma unroll
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
        rowmax[mma_id_q][0] = -FLT_MAX;
        rowmax[mma_id_q][1] = -FLT_MAX;
    }

    const int num_kv_iter = cdiv(len_kv, BLOCK_KV);
    const int max_kv_iter = is_causal
        ? min(num_kv_iter, cdiv((q_block_id + 1) * BLOCK_Q, BLOCK_KV))
        : num_kv_iter;

    /* ================================================================
     *  Main loop
     * ================================================================ */

    for (int kv_id = 0; kv_id < max_kv_iter; kv_id++) {

        const uint32_t K_cur = K_smem_thread + (kv_id % 2) * KV_SLOT_BYTES;
        const uint32_t V_cur_base = V_smem_thread + (kv_id % 2) * KV_SLOT_BYTES;

        /* ---- Issue K[next]+V[next] loads (overlap with QK+PV) ---- */
        if (kv_id + 1 < max_kv_iter) {
            load_kv_pair_async<BLOCK_KV, DIM, TB_SIZE>(
                K_smem + ((kv_id + 1) % 2) * KV_SLOT_BYTES,
                K_base_ptr + (kv_id + 1) * BLOCK_KV * seq_stride,
                V_smem + ((kv_id + 1) % 2) * KV_SLOT_BYTES,
                V_base_ptr + (kv_id + 1) * BLOCK_KV * seq_stride,
                seq_stride, tid);
        }

        /* ---- QK GEMM — S = Q @ K^T ---- */

        uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];

        #pragma unroll
        for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
            float S_local[BLOCK_KV / MMA_N][4] = {};

            uint32_t Q_frag[4];
            {
                uint32_t qaddr = Q_smem_thread
                    + mma_id_q * MMA_M * DIM * sizeof(nv_bfloat16);
                ldmatrix_x4(Q_frag, qaddr);
            }

            #pragma unroll
            for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
                uint32_t Q_next[4];
                if (mma_id_d + 1 < DIM / MMA_K) {
                    uint32_t qaddr = Q_smem_thread
                        + mma_id_q * MMA_M * DIM * sizeof(nv_bfloat16);
                    qaddr ^= (mma_id_d + 1) * MMA_K * sizeof(nv_bfloat16);
                    ldmatrix_x4(Q_next, qaddr);
                }

                #pragma unroll
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                    uint32_t K_frag[2];
                    {
                        uint32_t kaddr = K_cur;
                        kaddr += mma_id_kv * MMA_N * DIM * sizeof(nv_bfloat16);
                        kaddr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
                        ldmatrix_x2(K_frag, kaddr);
                    }
                    mma_m16n8k16(Q_frag, K_frag, S_local[mma_id_kv]);
                }

                if (mma_id_d + 1 < DIM / MMA_K) {
                    Q_frag[0] = Q_next[0]; Q_frag[1] = Q_next[1];
                    Q_frag[2] = Q_next[2]; Q_frag[3] = Q_next[3];
                }
            }

            /* ---- Softmax scale ---- */
            #pragma unroll
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
                #pragma unroll
                for (int reg_id = 0; reg_id < 4; reg_id++)
                    S_local[mma_id_kv][reg_id] *= softmax_scale_log2;

            /* ---- Causal mask ---- */
            if (is_causal) {
                const int q_start  = q_block_id * BLOCK_Q;
                const int kv_start = kv_id * BLOCK_KV;

                #pragma unroll
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                    int q_pos_0  = q_start + warp_id * WARP_Q
                                   + mma_id_q * MMA_M + (lane_id / 4);
                    int q_pos_1  = q_pos_0 + 8;
                    int kv_pos_0 = kv_start
                                   + mma_id_kv * MMA_N + (lane_id % 4) * 2;
                    int kv_pos_1 = kv_pos_0 + 1;

                    if (kv_pos_0 > q_pos_0) S_local[mma_id_kv][0] = -FLT_MAX;
                    if (kv_pos_1 > q_pos_0) S_local[mma_id_kv][1] = -FLT_MAX;
                    if (kv_pos_0 > q_pos_1) S_local[mma_id_kv][2] = -FLT_MAX;
                    if (kv_pos_1 > q_pos_1) S_local[mma_id_kv][3] = -FLT_MAX;
                }
            }

            /* ---- Online softmax + rescaling ---- */
            float this_rowmax[2];
            #pragma unroll
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

            float rescale[2];
            rescale[0] = exp2f(rowmax[mma_id_q][0] - this_rowmax[0]);
            rescale[1] = exp2f(rowmax[mma_id_q][1] - this_rowmax[1]);
            #pragma unroll
            for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
                O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
                O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
                O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
                O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
            }

            rowmax[mma_id_q][0] = this_rowmax[0];
            rowmax[mma_id_q][1] = this_rowmax[1];

            float this_rowsumexp[2];
            #pragma unroll
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                float *regs = S_local[mma_id_kv];
                regs[0] = exp2f(regs[0] - rowmax[mma_id_q][0]);
                regs[1] = exp2f(regs[1] - rowmax[mma_id_q][0]);
                regs[2] = exp2f(regs[2] - rowmax[mma_id_q][1]);
                regs[3] = exp2f(regs[3] - rowmax[mma_id_q][1]);

                if (mma_id_kv == 0) {
                    this_rowsumexp[0] = regs[0] + regs[1];
                    this_rowsumexp[1] = regs[2] + regs[3];
                } else {
                    this_rowsumexp[0] += regs[0] + regs[1];
                    this_rowsumexp[1] += regs[2] + regs[3];
                }

                nv_bfloat162 *this_P =
                    reinterpret_cast<nv_bfloat162 *>(
                        P_rmem[mma_id_q][mma_id_kv / 2]);
                this_P[(mma_id_kv % 2) * 2] =
                    __float22bfloat162_rn({regs[0], regs[1]});
                this_P[(mma_id_kv % 2) * 2 + 1] =
                    __float22bfloat162_rn({regs[2], regs[3]});
            }

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

        /* ---- PV GEMM — O += P @ V ---- */
        {
            uint32_t V_frag[2];
            ldmatrix_x2_trans(V_frag, V_cur_base);

            #pragma unroll
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++) {
                #pragma unroll
                for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
                    uint32_t V_next[2];
                    const bool has_next_d  = (mma_id_d + 1 < DIM / MMA_N);
                    const bool has_next_kv = (mma_id_kv + 1 < BLOCK_KV / MMA_K);
                    if (has_next_d) {
                        uint32_t addr = V_cur_base;
                        addr += mma_id_kv * MMA_K * DIM * sizeof(nv_bfloat16);
                        addr ^= (mma_id_d + 1) * MMA_N * sizeof(nv_bfloat16);
                        ldmatrix_x2_trans(V_next, addr);
                    } else if (has_next_kv) {
                        uint32_t addr = V_cur_base;
                        addr += (mma_id_kv + 1) * MMA_K * DIM * sizeof(nv_bfloat16);
                        ldmatrix_x2_trans(V_next, addr);
                    }

                    #pragma unroll
                    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
                        mma_m16n8k16(P_rmem[mma_id_q][mma_id_kv],
                                     V_frag,
                                     O_rmem[mma_id_q][mma_id_d]);

                    if (has_next_d || has_next_kv) {
                        V_frag[0] = V_next[0]; V_frag[1] = V_next[1];
                    }
                }
            }
        }

        /* ---- Wait for next K/V loads ---- */
        if (kv_id + 1 < max_kv_iter) {
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
        }

    } /* end kv_id loop */

    /* ================================================================
     *  Epilogue
     * ================================================================ */

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

            reinterpret_cast<nv_bfloat162 *>(O_base + (row + 0) * seq_stride + col)[0] =
                __float22bfloat162_rn({regs[0], regs[1]});
            reinterpret_cast<nv_bfloat162 *>(O_base + (row + 8) * seq_stride + col)[0] =
                __float22bfloat162_rn({regs[2], regs[3]});
        }
}

/* ======================================================================
 *  Simple integer parser helpers
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

/* ======================================================================
 *  kernel_run -- entry point
 * ====================================================================== */

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
    if (!causal && config_json) {
        causal = json_bool(config_json, "causal", false);
    }

    if (D != 128) return -2;

    {
        const int BLOCK_Q   = 128;
        const int BLOCK_KV  = 64;
        const int DIM_CONST = 128;
        const int TB_SIZE   = 128;

        int effective_bs = B * H;
        int num_blocks = effective_bs * cdiv(S, BLOCK_Q);

        int smem_q  = BLOCK_Q * DIM_CONST * (int)sizeof(nv_bfloat16);
        int smem_kv = 4 * BLOCK_KV * DIM_CONST * (int)sizeof(nv_bfloat16);
        int smem_size = smem_q + smem_kv;

        auto kernel = flash_attention_kernel<BLOCK_Q, BLOCK_KV, DIM_CONST>;
        if (smem_size > 48000)
            cudaFuncSetAttribute(kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        kernel<<<num_blocks, TB_SIZE, smem_size, stream>>>(
            inputs[0], inputs[1], inputs[2], outputs[0],
            B, S, H, S, S, causal ? 1 : 0);
    }

    return 0;
}
