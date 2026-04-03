/*
 * Flash Attention forward pass — BF16, SM90 Hopper tensor-core MMA kernel.
 * Unified architecture: 128 threads (4 warps), cooperative K/V loading,
 * Q cached in registers, single-buffered K/V with SMEM reuse.
 *
 * Target: NVIDIA H100 (SM90, GH100)
 *
 * Architecture vs baseline (warp-specialized, 160 threads):
 *   - Eliminates DMA warp and named barriers
 *   - 128 threads → more registers/thread (256 budget vs 204)
 *   - Q fragments cached in registers (saves 8 LDSM/iter)
 *   - K/V share same SMEM slot (loaded one at a time)
 *   - Simpler code → better compiler optimization
 *
 * Constants: BLOCK_Q=64, BLOCK_KV=64, DIM=128, 4 warps (128 threads).
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

static constexpr int WARP_SIZE = 32;

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
    constexpr int num_elems = 16 / sizeof(nv_bfloat16);
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
 *  Unified kernel — all 4 warps do both loading and compute
 * ====================================================================== */

template<int BLOCK_Q, int BLOCK_KV, int DIM>
__launch_bounds__(128, 2)
__global__
void flash_attention_kernel_unified(
    const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    nv_bfloat16 *O,
    int B, int S, int H,
    int len_q, int len_kv,
    int is_causal)
{
    constexpr int NUM_WARPS    = 4;
    constexpr int TB_SIZE      = NUM_WARPS * WARP_SIZE;  /* 128 */
    constexpr int WARP_Q       = BLOCK_Q / NUM_WARPS;    /* 16 */
    constexpr int MMA_M        = 16;
    constexpr int MMA_N        = 8;
    constexpr int MMA_K        = 16;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int num_q_blocks = cdiv(len_q, BLOCK_Q);
    const int bs_id       = bid / num_q_blocks;
    const int q_block_id  = bid % num_q_blocks;
    const int batch_id    = bs_id / H;
    const int head_id     = bs_id % H;
    const int seq_stride  = H * DIM;

    const nv_bfloat16 *Q_base = Q + batch_id * S * seq_stride + head_id * DIM
                                   + q_block_id * BLOCK_Q * seq_stride;
    const nv_bfloat16 *K_base = K + batch_id * S * seq_stride + head_id * DIM;
    const nv_bfloat16 *V_base = V + batch_id * S * seq_stride + head_id * DIM;
    nv_bfloat16       *O_base = O + batch_id * S * seq_stride + head_id * DIM
                                  + q_block_id * BLOCK_Q * seq_stride;

    /* SMEM layout: Q (persistent) + KV (shared slot for K then V)
     * Q:  BLOCK_Q * DIM * 2 = 16KB
     * KV: BLOCK_KV * DIM * 2 = 16KB
     * Total: 32KB */
    extern __shared__ nv_bfloat16 smem[];
    const uint32_t smem_base = __cvta_generic_to_shared(smem);
    const uint32_t Q_smem  = smem_base;
    const uint32_t KV_smem = smem_base + BLOCK_Q * DIM * sizeof(nv_bfloat16);

    /* ---- Load Q to shared memory (all threads cooperate) ---- */
    global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE>(
        Q_smem, Q_base, seq_stride, tid);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    /* ---- Cache Q fragments in registers (8 slices of 16-column width) ---- */
    const int q_row_off = warp_id * WARP_Q + (lane_id % 16);
    const int q_col_off = lane_id / 16 * 8;
    uint32_t Q_smem_base = swizzle<DIM * sizeof(nv_bfloat16)>(
        Q_smem + (q_row_off * DIM + q_col_off) * sizeof(nv_bfloat16));

    uint32_t Q_regs[DIM / MMA_K][4];
    #pragma unroll
    for (int md = 0; md < DIM / MMA_K; md++) {
        uint32_t qaddr = Q_smem_base;
        qaddr ^= md * MMA_K * sizeof(nv_bfloat16);
        ldmatrix_x4(Q_regs[md], qaddr);
    }

    /* ---- K/V smem thread base addresses ---- */
    uint32_t K_smem_thread;
    {
        const int row_off = lane_id % 8;
        const int col_off = lane_id / 8 * 8;
        K_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(
            KV_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
    uint32_t V_smem_thread;
    {
        const int row_off = lane_id % 16;
        const int col_off = lane_id / 16 * 8;
        V_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(
            KV_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }

    /* ---- Accumulators ---- */
    const float softmax_scale_log2 =
        rsqrtf(static_cast<float>(DIM)) * 1.4426950408889634f;

    float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};
    float rowmax[WARP_Q / MMA_M][2];
    float rowsumexp[WARP_Q / MMA_M][2] = {};
    for (int mq = 0; mq < WARP_Q / MMA_M; mq++) {
        rowmax[mq][0] = -FLT_MAX;
        rowmax[mq][1] = -FLT_MAX;
    }

    /* ---- KV iteration loop ---- */
    const int num_kv_iter = cdiv(len_kv, BLOCK_KV);
    const int max_kv_iter = is_causal
        ? min(num_kv_iter, cdiv((q_block_id + 1) * BLOCK_Q, BLOCK_KV))
        : num_kv_iter;

    const nv_bfloat16 *K_ptr = K_base;
    const nv_bfloat16 *V_ptr = V_base;

    for (int kv_id = 0; kv_id < max_kv_iter; kv_id++) {

        /* ==== Load K tile to shared memory (all 128 threads) ==== */
        global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(
            KV_smem, K_ptr, seq_stride, tid);
        K_ptr += BLOCK_KV * seq_stride;
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncthreads();

        /* ==== QK matmul: Q_regs × K_smem → S_local ==== */
        uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];

        for (int mq = 0; mq < WARP_Q / MMA_M; mq++) {
            float S_local[BLOCK_KV / MMA_N][4] = {};

            #pragma unroll
            for (int md = 0; md < DIM / MMA_K; md++) {
                #pragma unroll
                for (int mk = 0; mk < BLOCK_KV / MMA_N; mk++) {
                    uint32_t K_frag[2];
                    uint32_t kaddr = K_smem_thread +
                        mk * MMA_N * DIM * sizeof(nv_bfloat16);
                    kaddr ^= md * MMA_K * sizeof(nv_bfloat16);
                    ldmatrix_x2(K_frag, kaddr);
                    mma_m16n8k16(Q_regs[md], K_frag, S_local[mk]);
                }
            }

            /* ---- Scale ---- */
            #pragma unroll
            for (int i = 0; i < BLOCK_KV / MMA_N; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    S_local[i][j] *= softmax_scale_log2;

            /* ---- Causal mask ---- */
            if (is_causal) {
                const int q_first = q_block_id * BLOCK_Q +
                                    warp_id * WARP_Q + mq * MMA_M;
                const int kv_start = kv_id * BLOCK_KV;
                const int kv_last = kv_start + BLOCK_KV - 1;

                if (kv_last > q_first) {
                    for (int i = 0; i < BLOCK_KV / MMA_N; i++) {
                        int q0 = q_first + (lane_id / 4);
                        int q1 = q0 + 8;
                        int kv0 = kv_start + i * MMA_N + (lane_id % 4) * 2;
                        int kv1 = kv0 + 1;
                        if (kv0 > q0) S_local[i][0] = -FLT_MAX;
                        if (kv1 > q0) S_local[i][1] = -FLT_MAX;
                        if (kv0 > q1) S_local[i][2] = -FLT_MAX;
                        if (kv1 > q1) S_local[i][3] = -FLT_MAX;
                    }
                }
            }

            /* ---- Online softmax with fast_exp2f ---- */
            float this_rowmax[2];
            for (int i = 0; i < BLOCK_KV / MMA_N; i++) {
                float *r = S_local[i];
                if (i == 0) {
                    this_rowmax[0] = max(r[0], r[1]);
                    this_rowmax[1] = max(r[2], r[3]);
                } else {
                    this_rowmax[0] = max(this_rowmax[0], max(r[0], r[1]));
                    this_rowmax[1] = max(this_rowmax[1], max(r[2], r[3]));
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

            this_rowmax[0] = max(this_rowmax[0], rowmax[mq][0]);
            this_rowmax[1] = max(this_rowmax[1], rowmax[mq][1]);

            float rescale[2];
            rescale[0] = fast_exp2f(rowmax[mq][0] - this_rowmax[0]);
            rescale[1] = fast_exp2f(rowmax[mq][1] - this_rowmax[1]);
            #pragma unroll
            for (int d = 0; d < DIM / MMA_N; d++) {
                O_rmem[mq][d][0] *= rescale[0];
                O_rmem[mq][d][1] *= rescale[0];
                O_rmem[mq][d][2] *= rescale[1];
                O_rmem[mq][d][3] *= rescale[1];
            }

            rowmax[mq][0] = this_rowmax[0];
            rowmax[mq][1] = this_rowmax[1];

            float this_rowsumexp[2];
            #pragma unroll
            for (int i = 0; i < BLOCK_KV / MMA_N; i++) {
                float *r = S_local[i];
                r[0] = fast_exp2f(r[0] - rowmax[mq][0]);
                r[1] = fast_exp2f(r[1] - rowmax[mq][0]);
                r[2] = fast_exp2f(r[2] - rowmax[mq][1]);
                r[3] = fast_exp2f(r[3] - rowmax[mq][1]);

                if (i == 0) {
                    this_rowsumexp[0] = r[0] + r[1];
                    this_rowsumexp[1] = r[2] + r[3];
                } else {
                    this_rowsumexp[0] += r[0] + r[1];
                    this_rowsumexp[1] += r[2] + r[3];
                }

                nv_bfloat162 *p = reinterpret_cast<nv_bfloat162 *>(
                    P_rmem[mq][i / 2]);
                p[(i % 2) * 2]     = __float22bfloat162_rn({r[0], r[1]});
                p[(i % 2) * 2 + 1] = __float22bfloat162_rn({r[2], r[3]});
            }

            this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 1);
            this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 2);
            this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 1);
            this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 2);

            rowsumexp[mq][0] = rowsumexp[mq][0] * rescale[0] + this_rowsumexp[0];
            rowsumexp[mq][1] = rowsumexp[mq][1] * rescale[1] + this_rowsumexp[1];

        } /* end mq loop */

        /* ==== Sync before reusing KV_smem for V ==== */
        __syncthreads();

        /* ==== Load V tile to shared memory (reuses KV_smem) ==== */
        global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(
            KV_smem, V_ptr, seq_stride, tid);
        V_ptr += BLOCK_KV * seq_stride;
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncthreads();

        /* ==== O += P @ V ==== */
        {
            uint32_t V_cur[2];
            ldmatrix_x2_trans(V_cur, V_smem_thread);

            #pragma unroll
            for (int kv = 0; kv < BLOCK_KV / MMA_K; kv++) {
                #pragma unroll
                for (int d = 0; d < DIM / MMA_N; d++) {
                    uint32_t V_next[2];
                    const bool has_next_d = (d + 1 < DIM / MMA_N);
                    const bool has_next_kv = (kv + 1 < BLOCK_KV / MMA_K);
                    if (has_next_d) {
                        uint32_t addr = V_smem_thread +
                            kv * MMA_K * DIM * sizeof(nv_bfloat16);
                        addr ^= (d + 1) * MMA_N * sizeof(nv_bfloat16);
                        ldmatrix_x2_trans(V_next, addr);
                    } else if (has_next_kv) {
                        uint32_t addr = V_smem_thread +
                            (kv + 1) * MMA_K * DIM * sizeof(nv_bfloat16);
                        ldmatrix_x2_trans(V_next, addr);
                    }
                    #pragma unroll
                    for (int mq = 0; mq < WARP_Q / MMA_M; mq++)
                        mma_m16n8k16(P_rmem[mq][kv], V_cur, O_rmem[mq][d]);
                    if (has_next_d || has_next_kv) {
                        V_cur[0] = V_next[0]; V_cur[1] = V_next[1];
                    }
                }
            }
        }

        /* Sync before next KV iteration loads new K */
        __syncthreads();

    } /* end kv_id loop */

    /* ---- Write O — use fast_rcp instead of division ---- */
    #pragma unroll
    for (int mq = 0; mq < WARP_Q / MMA_M; mq++) {
        float inv_sum[2];
        inv_sum[0] = fast_rcp(rowsumexp[mq][0]);
        inv_sum[1] = fast_rcp(rowsumexp[mq][1]);

        #pragma unroll
        for (int d = 0; d < DIM / MMA_N; d++) {
            const int row = warp_id * WARP_Q + mq * MMA_M + (lane_id / 4);
            const int col = d * MMA_N + (lane_id % 4) * 2;

            float *regs = O_rmem[mq][d];
            regs[0] *= inv_sum[0];
            regs[1] *= inv_sum[0];
            regs[2] *= inv_sum[1];
            regs[3] *= inv_sum[1];

            reinterpret_cast<nv_bfloat162 *>(O_base + (row + 0) * seq_stride + col)[0] =
                __float22bfloat162_rn({regs[0], regs[1]});
            reinterpret_cast<nv_bfloat162 *>(O_base + (row + 8) * seq_stride + col)[0] =
                __float22bfloat162_rn({regs[2], regs[3]});
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

    {
        const int BLOCK_Q   = 64;
        const int BLOCK_KV  = 64;
        const int DIM_CONST = 128;
        const int TB_SIZE   = 128;

        int num_blocks = B * H * cdiv(S, BLOCK_Q);

        /* SMEM: Q(16KB) + KV(16KB) = 32KB */
        int smem_size = BLOCK_Q * DIM_CONST * (int)sizeof(nv_bfloat16)
                      + BLOCK_KV * DIM_CONST * (int)sizeof(nv_bfloat16);

        auto kernel = flash_attention_kernel_unified<BLOCK_Q, BLOCK_KV, DIM_CONST>;
        if (smem_size > 48000)
            cudaFuncSetAttribute(kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        kernel<<<num_blocks, TB_SIZE, smem_size, stream>>>(
            inputs[0], inputs[1], inputs[2], outputs[0],
            B, S, H, S, S, causal ? 1 : 0);
    }

    return 0;
}
