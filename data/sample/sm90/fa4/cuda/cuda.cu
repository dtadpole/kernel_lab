/*
 * Simple Flash Attention forward — pure CUDA C, zero PTX.
 *
 * Standard tiled attention with online softmax.
 * No TMA, no WGMMA, no warp specialization, no inline PTX.
 *
 * Each thread block processes one Q-block for one (batch, head).
 * Each thread owns one Q row and iterates over KV blocks.
 *
 * Input:  Q, K, V — (B*S, H*D) row-major BF16
 * Output: O       — (B*S, H*D) row-major BF16
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>

#define BLOCK_Q  32        /* Q rows per block */
#define BLOCK_KV 32        /* KV rows per block */
#define DIM      128       /* head dimension */

/* One thread per Q row → THREADS = BLOCK_Q */
#define THREADS  BLOCK_Q

/* ── Kernel ──────────────────────────────────────────────────────── */

__global__ void flash_attn_fwd(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    int B, int S, int H,
    float scale, bool causal)
{
    /* Shared memory for K and V tiles */
    __shared__ float smem_K[BLOCK_KV][DIM];
    __shared__ float smem_V[BLOCK_KV][DIM];

    const int tid = threadIdx.x;   /* = my Q row within block */
    const int bh = blockIdx.x;     /* batch*head index */
    const int q_block = blockIdx.y;
    const int b = bh / H, h = bh % H;

    const int q_row_global = b * S + q_block * BLOCK_Q + tid;
    const int stride = H * DIM;
    const int head_off = h * DIM;

    /* Load my Q row into registers */
    float q[DIM];
    if (q_block * BLOCK_Q + tid < S) {
        for (int d = 0; d < DIM; d++)
            q[d] = __bfloat162float(Q[q_row_global * stride + head_off + d]);
    } else {
        for (int d = 0; d < DIM; d++) q[d] = 0.f;
    }

    /* Online softmax state */
    float row_max = -FLT_MAX;
    float row_sum = 0.f;
    float acc[DIM];
    for (int d = 0; d < DIM; d++) acc[d] = 0.f;

    /* Number of KV blocks to process */
    int kv_blocks = (S + BLOCK_KV - 1) / BLOCK_KV;
    if (causal) {
        int max_kv = q_block * BLOCK_Q + BLOCK_Q;  /* causal: can't look beyond my last Q row */
        kv_blocks = min(kv_blocks, (max_kv + BLOCK_KV - 1) / BLOCK_KV);
    }

    for (int kv = 0; kv < kv_blocks; kv++) {
        int kv_start = kv * BLOCK_KV;

        /* Cooperative load K and V tiles into SMEM (all threads help) */
        /* Each thread loads BLOCK_KV * DIM / THREADS elements */
        for (int i = tid; i < BLOCK_KV * DIM; i += THREADS) {
            int r = i / DIM, c = i % DIM;
            int kv_row = b * S + kv_start + r;
            float val = 0.f;
            if (kv_start + r < S)
                val = __bfloat162float(K[kv_row * stride + head_off + c]);
            smem_K[r][c] = val;
        }
        for (int i = tid; i < BLOCK_KV * DIM; i += THREADS) {
            int r = i / DIM, c = i % DIM;
            int kv_row = b * S + kv_start + r;
            float val = 0.f;
            if (kv_start + r < S)
                val = __bfloat162float(V[kv_row * stride + head_off + c]);
            smem_V[r][c] = val;
        }
        __syncthreads();

        /* Compute QK scores for my row */
        float scores[BLOCK_KV];
        float block_max = -FLT_MAX;

        for (int j = 0; j < BLOCK_KV; j++) {
            float dot = 0.f;
            for (int d = 0; d < DIM; d++)
                dot += q[d] * smem_K[j][d];
            dot *= scale;

            /* Causal mask */
            if (causal && (q_block * BLOCK_Q + tid) < (kv_start + j))
                dot = -FLT_MAX;

            /* Boundary mask */
            if (kv_start + j >= S)
                dot = -FLT_MAX;

            scores[j] = dot;
            block_max = fmaxf(block_max, dot);
        }

        /* Online softmax update */
        float new_max = fmaxf(row_max, block_max);
        float rescale = expf(row_max - new_max);

        /* Rescale old accumulator */
        row_sum *= rescale;
        for (int d = 0; d < DIM; d++)
            acc[d] *= rescale;

        /* Exp scores and accumulate PV */
        float block_sum = 0.f;
        for (int j = 0; j < BLOCK_KV; j++) {
            float p = expf(scores[j] - new_max);
            block_sum += p;
            for (int d = 0; d < DIM; d++)
                acc[d] += p * smem_V[j][d];
        }

        row_max = new_max;
        row_sum += block_sum;

        __syncthreads();
    }

    /* Normalize and store */
    if (q_block * BLOCK_Q + tid < S) {
        float inv = (row_sum > 0.f) ? (1.f / row_sum) : 0.f;
        for (int d = 0; d < DIM; d++)
            O[q_row_global * stride + head_off + d] =
                __float2bfloat16(acc[d] * inv);
    }
}

/* ── Host ─────────────────────────────────────────────────────────── */

static int json_int(const char *j, const char *k, int d) {
    if (!j) return d;
    char p[64]; snprintf(p, sizeof(p), "\"%s\":", k);
    const char *f = strstr(j, p);
    return f ? atoi(f + strlen(p)) : d;
}
static bool json_bool(const char *j, const char *k, bool d) {
    if (!j) return d;
    char p[64]; snprintf(p, sizeof(p), "\"%s\":", k);
    const char *f = strstr(j, p);
    if (!f) return d;
    f += strlen(p); while (*f == ' ') f++;
    return *f == 't' || *f == '1';
}
static int env_int(const char *n, int d) { const char *v = getenv(n); return v ? atoi(v) : d; }
static bool env_bool(const char *n, bool d) { const char *v = getenv(n); return v ? (*v=='1'||*v=='t') : d; }

extern "C" int kernel_run(
    __nv_bfloat16 **inputs, int num_inputs,
    __nv_bfloat16 **outputs, int num_outputs,
    int n, cudaStream_t stream)
{
    const char *cj = getenv("CUDA_EXEC_CONFIG_JSON");
    int B = env_int("CUDA_EXEC_PARAM_BATCH_SIZE", 0);
    int S = env_int("CUDA_EXEC_PARAM_SEQ_LEN", 0);
    int H = env_int("CUDA_EXEC_PARAM_NUM_HEADS", 0);
    int D = env_int("CUDA_EXEC_PARAM_HEAD_DIM", 0);
    if (!B) B = json_int(cj, "batch_size", 0);
    if (!S) S = json_int(cj, "seq_len", 0);
    if (!H) H = json_int(cj, "num_heads", 0);
    if (!D) D = json_int(cj, "head_dim", 0);
    if (!D) D = 128; if (!H) H = 16;
    if (!S && !B && n > 0) { B = 1; S = n / (H * D); }
    if (!B || !S) return -1;
    if (D != 128) return -2;

    bool causal = env_bool("CUDA_EXEC_PARAM_CAUSAL", false);
    if (!causal && cj) causal = json_bool(cj, "causal", false);

    float scale = 1.0f / sqrtf((float)D);
    dim3 grid(B * H, (S + BLOCK_Q - 1) / BLOCK_Q);
    flash_attn_fwd<<<grid, THREADS, 0, stream>>>(
        inputs[0], inputs[1], inputs[2], outputs[0],
        B, S, H, scale, causal);
    return 0;
}
