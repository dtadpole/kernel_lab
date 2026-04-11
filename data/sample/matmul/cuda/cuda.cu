/*
 * Sample BF16 matmul kernel with tunable tile parameters.
 *
 * This is a simple shared-memory tiled matmul for testing autotune.
 * It is NOT optimized — it exists to validate the autotune pipeline
 * with different tile sizes across different matrix configs.
 *
 * Tunable parameters (via #ifndef / autotune.yaml):
 *   BM  — tile height (rows of A per block)
 *   BN  — tile width  (cols of B per block)
 *   BK  — reduction tile (shared K dimension per iteration)
 *
 * Interface: kernel_run() per eval_harness.cu contract.
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* --- Tunable tile parameters (overridden by autotune -D flags) --- */
#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 16
#endif

#define THREADS_X 16
#define THREADS_Y 16

/* ------------------------------------------------------------------ */
/* Tiled matmul kernel: C = A @ B (row-major, BF16)                   */
/* ------------------------------------------------------------------ */
__global__ void matmul_tiled(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    __shared__ __nv_bfloat16 As[BM][BK];
    __shared__ __nv_bfloat16 Bs[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BM + ty;
    int col = bx * BN + tx;

    float acc = 0.0f;

    /* Number of load iterations per thread for tiles larger than thread block */
    int load_iters_m = (BM + THREADS_Y - 1) / THREADS_Y;
    int load_iters_n = (BN + THREADS_X - 1) / THREADS_X;
    int load_iters_k_a = (BK + THREADS_X - 1) / THREADS_X;
    int load_iters_k_b = (BK + THREADS_Y - 1) / THREADS_Y;

    for (int k0 = 0; k0 < K; k0 += BK) {
        /* Load A tile: BM x BK */
        for (int li = 0; li < load_iters_m; li++) {
            for (int lj = 0; lj < load_iters_k_a; lj++) {
                int lr = ty + li * THREADS_Y;
                int lc = tx + lj * THREADS_X;
                if (lr < BM && lc < BK) {
                    int global_r = by * BM + lr;
                    int global_c = k0 + lc;
                    if (global_r < M && global_c < K)
                        As[lr][lc] = A[global_r * K + global_c];
                    else
                        As[lr][lc] = __float2bfloat16(0.0f);
                }
            }
        }
        /* Load B tile: BK x BN */
        for (int li = 0; li < load_iters_k_b; li++) {
            for (int lj = 0; lj < load_iters_n; lj++) {
                int lr = ty + li * THREADS_Y;
                int lc = tx + lj * THREADS_X;
                if (lr < BK && lc < BN) {
                    int global_r = k0 + lr;
                    int global_c = bx * BN + lc;
                    if (global_r < K && global_c < N)
                        Bs[lr][lc] = B[global_r * N + global_c];
                    else
                        Bs[lr][lc] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        /* Compute — each thread accumulates multiple output elements */
        for (int li = 0; li < load_iters_m; li++) {
            for (int lj = 0; lj < load_iters_n; lj++) {
                int local_r = ty + li * THREADS_Y;
                int local_c = tx + lj * THREADS_X;
                if (local_r < BM && local_c < BN) {
                    float sum = 0.0f;
                    for (int ki = 0; ki < BK && (k0 + ki) < K; ki++) {
                        sum += __bfloat162float(As[local_r][ki]) *
                               __bfloat162float(Bs[ki][local_c]);
                    }
                    int out_r = by * BM + local_r;
                    int out_c = bx * BN + local_c;
                    if (out_r < M && out_c < N) {
                        float prev = (k0 == 0) ? 0.0f : __bfloat162float(C[out_r * N + out_c]);
                        C[out_r * N + out_c] = __float2bfloat16(prev + sum);
                    }
                }
            }
        }
        __syncthreads();
    }
}

/* ------------------------------------------------------------------ */
/* eval_harness interface                                             */
/* ------------------------------------------------------------------ */
extern "C" int kernel_run(
    __nv_bfloat16** inputs,  int num_inputs,
    __nv_bfloat16** outputs, int num_outputs,
    int n, cudaStream_t stream)
{
    if (num_inputs < 2 || num_outputs < 1) return -1;

    /* Parse shape from env: CUDA_EXEC_PARAM_SHAPE = "[M, N]" */
    int M = 1024, N = 1024, K = 1024;
    const char* shape_env = getenv("CUDA_EXEC_PARAM_SHAPE");
    if (shape_env) {
        sscanf(shape_env, "[%d, %d]", &M, &N);
        K = M;  /* square matmul */
    }

    __nv_bfloat16* A = inputs[0];   /* M x K */
    __nv_bfloat16* B = inputs[1];   /* K x N */
    __nv_bfloat16* C = outputs[0];  /* M x N */

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(THREADS_X, THREADS_Y);

    matmul_tiled<<<grid, block, 0, stream>>>(A, B, C, M, N, K);

    return 0;
}
