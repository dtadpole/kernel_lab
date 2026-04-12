/*
 * WMMA matmul with register tiling — pure CUDA C, zero PTX.
 *
 * Each warp computes WARP_M × WARP_N grid of 16×16 tiles.
 * Larger BK (64) amortizes SMEM loads over multiple wmma K-steps.
 *
 * Tunable: BM, BN (CTA tile), BK (K-tile, multiple of 16).
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;

#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef BK
#define BK 64
#endif

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define K_STEPS (BK / WMMA_K)    /* wmma K-steps per SMEM tile */

/* Register tiling: each warp computes WARP_M × WARP_N output tiles (16×16 each).
 * Fewer warps but more work per warp = higher Tensor Core utilization. */
#define WARP_M 4                 /* 4 × 16 = 64 rows per warp */
#define WARP_N 4                 /* 4 × 16 = 64 cols per warp */
#define WARPS_M (BM / (WARP_M * WMMA_M))
#define WARPS_N (BN / (WARP_N * WMMA_N))
#define NUM_WARPS (WARPS_M * WARPS_N)
#define THREADS  (NUM_WARPS * 32)

__global__ void matmul_wmma(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    /* Double-buffered SMEM */
    __shared__ __nv_bfloat16 smem_A[2][BM][BK];
    __shared__ __nv_bfloat16 smem_B[2][BK][BN];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;
    const int ctaRow = blockIdx.y * BM;
    const int ctaCol = blockIdx.x * BN;
    const int numK = (K + BK - 1) / BK;

    /* Accumulators: WARP_M × WARP_N grid of 16×16 tiles in registers */
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WARP_M][WARP_N];
    for (int wm = 0; wm < WARP_M; wm++)
        for (int wn = 0; wn < WARP_N; wn++)
            wmma::fill_fragment(acc[wm][wn], 0.0f);

    /* Load first tile */
    auto load_tile = [&](int buf, int kt) {
        for (int i = tid; i < BM * BK; i += THREADS) {
            int r = i / BK, c = i % BK;
            int gr = ctaRow + r, gc = kt * BK + c;
            smem_A[buf][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __nv_bfloat16(0);
        }
        for (int i = tid; i < BK * BN; i += THREADS) {
            int r = i / BN, c = i % BN;
            int gr = kt * BK + r, gc = ctaCol + c;
            smem_B[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : __nv_bfloat16(0);
        }
    };

    load_tile(0, 0);
    __syncthreads();

    /* Main K-loop with double buffering */
    for (int kt = 0; kt < numK; kt++) {
        int cur = kt % 2;

        /* Start loading next tile (if exists) into the other buffer */
        if (kt + 1 < numK) {
            load_tile(1 - cur, kt + 1);
        }

        /* WMMA compute: WARP_M × WARP_N tiles × K_STEPS k-iterations */
        int base_row = warp_row * WARP_M * WMMA_M;
        int base_col = warp_col * WARP_N * WMMA_N;

        for (int ks = 0; ks < K_STEPS; ks++) {
            /* Load A and B fragments, compute */
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag[WARP_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag[WARP_N];

            for (int wm = 0; wm < WARP_M; wm++)
                wmma::load_matrix_sync(a_frag[wm],
                    &smem_A[cur][base_row + wm * WMMA_M][ks * WMMA_K], BK);

            for (int wn = 0; wn < WARP_N; wn++)
                wmma::load_matrix_sync(b_frag[wn],
                    &smem_B[cur][ks * WMMA_K][base_col + wn * WMMA_N], BN);

            for (int wm = 0; wm < WARP_M; wm++)
                for (int wn = 0; wn < WARP_N; wn++)
                    wmma::mma_sync(acc[wm][wn], a_frag[wm], b_frag[wn], acc[wm][wn]);
        }

        __syncthreads();
    }

    /* Store — each warp writes its WARP_M × WARP_N tiles */
    int base_row = ctaRow + warp_row * WARP_M * WMMA_M;
    int base_col = ctaCol + warp_col * WARP_N * WMMA_N;

    for (int wm = 0; wm < WARP_M; wm++) {
        for (int wn = 0; wn < WARP_N; wn++) {
            int outRow = base_row + wm * WMMA_M;
            int outCol = base_col + wn * WMMA_N;
            if (outRow < M && outCol < N) {
                /* Store float acc to temp, convert to bf16 */
                __shared__ float tmp[NUM_WARPS][WMMA_M * WMMA_N];
                wmma::store_matrix_sync(&tmp[warp_id][0], acc[wm][wn], WMMA_N, wmma::mem_row_major);
                __syncthreads();
                int lane = tid % 32;
                for (int i = lane; i < WMMA_M * WMMA_N; i += 32) {
                    int r = outRow + i / WMMA_N;
                    int c = outCol + i % WMMA_N;
                    if (r < M && c < N)
                        C[r * N + c] = __float2bfloat16(tmp[warp_id][i]);
                }
                __syncthreads();
            }
        }
    }
}

/* ── Host ─────────────────────────────────────────────────────────── */

extern "C" int kernel_run(
    __nv_bfloat16** inputs, int num_inputs,
    __nv_bfloat16** outputs, int num_outputs,
    int n, cudaStream_t stream)
{
    const char* shape_env = getenv("CUDA_EXEC_PARAM_SHAPE");
    int M = 1024, N = 1024, K = 1024;
    if (shape_env) {
        sscanf(shape_env, "[%d, %d]", &M, &N);
        K = M;
    }

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    int smem = sizeof(__nv_bfloat16) * 2 * (BM * BK + BK * BN);
    cudaFuncSetAttribute(matmul_wmma, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    matmul_wmma<<<grid, THREADS, 0, stream>>>(
        inputs[0], inputs[1], outputs[0], M, N, K);
    return 0;
}
