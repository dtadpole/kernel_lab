/*
 * BF16 matmul for SM90 — WMMA tensor cores, 128×128 tiles, double buffering.
 *
 * CTA tile: 128×128, 8 warps (4×2 layout), WMMA 16×16×16.
 * Each warp: 2×4 WMMA tiles = 32×64 output.
 * Double-buffered shared memory for K-loop pipelining.
 * FP32 accumulation → BF16 output.
 */
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32
#define TILE_M 128
#define TILE_N 128
#define TILE_K 16
#define WARPS 8         /* 4×2 warp layout */
#define THREADS (WARPS * WARP_SIZE)  /* 256 */
#define WARP_ROWS 4
#define WARP_COLS 2
/* Each warp: 2 WMMA tiles in M (32 rows), 4 WMMA tiles in N (64 cols) */
#define WARP_WMMA_M 2
#define WARP_WMMA_N 4

__global__ void matmul_wmma128(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    /* Double-buffered shared memory */
    __shared__ __nv_bfloat16 sA[2][TILE_M][TILE_K];
    __shared__ __nv_bfloat16 sB[2][TILE_K][TILE_N];

    int warpId = threadIdx.x / WARP_SIZE;
    int warpRow = warpId / WARP_COLS;  /* 0..3 */
    int warpCol = warpId % WARP_COLS;  /* 0..1 */

    int ctaRow = blockIdx.y * TILE_M;
    int ctaCol = blockIdx.x * TILE_N;
    int tid = threadIdx.x;

    /* Accumulators: each warp has 2×4 WMMA tiles */
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WARP_WMMA_M][WARP_WMMA_N];
    for (int i = 0; i < WARP_WMMA_M; i++)
        for (int j = 0; j < WARP_WMMA_N; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    /* Load first tile into buffer 0 */
    for (int i = tid; i < TILE_M * TILE_K; i += THREADS) {
        int r = i / TILE_K, c = i % TILE_K;
        int gR = ctaRow + r, gC = c;
        sA[0][r][c] = (gR < M && gC < K) ? A[gR * K + gC] : __float2bfloat16(0.0f);
    }
    for (int i = tid; i < TILE_K * TILE_N; i += THREADS) {
        int r = i / TILE_N, c = i % TILE_N;
        int gR = r, gC = ctaCol + c;
        sB[0][r][c] = (gR < K && gC < N) ? B[gR * N + gC] : __float2bfloat16(0.0f);
    }
    __syncthreads();

    int numK = (K + TILE_K - 1) / TILE_K;

    for (int kt = 0; kt < numK; kt++) {
        int cur = kt & 1;
        int nxt = 1 - cur;

        /* Prefetch next tile into other buffer */
        if (kt + 1 < numK) {
            int nextK = (kt + 1) * TILE_K;
            for (int i = tid; i < TILE_M * TILE_K; i += THREADS) {
                int r = i / TILE_K, c = i % TILE_K;
                int gR = ctaRow + r, gC = nextK + c;
                sA[nxt][r][c] = (gR < M && gC < K) ? A[gR * K + gC] : __float2bfloat16(0.0f);
            }
            for (int i = tid; i < TILE_K * TILE_N; i += THREADS) {
                int r = i / TILE_N, c = i % TILE_N;
                int gR = nextK + r, gC = ctaCol + c;
                sB[nxt][r][c] = (gR < K && gC < N) ? B[gR * N + gC] : __float2bfloat16(0.0f);
            }
        }

        /* Compute: each warp does WARP_WMMA_M × WARP_WMMA_N WMMA ops */
        for (int wi = 0; wi < WARP_WMMA_M; wi++) {
            for (int wj = 0; wj < WARP_WMMA_N; wj++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;

                int aRow = warpRow * (TILE_M / WARP_ROWS) + wi * WMMA_M;
                int bCol = warpCol * (TILE_N / WARP_COLS) + wj * WMMA_N;

                wmma::load_matrix_sync(a_frag, &sA[cur][aRow][0], TILE_K);
                wmma::load_matrix_sync(b_frag, &sB[cur][0][bCol], TILE_N);
                wmma::mma_sync(acc[wi][wj], a_frag, b_frag, acc[wi][wj]);
            }
        }
        __syncthreads();
    }

    /* Store: FP32 acc → BF16 global directly from fragment registers.
     * Each WMMA 16×16 accumulator has 8 FP32 elements per thread (256 total).
     * WMMA row-major accumulator layout: thread t holds elements at
     * specific (row, col) positions. We use store_matrix_sync to shared
     * with per-warp offset to avoid conflicts. */
    /* Per-warp temp: 8 warps × 16×16 × 4 bytes = 8KB — fits in SMEM */
    __shared__ float sOut[WARPS][WMMA_M][WMMA_N];

    for (int wi = 0; wi < WARP_WMMA_M; wi++) {
        for (int wj = 0; wj < WARP_WMMA_N; wj++) {
            int outRow = ctaRow + warpRow * (TILE_M / WARP_ROWS) + wi * WMMA_M;
            int outCol = ctaCol + warpCol * (TILE_N / WARP_COLS) + wj * WMMA_N;

            wmma::store_matrix_sync(&sOut[warpId][0][0], acc[wi][wj], WMMA_N, wmma::mem_row_major);

            int laneId = threadIdx.x % WARP_SIZE;
            for (int idx = laneId; idx < WMMA_M * WMMA_N; idx += WARP_SIZE) {
                int r = idx / WMMA_N;
                int c = idx % WMMA_N;
                int gR = outRow + r;
                int gC = outCol + c;
                if (gR < M && gC < N)
                    C[gR * N + gC] = __float2bfloat16(sOut[warpId][r][c]);
            }
        }
    }
}

extern "C" int kernel_run(
    __nv_bfloat16** inputs, int num_inputs,
    __nv_bfloat16** outputs, int num_outputs,
    int n, cudaStream_t stream)
{
    int dim = (int)sqrtf((float)n);
    if (dim * dim != n) return 1;

    dim3 block(THREADS);
    dim3 grid((dim + TILE_N - 1) / TILE_N, (dim + TILE_M - 1) / TILE_M);
    matmul_wmma128<<<grid, block, 0, stream>>>(inputs[0], inputs[1], outputs[0], dim, dim, dim);
    return 0;
}
