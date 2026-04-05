/*
 * BF16 matrix multiplication for SM90 (H100) — Tensor Core via mma.sync.
 *
 * Uses mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
 * Each warp computes a 16×8 output tile per mma instruction.
 * Tile: 64×64 per CTA, K-loop with 16-element steps.
 * FP32 accumulation → BF16 output.
 *
 * kernel_run contract:
 *   inputs[0] = A (M*K elements, row-major BF16)
 *   inputs[1] = B (K*N elements, row-major BF16)
 *   outputs[0] = C (M*N elements, row-major BF16)
 *   n = M*K (square: M=K=N=sqrt(n))
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
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define WARPS_PER_BLOCK 4  /* 2×2 warp layout */

__global__ void matmul_wmma(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    /* Shared memory for A and B tiles */
    __shared__ __nv_bfloat16 sA[TILE_M][TILE_K];
    __shared__ __nv_bfloat16 sB[TILE_K][TILE_N];

    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    /* 2×2 warp layout within the 64×64 CTA tile */
    int warpRow = warpId / 2;  /* 0 or 1 */
    int warpCol = warpId % 2;  /* 0 or 1 */

    /* Each warp handles 32×32 of the 64×64 tile (2×2 WMMA tiles) */
    int ctaRow = blockIdx.y * TILE_M;
    int ctaCol = blockIdx.x * TILE_N;

    /* Declare WMMA fragments — 2×2 per warp */
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    /* K-loop */
    for (int k = 0; k < K; k += TILE_K) {
        /* Cooperative load A tile (64×16) and B tile (16×64) */
        /* 128 threads, each loads multiple elements */
        int tid = threadIdx.x;
        /* Load A: 64 rows × 16 cols = 1024 elements, 128 threads → 8 each */
        for (int i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            int r = i / TILE_K;
            int c = i % TILE_K;
            int gRow = ctaRow + r;
            int gCol = k + c;
            sA[r][c] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : __float2bfloat16(0.0f);
        }
        /* Load B: 16 rows × 64 cols = 1024 elements */
        for (int i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
            int r = i / TILE_N;
            int c = i % TILE_N;
            int gRow = k + r;
            int gCol = ctaCol + c;
            sB[r][c] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : __float2bfloat16(0.0f);
        }
        __syncthreads();

        /* Each warp does 2×2 WMMA on its 32×32 sub-tile */
        for (int wi = 0; wi < 2; wi++) {
            for (int wj = 0; wj < 2; wj++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;

                int aRow = warpRow * 32 + wi * WMMA_M;
                int bCol = warpCol * 32 + wj * WMMA_N;

                wmma::load_matrix_sync(a_frag, &sA[aRow][0], TILE_K);
                wmma::load_matrix_sync(b_frag, &sB[0][bCol], TILE_N);
                wmma::mma_sync(acc[wi][wj], a_frag, b_frag, acc[wi][wj]);
            }
        }
        __syncthreads();
    }

    /* Store results: FP32 acc → BF16 output via shared memory */
    __shared__ float sC[TILE_M][TILE_N];

    for (int wi = 0; wi < 2; wi++) {
        for (int wj = 0; wj < 2; wj++) {
            int outRow = warpRow * 32 + wi * WMMA_M;
            int outCol = warpCol * 32 + wj * WMMA_N;

            /* Store FP32 accumulators to shared memory */
            wmma::store_matrix_sync(&sC[outRow][outCol], acc[wi][wj], TILE_N, wmma::mem_row_major);
        }
    }
    __syncthreads();

    /* Convert FP32 → BF16 and write to global memory */
    int tid = threadIdx.x;
    for (int i = tid; i < TILE_M * TILE_N; i += blockDim.x) {
        int r = i / TILE_N;
        int c = i % TILE_N;
        int gRow = ctaRow + r;
        int gCol = ctaCol + c;
        if (gRow < M && gCol < N)
            C[gRow * N + gCol] = __float2bfloat16(sC[r][c]);
    }
}

extern "C" int kernel_run(
    __nv_bfloat16** inputs, int num_inputs,
    __nv_bfloat16** outputs, int num_outputs,
    int n, cudaStream_t stream)
{
    int dim = (int)sqrtf((float)n);
    if (dim * dim != n) {
        fprintf(stderr, "matmul: n=%d is not a perfect square\n", n);
        return 1;
    }

    const __nv_bfloat16* A = inputs[0];
    const __nv_bfloat16* B = inputs[1];
    __nv_bfloat16* C = outputs[0];

    dim3 block(WARPS_PER_BLOCK * WARP_SIZE);  /* 128 threads */
    dim3 grid((dim + TILE_N - 1) / TILE_N, (dim + TILE_M - 1) / TILE_M);

    matmul_wmma<<<grid, block, 0, stream>>>(A, B, C, dim, dim, dim);
    return 0;
}
