/*
 * BF16 matmul for SM90 — WMMA + cp.async, 128×128×32 tiles, double buffer.
 *
 * Improvements over previous:
 * - TILE_K=32 (2 WMMA K-steps per tile load)
 * - cp.async for async global→shared memory copies
 * - Vectorized 64-bit loads (4 × BF16 per load)
 * - cp.async.commit_group / cp.async.wait_group for pipelining
 */
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
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
#define TILE_K 32
#define WARPS 8
#define THREADS (WARPS * WARP_SIZE)
#define WARP_ROWS 4
#define WARP_COLS 2
#define WARP_WMMA_M 2
#define WARP_WMMA_N 4

/* Shared memory: double-buffered A and B tiles */
/* A: 128×32 = 8KB per buffer, B: 32×128 = 8KB per buffer → 32KB total */

__device__ __forceinline__
void load_tile_async(
    __nv_bfloat16 sA[][TILE_K],
    __nv_bfloat16 sB[][TILE_N],
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    int ctaRow, int ctaCol, int k,
    int M, int N, int K, int tid)
{
    /* Load A tile: 128 rows × 32 cols = 4096 BF16 = 8KB */
    /* 256 threads, each loads 16 elements (8 × uint32 = 4 BF16 per uint32) */
    for (int i = tid; i < TILE_M * TILE_K / 4; i += THREADS) {
        int elem_idx = i * 4;
        int r = elem_idx / TILE_K;
        int c = elem_idx % TILE_K;
        int gR = ctaRow + r;
        int gC = k + c;

        if (gR < M && gC + 3 < K) {
            /* Vectorized 64-bit async copy (4 × BF16) */
            __pipeline_memcpy_async(
                &sA[r][c],
                &A[gR * K + gC],
                8  /* 4 × sizeof(BF16) = 8 bytes */
            );
        } else {
            /* Boundary: scalar fallback */
            for (int j = 0; j < 4; j++) {
                sA[r][c + j] = (gR < M && gC + j < K)
                    ? A[gR * K + gC + j]
                    : __float2bfloat16(0.0f);
            }
        }
    }

    /* Load B tile: 32 rows × 128 cols = 4096 BF16 = 8KB */
    for (int i = tid; i < TILE_K * TILE_N / 4; i += THREADS) {
        int elem_idx = i * 4;
        int r = elem_idx / TILE_N;
        int c = elem_idx % TILE_N;
        int gR = k + r;
        int gC = ctaCol + c;

        if (gR < K && gC + 3 < N) {
            __pipeline_memcpy_async(
                &sB[r][c],
                &B[gR * N + gC],
                8
            );
        } else {
            for (int j = 0; j < 4; j++) {
                sB[r][c + j] = (gR < K && gC + j < N)
                    ? B[gR * N + gC + j]
                    : __float2bfloat16(0.0f);
            }
        }
    }
}

__global__ void matmul_wmma_async(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    __shared__ __nv_bfloat16 sA[2][TILE_M][TILE_K];
    __shared__ __nv_bfloat16 sB[2][TILE_K][TILE_N];

    int warpId = threadIdx.x / WARP_SIZE;
    int warpRow = warpId / WARP_COLS;
    int warpCol = warpId % WARP_COLS;
    int ctaRow = blockIdx.y * TILE_M;
    int ctaCol = blockIdx.x * TILE_N;
    int tid = threadIdx.x;

    /* Accumulators */
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WARP_WMMA_M][WARP_WMMA_N];
    for (int i = 0; i < WARP_WMMA_M; i++)
        for (int j = 0; j < WARP_WMMA_N; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    /* Load first tile asynchronously */
    load_tile_async(sA[0], sB[0], A, B, ctaRow, ctaCol, 0, M, N, K, tid);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    int numK = (K + TILE_K - 1) / TILE_K;

    for (int kt = 0; kt < numK; kt++) {
        int cur = kt & 1;
        int nxt = 1 - cur;

        /* Prefetch next tile */
        if (kt + 1 < numK) {
            load_tile_async(sA[nxt], sB[nxt], A, B, ctaRow, ctaCol,
                           (kt + 1) * TILE_K, M, N, K, tid);
            __pipeline_commit();
        }

        /* Compute: 2 WMMA K-steps per TILE_K=32 */
        for (int kk = 0; kk < TILE_K; kk += WMMA_K) {
            for (int wi = 0; wi < WARP_WMMA_M; wi++) {
                for (int wj = 0; wj < WARP_WMMA_N; wj++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                                   __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                   __nv_bfloat16, wmma::row_major> b_frag;

                    int aRow = warpRow * (TILE_M / WARP_ROWS) + wi * WMMA_M;
                    int bCol = warpCol * (TILE_N / WARP_COLS) + wj * WMMA_N;

                    wmma::load_matrix_sync(a_frag, &sA[cur][aRow][kk], TILE_K);
                    wmma::load_matrix_sync(b_frag, &sB[cur][kk][bCol], TILE_N);
                    wmma::mma_sync(acc[wi][wj], a_frag, b_frag, acc[wi][wj]);
                }
            }
        }

        /* Wait for prefetch before swapping buffers */
        if (kt + 1 < numK) {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
    }

    /* Store: per-warp temp buffer for FP32 → BF16 conversion */
    __shared__ float sOut[WARPS][WMMA_M][WMMA_N];

    for (int wi = 0; wi < WARP_WMMA_M; wi++) {
        for (int wj = 0; wj < WARP_WMMA_N; wj++) {
            int outRow = ctaRow + warpRow * (TILE_M / WARP_ROWS) + wi * WMMA_M;
            int outCol = ctaCol + warpCol * (TILE_N / WARP_COLS) + wj * WMMA_N;

            wmma::store_matrix_sync(&sOut[warpId][0][0], acc[wi][wj],
                                    WMMA_N, wmma::mem_row_major);

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
    matmul_wmma_async<<<grid, block, 0, stream>>>(
        inputs[0], inputs[1], outputs[0], dim, dim, dim);
    return 0;
}
