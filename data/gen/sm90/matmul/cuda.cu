/*
 * BF16 matrix multiplication kernel for SM90 (H100).
 *
 * Simple tiled implementation using shared memory.
 * Computes C = A * B where A(M,K), B(K,N), C(M,N) are all BF16.
 * Accumulation in FP32 for numerical accuracy.
 *
 * kernel_run contract:
 *   inputs[0] = A (M*K elements, row-major)
 *   inputs[1] = B (K*N elements, row-major)
 *   outputs[0] = C (M*N elements, row-major)
 *   n = M*K = K*N (square matrices: M=K=N=sqrt(n))
 */
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>

#define TILE 32

__global__ void matmul_tiled(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K)
            ? __bfloat162float(A[row * K + aCol])
            : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N)
            ? __bfloat162float(B[bRow * N + col])
            : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE; i++) {
            acc += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16(acc);
    }
}

extern "C" int kernel_run(
    __nv_bfloat16** inputs, int num_inputs,
    __nv_bfloat16** outputs, int num_outputs,
    int n, cudaStream_t stream)
{
    /* Square matrices: M = K = N = sqrt(n) */
    int dim = (int)sqrtf((float)n);
    if (dim * dim != n) {
        fprintf(stderr, "matmul: n=%d is not a perfect square\n", n);
        return 1;
    }

    const __nv_bfloat16* A = inputs[0];
    const __nv_bfloat16* B = inputs[1];
    __nv_bfloat16* C = outputs[0];

    dim3 block(TILE, TILE);
    dim3 grid((dim + TILE - 1) / TILE, (dim + TILE - 1) / TILE);

    matmul_tiled<<<grid, block, 0, stream>>>(A, B, C, dim, dim, dim);
    return 0;
}
