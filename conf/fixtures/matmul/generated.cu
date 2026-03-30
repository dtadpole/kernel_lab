/*
 * BF16 tiled matrix multiplication kernel for cuda_exec evaluation.
 *
 * Implements the kernel_run contract:
 *   extern "C" int kernel_run(__nv_bfloat16** inputs, int num_inputs,
 *                             __nv_bfloat16** outputs, int num_outputs,
 *                             int n, cudaStream_t stream);
 *
 * Optimization techniques:
 *   1. Shared-memory tiling (32x32) — each threadblock loads a tile of A and B
 *      into SMEM, reducing global memory bandwidth by a factor of TILE_SIZE.
 *   2. Double buffering — two SMEM buffers alternate: the next tile is prefetched
 *      while the current tile is being consumed in the FMA loop, hiding GMEM
 *      latency behind compute.
 *   3. Vectorized loads — nv_bfloat162 (2-wide) loads cut the number of GMEM
 *      transactions in half for tile prefetch.
 *   4. BF16 storage / FP32 accumulation — inputs and output are __nv_bfloat16;
 *      the multiply-accumulate runs in FP32 for numerical accuracy, converted
 *      back to BF16 on store.
 *   5. Matrix dimensions are read from CUDA_EXEC_PARAM_SHAPE when available,
 *      with a sqrt(n) fallback for square matrices.
 */
#include <cuda_bf16.h>

#define TILE_SIZE 32

/* --------------------------------------------------------------------------
 * Double-buffered tiled GEMM:  C = A * B
 *   A is M x K (row-major), B is K x N (row-major), C is M x N (row-major).
 *   All pointers are __nv_bfloat16; accumulation is FP32.
 *
 *   Shared memory layout (per threadblock):
 *     smem_a[2][TILE_SIZE][TILE_SIZE]  — double-buffered tiles of A
 *     smem_b[2][TILE_SIZE][TILE_SIZE]  — double-buffered tiles of B
 * -------------------------------------------------------------------------- */
__global__ void matmul_bf16(const __nv_bfloat16* __restrict__ A,
                            const __nv_bfloat16* __restrict__ B,
                            __nv_bfloat16* __restrict__ C,
                            int M, int N, int K) {
    /* Double-buffered shared memory tiles. */
    __shared__ __nv_bfloat16 smem_a[2][TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 smem_b[2][TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;           /* column within tile  */
    const int ty = threadIdx.y;           /* row within tile     */
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;

    float acc = 0.0f;

    const int num_tiles = K / TILE_SIZE;  /* K is a multiple of TILE_SIZE */

    /* ---- Load the first tile (buf = 0) --------------------------------- */
    int tile = 0;
    {
        int a_col = tile * TILE_SIZE + tx;
        int b_row = tile * TILE_SIZE + ty;
        smem_a[0][ty][tx] = A[row * K + a_col];
        smem_b[0][ty][tx] = B[b_row * N + col];
    }
    __syncthreads();

    /* ---- Main loop: compute current tile while prefetching next tile ---- */
    for (tile = 0; tile < num_tiles - 1; ++tile) {
        int cur = tile & 1;
        int nxt = 1 - cur;

        /* Prefetch tile (tile+1) into buffer nxt using vectorized loads
         * where alignment permits.  Each thread loads one element of A and
         * one element of B; the vectorised path packs two adjacent BF16
         * values into a single 32-bit load when the thread's column index
         * is even and there is room for a pair. */
        {
            int next_tile = tile + 1;
            int a_col = next_tile * TILE_SIZE + tx;
            int b_row = next_tile * TILE_SIZE + ty;

            /* --- Vectorized load for A tile --- */
            if ((tx & 1) == 0 && tx + 1 < TILE_SIZE) {
                const __nv_bfloat162* src_a =
                    reinterpret_cast<const __nv_bfloat162*>(&A[row * K + a_col]);
                __nv_bfloat162 va = *src_a;
                smem_a[nxt][ty][tx]     = __low2bfloat16(va);
                smem_a[nxt][ty][tx + 1] = __high2bfloat16(va);
            }
            /* Odd columns already handled by the even-column thread above,
             * but we still need the odd-column threads to load B. */

            /* --- Vectorized load for B tile --- */
            if ((tx & 1) == 0 && tx + 1 < TILE_SIZE) {
                const __nv_bfloat162* src_b =
                    reinterpret_cast<const __nv_bfloat162*>(&B[b_row * N + col]);
                __nv_bfloat162 vb = *src_b;
                smem_b[nxt][ty][tx]     = __low2bfloat16(vb);
                smem_b[nxt][ty][tx + 1] = __high2bfloat16(vb);
            }
        }

        /* Compute on current tile (cur). */
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            float a_val = __bfloat162float(smem_a[cur][ty][k]);
            float b_val = __bfloat162float(smem_b[cur][k][tx]);
            acc = fmaf(a_val, b_val, acc);
        }

        __syncthreads();  /* Ensure prefetch writes are visible. */
    }

    /* ---- Process the last tile (no prefetch needed) --------------------- */
    {
        int cur = (num_tiles - 1) & 1;
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            float a_val = __bfloat162float(smem_a[cur][ty][k]);
            float b_val = __bfloat162float(smem_b[cur][k][tx]);
            acc = fmaf(a_val, b_val, acc);
        }
    }

    /* ---- Store result --------------------------------------------------- */
    C[row * N + col] = __float2bfloat16(acc);
}

/* --------------------------------------------------------------------------
 * kernel_run — entry point called by the cuda_exec harness.
 * -------------------------------------------------------------------------- */
extern "C" int kernel_run(__nv_bfloat16** inputs, int num_inputs,
                          __nv_bfloat16** outputs, int num_outputs,
                          int n, cudaStream_t stream) {
    const __nv_bfloat16* A = inputs[0];
    const __nv_bfloat16* B = inputs[1];
    __nv_bfloat16*       C = outputs[0];

    /* --- Determine M, N, K ----------------------------------------------- */
    int M, N, K;

    /* Try to read shape from environment (JSON array "[M, K]" or "[M, N]"). */
    const char* shape = getenv("CUDA_EXEC_PARAM_SHAPE");
    if (shape) {
        /* Minimal JSON parse: expect "[<int>, <int>]". */
        int d0 = 0, d1 = 0;
        const char* p = shape;
        while (*p && *p != '[') ++p;
        if (*p == '[') ++p;
        while (*p == ' ') ++p;
        while (*p >= '0' && *p <= '9') { d0 = d0 * 10 + (*p - '0'); ++p; }
        while (*p == ' ' || *p == ',') ++p;
        while (*p >= '0' && *p <= '9') { d1 = d1 * 10 + (*p - '0'); ++p; }
        M = d0;
        N = d1;
        K = d0;  /* Square matrices: K == M */
    } else {
        /* Fallback for square matrices. */
        M = (int)sqrtf((float)n);
        N = M;
        K = M;
    }

    /* --- Launch ---------------------------------------------------------- */
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_bf16<<<blocks, threads, 0, stream>>>(A, B, C, M, N, K);

    return 0;
}
