/*
 * SM90 BF16×BF16→BF16 GEMM Kernel
 *
 * Uses WGMMA (m64n128k16.f32.bf16.bf16) with inline PTX.
 * Single-stage (no pipelining) for v1 — correctness first.
 *
 * Interface: kernel_run(inputs[A,B], outputs[C], n, stream)
 *   A[M][K], B[K][N], C[M][N] all BF16.  K == M (square) per harness.
 *
 * Grid: (ceil(M/BM), ceil(N/BN)), Block: 128 threads = 1 warpgroup.
 * BM=64, BN=128, BK=64 (tunable via #define).
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>

typedef __nv_bfloat16 bf16;

/* ---- Tile parameters ---- */
#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 128
#endif
#ifndef BK
#define BK 64
#endif

/* WGMMA instruction shape */
#define WGMMA_M  64
#define WGMMA_N 128
#define WGMMA_K  16

/* fp32 accumulators per thread: (WGMMA_M × WGMMA_N) / 128 threads = 64 */
#define N_ACCUM 64

/* =========================================================================
 * WGMMA helpers
 * ========================================================================= */

/*
 * Build a WGMMA shared-memory descriptor (no swizzle).
 *   bits [13:0]  = byte_address_of_start / 16
 *   bits [29:16] = leading_dimension_bytes / 16  (row stride)
 */
__device__ __forceinline__
uint64_t make_smem_desc(const void* ptr, uint32_t ld_bytes)
{
    uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0;
    desc |= (uint64_t)(saddr >> 4) & 0x3FFFull;                    /* [13:0]  */
    desc |= (uint64_t)((ld_bytes >> 4) & 0x3FFFull) << 16;         /* [29:16] */
    return desc;
}

__device__ __forceinline__ void wgmma_fence()
{
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit()
{
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wgmma_wait0()
{
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
}

/*
 * WGMMA: acc[64] += A[64×16] × B[16×128]
 *   A, B in shared memory (via descriptors).
 *   scale_d=1 (accumulate), im_scale_A=1, im_scale_B=1, trans_A=0, trans_B=0.
 */
__device__ __forceinline__
void wgmma_m64n128k16(float d[N_ACCUM], uint64_t descA, uint64_t descB)
{
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{"
        "%0,%1,%2,%3,%4,%5,%6,%7,"
        "%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,"
        "%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,"
        "%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,"
        "%56,%57,%58,%59,%60,%61,%62,%63"
        "},"
        "%64, %65, 1, 1, 1, 0, 0;\n"
        :
        "+f"(d[ 0]),"+f"(d[ 1]),"+f"(d[ 2]),"+f"(d[ 3]),
        "+f"(d[ 4]),"+f"(d[ 5]),"+f"(d[ 6]),"+f"(d[ 7]),
        "+f"(d[ 8]),"+f"(d[ 9]),"+f"(d[10]),"+f"(d[11]),
        "+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
        "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),
        "+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
        "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),
        "+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),
        "+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),
        "+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),
        "+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),
        "+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),
        "+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),
        "+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),
        "+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),
        "+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])
        :
        "l"(descA), "l"(descB)
    );
}

/* =========================================================================
 * Main kernel
 * ========================================================================= */

__global__ void __launch_bounds__(128, 1)
matmul_kernel(const bf16* __restrict__ A,
              const bf16* __restrict__ B,
              bf16*       __restrict__ C,
              int M, int N, int K)
{
    const int bm = blockIdx.x * BM;   /* tile row base */
    const int bn = blockIdx.y * BN;   /* tile col base */
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;     /* 0..3 */
    const int lane_id = tid & 31;     /* 0..31 */

    /* Shared memory layout:
     *   A_smem[BM][BK]  (row-major, ld = BK)
     *   B_smem[BK][BN]  (row-major, ld = BN)
     */
    extern __shared__ __align__(128) char smem_raw[];
    bf16* __restrict__ A_smem = reinterpret_cast<bf16*>(smem_raw);
    bf16* __restrict__ B_smem = A_smem + BM * BK;

    /* FP32 accumulator registers */
    float acc[N_ACCUM];
    #pragma unroll
    for (int i = 0; i < N_ACCUM; i++) acc[i] = 0.0f;

    /* ------------------------------------------------------------------ *
     * K-dimension loop                                                    *
     * ------------------------------------------------------------------ */
    for (int k = 0; k < K; k += BK) {

        /* --- Load A tile: A_smem[BM][BK] from A[bm..][k..] --- */
        #pragma unroll 4
        for (int idx = tid; idx < BM * BK; idx += 128) {
            int r = idx / BK, c = idx % BK;
            int gr = bm + r, gc = k + c;
            A_smem[idx] = (gr < M && gc < K) ? A[gr * K + gc]
                                              : __float2bfloat16(0.0f);
        }

        /* --- Load B tile: B_smem[BK][BN] from B[k..][bn..] --- */
        #pragma unroll 4
        for (int idx = tid; idx < BK * BN; idx += 128) {
            int r = idx / BN, c = idx % BN;
            int gr = k + r, gc = bn + c;
            B_smem[idx] = (gr < K && gc < N) ? B[gr * N + gc]
                                              : __float2bfloat16(0.0f);
        }

        __syncthreads();

        /* --- WGMMA over BK/WGMMA_K steps --- */
        wgmma_fence();

        #pragma unroll
        for (int ki = 0; ki < BK; ki += WGMMA_K) {
            /*
             * A descriptor: tile A[:,ki:ki+16]
             *   start = A_smem[0][ki]  = A_smem + ki
             *   LDM   = BK * sizeof(bf16) = BK*2 bytes
             */
            uint64_t descA = make_smem_desc(A_smem + ki,
                                            (uint32_t)(BK * sizeof(bf16)));
            /*
             * B descriptor: tile B[ki:ki+16,:]
             *   start = B_smem[ki][0] = B_smem + ki*BN
             *   LDM   = BN * sizeof(bf16) = BN*2 bytes
             */
            uint64_t descB = make_smem_desc(B_smem + ki * BN,
                                            (uint32_t)(BN * sizeof(bf16)));

            wgmma_m64n128k16(acc, descA, descB);
        }

        wgmma_commit();
        wgmma_wait0();

        __syncthreads();
    }

    /* ------------------------------------------------------------------ *
     * Epilogue: write fp32 accumulators → bf16 output C                  *
     *                                                                     *
     * WGMMA m64n128k16 output layout (validated analytically):           *
     *   thread t = warp_id*32 + lane_id                                  *
     *   register i = 0..63:                                               *
     *     n_tile = i / 4          (column group: 0..15, 8 cols each)     *
     *     loc    = i % 4          (position within group)                 *
     *     row    = warp_id*16 + lane_id/4 + 8*(loc/2)                    *
     *     col    = n_tile*8 + (lane_id%4)*2 + (loc%2)                    *
     * ------------------------------------------------------------------ */
    #pragma unroll
    for (int i = 0; i < N_ACCUM; i++) {
        const int n_tile = i >> 2;                    /* i / 4  */
        const int loc    = i & 3;                     /* i % 4  */
        const int row = (warp_id << 4)                /* warp*16 */
                      + (lane_id >> 2)                /* lane/4  */
                      + ((loc >> 1) << 3);            /* 8*(loc/2) */
        const int col = (n_tile << 3)                 /* n_tile*8 */
                      + ((lane_id & 3) << 1)          /* (lane%4)*2 */
                      + (loc & 1);                    /* loc%2 */

        const int out_r = bm + row;
        const int out_c = bn + col;

        if (out_r < M && out_c < N) {
            C[out_r * N + out_c] = __float2bfloat16(acc[i]);
        }
    }
}

/* =========================================================================
 * Harness entry point
 * ========================================================================= */

static void parse_shape(int n, int* pM, int* pN, int* pK)
{
    int M = 0, N_val = 0;
    const char* s = getenv("CUDA_EXEC_PARAM_SHAPE");
    if (s) {
        while (*s == '[' || *s == ' ') ++s;
        while (*s >= '0' && *s <= '9') { M = M * 10 + (*s++ - '0'); }
        while (*s == ',' || *s == ' ') ++s;
        while (*s >= '0' && *s <= '9') { N_val = N_val * 10 + (*s++ - '0'); }
    }
    if (!M || !N_val) {
        M = (int)sqrtf((float)n);
        N_val = M;
    }
    *pM = M; *pN = N_val; *pK = M;   /* square: K == M */
}

extern "C" int kernel_run(
    __nv_bfloat16** inputs,  int num_inputs,
    __nv_bfloat16** outputs, int num_outputs,
    int n, cudaStream_t stream)
{
    if (num_inputs < 2 || num_outputs < 1) return -1;

    int M, N, K;
    parse_shape(n, &M, &N, &K);
    if (M <= 0 || N <= 0 || K <= 0) return -2;

    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(128);
    size_t smem_bytes = (size_t)(BM * BK + BK * BN) * sizeof(bf16);

    matmul_kernel<<<grid, block, smem_bytes, stream>>>(
        inputs[0], inputs[1], outputs[0], M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "matmul_kernel launch error: %s\n",
                cudaGetErrorString(err));
        return -3;
    }
    return 0;
}
