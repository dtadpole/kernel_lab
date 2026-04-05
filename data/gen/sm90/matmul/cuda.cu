/*
 * BF16 matmul for SM90 — WGMMA m64n128k16 via inline PTX.
 *
 * 1 warpgroup (128 threads), CTA tile 128×128, TILE_K=64.
 * 2× m64n128k16 per K-step (rows 0-63 and 64-127) × 4 k16 steps.
 * 128B swizzle SMEM layout, double buffer.
 * FP32 accumulation → BF16 output.
 */
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

#define TILE_M    128
#define TILE_N    128
#define TILE_K    64
#define THREADS   128
#define STAGES    2

#define A_BYTES   (TILE_M * TILE_K * 2)   /* 128×64×2 = 16384 */
#define B_BYTES   (TILE_N * TILE_K * 2)   /* 128×64×2 = 16384 */
#define STAGE_BYTES (A_BYTES + B_BYTES)    /* 32768 */

/* =========================================================================
 * WGMMA descriptor (128B swizzle layout)
 * ========================================================================= */
__device__ __forceinline__
uint64_t make_desc(const void* smem_ptr, int tile_k) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    int stride_16B = (8 * tile_k * 2) >> 4;  /* 8 rows × K cols × 2 bytes */
    uint64_t desc = 0;
    desc |= (uint64_t)((addr >> 4) & 0x3FFF);               // start_address
    desc |= (uint64_t)(1) << 16;                              // leading_byte_off
    desc |= (uint64_t)(stride_16B & 0x3FFF) << 32;           // stride_byte_off
    desc |= (uint64_t)(1) << 62;                              // layout_type = B128
    return desc;
}

__device__ __forceinline__
uint64_t desc_advance(uint64_t desc, int offset_16B) {
    uint32_t lo = (uint32_t)desc + (uint32_t)offset_16B;
    uint32_t hi = (uint32_t)(desc >> 32);
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

/* =========================================================================
 * WGMMA helpers
 * ========================================================================= */
/* 128B XOR swizzle: permutes the 16B-granule index within each 128B line
 * by XOR with the line index (bits [9:7] of byte offset).
 * This eliminates shared memory bank conflicts for WGMMA access patterns.
 * byte_offset → swizzled_byte_offset within the matrix. */
__device__ __forceinline__
int swizzle_128B(int byte_offset) {
    return byte_offset ^ (((byte_offset >> 7) & 0x7) << 4);
}

__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_wait0() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

/* Single m64n128k16 WGMMA instruction */
__device__ __forceinline__
void wgmma_m64n128k16(float (&d)[64], uint64_t da, uint64_t db, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
        "%64,%65,p,1,1,0,0;\n}\n"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),
          "+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
          "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),
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
        : "l"(da), "l"(db), "r"(scale_d));
}

/* =========================================================================
 * Epilogue: WGMMA m64n128 accumulator → global
 *
 * Register pair j (0..31): acc[4*(j/2) + 2*(j%2)] and acc[4*(j/2) + 2*(j%2) + 1]
 * row = warp*16 + (j%2)*8 + lane/4
 * col = (j/2)*8 + (lane%4)*2 + {0,1}
 * ========================================================================= */
__device__ __forceinline__
void store_acc(__nv_bfloat16* C, float (&acc)[64],
               int ctaRow, int ctaCol, int M, int N) {
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row_base = ctaRow + warp * 16 + lane / 4;
    int col_base = ctaCol + (lane % 4) * 2;

    /* acc[4p+s]: row = row_base + (s/2)*8, col = col_base + p*8 + (s%2) */
    for (int p = 0; p < 16; p++) {
        int col = col_base + p * 8;
        int row0 = row_base;       /* s=0,1 */
        int row8 = row_base + 8;   /* s=2,3 */

        if (row0 < M && col < N)
            C[row0 * N + col] = __float2bfloat16(acc[4*p]);
        if (row0 < M && col + 1 < N)
            C[row0 * N + col + 1] = __float2bfloat16(acc[4*p + 1]);
        if (row8 < M && col < N)
            C[row8 * N + col] = __float2bfloat16(acc[4*p + 2]);
        if (row8 < M && col + 1 < N)
            C[row8 * N + col + 1] = __float2bfloat16(acc[4*p + 3]);
    }
}

/* =========================================================================
 * Kernel
 * ========================================================================= */
__global__ void __launch_bounds__(THREADS, 1)
matmul_wgmma(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");

    extern __shared__ char smem[];
    /* Double buffer: [stage0_A | stage0_B | stage1_A | stage1_B] */
    __nv_bfloat16* sA[2] = {
        (__nv_bfloat16*)(smem),
        (__nv_bfloat16*)(smem + STAGE_BYTES)
    };
    __nv_bfloat16* sB[2] = {
        (__nv_bfloat16*)(smem + A_BYTES),
        (__nv_bfloat16*)(smem + STAGE_BYTES + A_BYTES)
    };

    int ctaRow = blockIdx.y * TILE_M;
    int ctaCol = blockIdx.x * TILE_N;
    int tid = threadIdx.x;

    /* Two sets of accumulators: rows 0-63 and rows 64-127 */
    float acc0[64], acc1[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) { acc0[i] = 0.0f; acc1[i] = 0.0f; }

    int numK = (K + TILE_K - 1) / TILE_K;

    /* Helper: load tile into SMEM with 128B swizzle + cp.async 16B vectorized */
    /* Each 128B line = 64 BF16. With TILE_K=64, each row = 1 line exactly.
     * cp.async 16B loads: 8 BF16 per load, 8 loads per row.
     * 128 threads, A has 128 rows × 8 loads = 1024 loads total → 8 per thread.
     * Same for B: 128 rows × 8 loads = 1024 loads → 8 per thread. */
    auto load_tile = [&](int buf, int kOff) {
        /* A: row-major M×K, each row = TILE_K BF16 = 128B */
        for (int i = tid; i < TILE_M * (TILE_K / 8); i += THREADS) {
            int row = i / (TILE_K / 8);
            int chunk = i % (TILE_K / 8);  /* 0..7, each chunk = 8 BF16 = 16B */
            int gR = ctaRow + row;
            int gC = kOff + chunk * 8;

            int byte_off = row * TILE_K * 2 + chunk * 16;
            int sw_byte = swizzle_128B(byte_off);

            if (gR < M && gC + 7 < K) {
                __pipeline_memcpy_async(
                    (char*)sA[buf] + sw_byte,
                    &A[gR * K + gC],
                    16);
            } else {
                /* Boundary: scalar fallback */
                for (int j = 0; j < 8; j++) {
                    int elem_byte = sw_byte + j * 2;
                    ((__nv_bfloat16*)((char*)sA[buf]))[elem_byte / 2] =
                        (gR < M && gC + j < K) ? A[gR * K + gC + j] : __float2bfloat16(0.0f);
                }
            }
        }
        /* B: stored N×K transposed. Load B_orig[k][n] into sB[n][k].
         * sB layout: 128 rows (N) × 64 cols (K), each row = 128B.
         * But we read from B_orig which is K×N row-major.
         * Cannot vectorize the transpose — each element goes to a different position.
         * Scalar load with swizzle for now. */
        for (int i = tid; i < TILE_N * TILE_K; i += THREADS) {
            int n = i / TILE_K, k = i % TILE_K;
            int gN = ctaCol + n, gK = kOff + k;
            int byte_off = (n * TILE_K + k) * 2;
            int sw_off = swizzle_128B(byte_off) / 2;
            sB[buf][sw_off] = (gN < N && gK < K) ? B[gK * N + gN] : __float2bfloat16(0.0f);
        }
    };

    load_tile(0, 0);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    for (int kt = 0; kt < numK; kt++) {
        int cur = kt & 1;
        int nxt = 1 - cur;

        if (kt + 1 < numK) {
            load_tile(nxt, (kt + 1) * TILE_K);
            __pipeline_commit();
        }

        /* WGMMA: 2 row groups × 4 k16 steps = 8 WGMMA calls per K-tile */
        /* A top half (rows 0-63) */
        uint64_t da0 = make_desc(sA[cur], TILE_K);
        /* A bottom half (rows 64-127): offset by 64*TILE_K*2 bytes */
        uint64_t da1 = make_desc((__nv_bfloat16*)((char*)sA[cur] + 64 * TILE_K * 2), TILE_K);
        /* B (shared for both row groups) */
        uint64_t db = make_desc(sB[cur], TILE_K);

        int sd = (kt == 0) ? 0 : 1;

        wgmma_fence();
        for (int ks = 0; ks < 4; ks++) {
            uint64_t dak0 = desc_advance(da0, ks * 2);
            uint64_t dak1 = desc_advance(da1, ks * 2);
            uint64_t dbk  = desc_advance(db,  ks * 2);
            wgmma_m64n128k16(acc0, dak0, dbk, (ks == 0) ? sd : 1);
            wgmma_m64n128k16(acc1, dak1, dbk, (ks == 0) ? sd : 1);
        }
        wgmma_commit();
        wgmma_wait0();

        if (kt + 1 < numK) {
            __pipeline_wait_prior(0);
            __syncthreads();
        }
    }

    /* Epilogue: store both row groups */
    store_acc(C, acc0, ctaRow, ctaCol, M, N);        /* rows 0-63 */
    store_acc(C, acc1, ctaRow + 64, ctaCol, M, N);   /* rows 64-127 */
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
    size_t smem = STAGES * STAGE_BYTES;
    cudaFuncSetAttribute(matmul_wgmma, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    matmul_wgmma<<<grid, block, smem, stream>>>(
        inputs[0], inputs[1], outputs[0], dim, dim, dim);
    return 0;
}
