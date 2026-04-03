/*
 * Flash Attention forward pass — BF16, SM90 Hopper tensor-core MMA kernel.
 * Warp-specialized: 1 DMA warp (loads K/V via cp.async) + 4 MMA warps (compute).
 *
 * Target: NVIDIA H100 (SM90, GH100)
 *   - H100 SXM5: 132 SMs, 228 KB SMEM/SM, 989.5 BF16 TFLOPS dense
 *   - H100 PCIe: 114 SMs, 228 KB SMEM/SM, 756 BF16 TFLOPS dense
 *
 * Layout:
 *   Q, K, V, O  : [B, S, H, D]  (row-major __nv_bfloat16)
 *
 * Strategy: work directly on [B, S, H, D] layout using strided
 * global-to-shared copies. Each (batch, head) pair addresses its
 * slice via seq_stride = H * D, eliminating the transpose kernel
 * and its temporary buffer allocations.
 *
 * Constants: BLOCK_Q=128, BLOCK_KV=64, DIM=128, 5 warps (160 threads).
 *
 * Warp specialization:
 *   Warp 0 (tid 0-31):    DMA warp — loads K/V from GMEM->SMEM via cp.async
 *   Warps 1-4 (tid 32-159): MMA warps — compute QK, softmax, PV via mma.sync
 *
 * Named barriers replace __syncthreads() for DMA/MMA synchronization.
 *
 * Register strategy: on-the-fly SMEM->register loading for K and V fragments
 * to stay within 255 registers at BLOCK_Q=128. K_rmem and V_rmem are
 * eliminated; instead, 2-register fragments are loaded and consumed
 * immediately inside the MMA inner loops.
 *
 * MMA/ldmatrix pipelining: in the QK d-loop, the Q fragment for the
 * NEXT d-step is prefetched via ldmatrix while the CURRENT d-step's
 * MMA instructions execute, overlapping SMEM reads with tensor core
 * computation. Same pattern for V fragments in the PV accumulation.
 *
 * SM90 notes:
 *   - 228 KB shared memory per SM allows up to 2 blocks/SM at 80 KB each
 *   - cp.async with commit/wait groups for efficient GMEM->SMEM transfers
 *   - mma.sync.aligned.m16n8k16 is the workhorse MMA instruction
 *   - Future optimization: migrate to wgmma (warp group MMA) and TMA
 *     for SM90-native performance (requires architectural rewrite)
 *
 * kernel_run contract:
 *   extern "C" int kernel_run(__nv_bfloat16** inputs, int num_inputs,
 *                             __nv_bfloat16** outputs, int num_outputs,
 *                             int n, cudaStream_t stream);
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cfloat>

/* ======================================================================
 *  Helper constants & device functions
 * ====================================================================== */

static constexpr int WARP_SIZE = 32;

__device__ __host__ constexpr
int cdiv(int a, int b) { return (a + b - 1) / b; }

/* --- Swizzle --------------------------------------------------------- */

template <int STRIDE>
__device__
uint32_t swizzle(uint32_t index) {
    if constexpr (STRIDE == 16)
        return index;
    uint32_t row_idx = (index / STRIDE) % 8;
    uint32_t bits_to_xor = row_idx / max(64 / STRIDE, 1);
    return index ^ (bits_to_xor << 4);
}

/* --- cp.async global -> shared (swizzled) ----------------------------- */

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared_swizzle(uint32_t dst, const nv_bfloat16 *src,
                              int src_stride, int tid) {
    constexpr int num_elems = 16 / sizeof(nv_bfloat16);          /* 8 */
    constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

    for (int iter = 0; iter < num_iters; iter++) {
        const int idx = (iter * TB_SIZE + tid) * num_elems;
        const int row = idx / WIDTH;
        const int col = idx % WIDTH;

        const uint32_t dst_addr =
            swizzle<WIDTH * sizeof(nv_bfloat16)>(
                dst + (row * WIDTH + col) * sizeof(nv_bfloat16));
        const nv_bfloat16 *src_addr = src + (row * src_stride + col);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :: "r"(dst_addr), "l"(src_addr));
    }
}

/* --- ldmatrix helpers ------------------------------------------------- */

__device__ inline
void ldmatrix_x2(uint32_t regs[2], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(regs[0]), "=r"(regs[1])
        : "r"(addr));
}

__device__ inline
void ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "r"(addr));
}

__device__ inline
void ldmatrix_x2_trans(uint32_t regs[2], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(regs[0]), "=r"(regs[1])
        : "r"(addr));
}

__device__ inline
void ldmatrix_x4_trans(uint32_t regs[4], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "r"(addr));
}

/* --- mma.m16n8k16 ---------------------------------------------------- */

__device__ inline
void mma_m16n8k16(uint32_t A[4], uint32_t B[2], float D[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]),
          "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

/* ======================================================================
 *  Named barrier helpers for warp specialization
 * ====================================================================== */

__device__ inline void bar_sync(int barrier_id, int num_threads) {
    asm volatile("bar.sync %0, %1;" :: "r"(barrier_id), "r"(num_threads));
}

__device__ inline void bar_arrive(int barrier_id, int num_threads) {
    asm volatile("bar.arrive %0, %1;" :: "r"(barrier_id), "r"(num_threads));
}

/* Barrier IDs */
static constexpr int BAR_K_FULL  = 1;  /* DMA signals K ready       */
static constexpr int BAR_K_EMPTY = 2;  /* MMA signals K consumed    */
static constexpr int BAR_V_FULL  = 3;  /* DMA signals V ready       */
static constexpr int BAR_V_EMPTY = 4;  /* MMA signals V consumed    */
static constexpr int BAR_THREADS = 160; /* all 5 warps participate   */

/* ======================================================================
 *  DMA warp function — loads K/V tiles from GMEM to SMEM
 *
 *  __noinline__ gives the compiler a separate register allocation
 *  scope, keeping DMA regs (~40) isolated from MMA regs (~220+).
 * ====================================================================== */

template <int BLOCK_KV, int DIM>
__device__ __noinline__
void dma_warp_fn(
    const nv_bfloat16 *K_base_ptr,
    const nv_bfloat16 *V_base_ptr,
    int seq_stride,
    int max_kv_iter,
    uint32_t K_smem,
    uint32_t V_smem,
    int tid)  /* tid 0-31 within the DMA warp */
{
    constexpr int DMA_THREADS = 32;

    const nv_bfloat16 *K_ptr = K_base_ptr;
    const nv_bfloat16 *V_ptr = V_base_ptr;

    for (int kv_id = 0; kv_id < max_kv_iter; kv_id++) {
        /* Wait until MMA warps have consumed the previous K in this buffer slot. */
        if (kv_id > 0) bar_sync(BAR_K_EMPTY, BAR_THREADS);

        /* Load K[kv_id] into double-buffered K_smem slot */
        const uint32_t K_dst = K_smem +
            (kv_id % 2) * (BLOCK_KV * DIM * (int)sizeof(nv_bfloat16));
        global_to_shared_swizzle<BLOCK_KV, DIM, DMA_THREADS>(
            K_dst, K_ptr, seq_stride, tid);
        K_ptr += BLOCK_KV * seq_stride;

        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");

        /* Signal K is ready */
        bar_arrive(BAR_K_FULL, BAR_THREADS);

        /* Wait until MMA warps have consumed the previous V. */
        if (kv_id > 0) bar_sync(BAR_V_EMPTY, BAR_THREADS);

        /* Load V[kv_id] into single-buffered V_smem */
        global_to_shared_swizzle<BLOCK_KV, DIM, DMA_THREADS>(
            V_smem, V_ptr, seq_stride, tid);
        V_ptr += BLOCK_KV * seq_stride;

        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");

        /* Signal V is ready */
        bar_arrive(BAR_V_FULL, BAR_THREADS);
    }
}

/* ======================================================================
 *  MMA warp function — computes QK, softmax, PV for one MMA warp
 *
 *  __noinline__ for register isolation from DMA warp.
 * ====================================================================== */

template <int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_MMA_WARPS>
__device__ __noinline__
void mma_warp_fn(
    nv_bfloat16 *O_base,
    int seq_stride,
    int max_kv_iter,
    int q_block_id,
    int is_causal,
    uint32_t Q_smem,
    uint32_t K_smem,
    uint32_t V_smem,
    int mma_warp_id,   /* 0-3 */
    int lane_id)
{
    constexpr int WARP_Q = BLOCK_Q / NUM_MMA_WARPS;  /* 32 */
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;

    /* O accumulator in registers */
    float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};

    /* Pre-compute swizzled SMEM base addresses for ldmatrix */
    uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
    {
        /* A tile (Q) — ldmatrix x4 for m16n8k16 A operand */
        const int row_off = mma_warp_id * WARP_Q + (lane_id % 16);
        const int col_off = lane_id / 16 * 8;
        Q_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(
            Q_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
    {
        /* B tile (K — non-transposed ldmatrix) */
        const int row_off = lane_id % 8;
        const int col_off = lane_id / 8 * 8;
        K_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(
            K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
    {
        /* B tile trans (V — transposed ldmatrix) */
        const int row_off = lane_id % 16;
        const int col_off = lane_id / 16 * 8;
        V_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(
            V_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }

    const float softmax_scale = rsqrtf(static_cast<float>(DIM));
    const float softmax_scale_log2 = softmax_scale * 1.4426950408889634f;

    float rowmax[WARP_Q / MMA_M][2];
    float rowsumexp[WARP_Q / MMA_M][2] = {};
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
        rowmax[mma_id_q][0] = -FLT_MAX;
        rowmax[mma_id_q][1] = -FLT_MAX;
    }

    for (int kv_id = 0; kv_id < max_kv_iter; kv_id++) {

        /* Wait for DMA to signal K is ready */
        bar_sync(BAR_K_FULL, BAR_THREADS);

        const uint32_t K_cur = K_smem_thread +
            (kv_id % 2) * (BLOCK_KV * DIM * (int)sizeof(nv_bfloat16));

        uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];

        for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
            /* ---- QK: S = Q[mma_id_q] @ K^T ---- */
            float S_local[BLOCK_KV / MMA_N][4] = {};

            /* Prefetch Q for d=0 */
            uint32_t Q_cur[4];
            {
                uint32_t qaddr = Q_smem_thread;
                qaddr += mma_id_q * MMA_M * DIM * sizeof(nv_bfloat16);
                ldmatrix_x4(Q_cur, qaddr);
            }

            #pragma unroll
            for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
                /* Prefetch Q for next d-step (if not last) */
                uint32_t Q_next[4];
                if (mma_id_d + 1 < DIM / MMA_K) {
                    uint32_t qaddr = Q_smem_thread;
                    qaddr += mma_id_q * MMA_M * DIM * sizeof(nv_bfloat16);
                    qaddr ^= (mma_id_d + 1) * MMA_K * sizeof(nv_bfloat16);
                    ldmatrix_x4(Q_next, qaddr);
                }
                #pragma unroll
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                    uint32_t K_frag[2];
                    {
                        uint32_t kaddr = K_cur;
                        kaddr += mma_id_kv * MMA_N * DIM * sizeof(nv_bfloat16);
                        kaddr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
                        ldmatrix_x2(K_frag, kaddr);
                    }
                    mma_m16n8k16(Q_cur, K_frag, S_local[mma_id_kv]);
                }
                /* Swap current and next */
                if (mma_id_d + 1 < DIM / MMA_K) {
                    Q_cur[0] = Q_next[0]; Q_cur[1] = Q_next[1];
                    Q_cur[2] = Q_next[2]; Q_cur[3] = Q_next[3];
                }
            }

            /* Signal K_EMPTY early after last mma_id_q's QK loop */
            if (mma_id_q == WARP_Q / MMA_M - 1) {
                bar_arrive(BAR_K_EMPTY, BAR_THREADS);
            }

            /* ---- Softmax scale ---- */
            #pragma unroll
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
                #pragma unroll
                for (int reg_id = 0; reg_id < 4; reg_id++)
                    S_local[mma_id_kv][reg_id] *= softmax_scale_log2;

            /* ---- Causal mask ---- */
            if (is_causal) {
                const int q_start = q_block_id * BLOCK_Q;
                const int kv_start = kv_id * BLOCK_KV;

                const int q_first = q_start + mma_warp_id * WARP_Q +
                                    mma_id_q * MMA_M;
                const int kv_last = kv_start + BLOCK_KV - 1;

                if (kv_last > q_first) {
                    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                        int q_pos_0 = q_first + (lane_id / 4);
                        int q_pos_1 = q_pos_0 + 8;
                        int kv_pos_0 = kv_start +
                                       mma_id_kv * MMA_N + (lane_id % 4) * 2;
                        int kv_pos_1 = kv_pos_0 + 1;

                        if (kv_pos_0 > q_pos_0)
                            S_local[mma_id_kv][0] = -FLT_MAX;
                        if (kv_pos_1 > q_pos_0)
                            S_local[mma_id_kv][1] = -FLT_MAX;
                        if (kv_pos_0 > q_pos_1)
                            S_local[mma_id_kv][2] = -FLT_MAX;
                        if (kv_pos_1 > q_pos_1)
                            S_local[mma_id_kv][3] = -FLT_MAX;
                    }
                }
            }

            /* ---- Online softmax + rescaling ---- */
            float this_rowmax[2];
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                float *regs = S_local[mma_id_kv];
                if (mma_id_kv == 0) {
                    this_rowmax[0] = max(regs[0], regs[1]);
                    this_rowmax[1] = max(regs[2], regs[3]);
                } else {
                    this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));
                    this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));
                }
            }

            /* Butterfly reduction within 4 threads */
            this_rowmax[0] = max(this_rowmax[0],
                __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 1));
            this_rowmax[0] = max(this_rowmax[0],
                __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 2));
            this_rowmax[1] = max(this_rowmax[1],
                __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 1));
            this_rowmax[1] = max(this_rowmax[1],
                __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 2));

            this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
            this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);

            /* Rescale previous O accumulator */
            float rescale[2];
            rescale[0] = exp2f(rowmax[mma_id_q][0] - this_rowmax[0]);
            rescale[1] = exp2f(rowmax[mma_id_q][1] - this_rowmax[1]);
            #pragma unroll
            for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
                O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
                O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
                O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
                O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
            }

            rowmax[mma_id_q][0] = this_rowmax[0];
            rowmax[mma_id_q][1] = this_rowmax[1];

            /* Row sum-exp + pack S -> P */
            float this_rowsumexp[2];
            #pragma unroll
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                float *regs = S_local[mma_id_kv];
                regs[0] = exp2f(regs[0] - rowmax[mma_id_q][0]);
                regs[1] = exp2f(regs[1] - rowmax[mma_id_q][0]);
                regs[2] = exp2f(regs[2] - rowmax[mma_id_q][1]);
                regs[3] = exp2f(regs[3] - rowmax[mma_id_q][1]);

                if (mma_id_kv == 0) {
                    this_rowsumexp[0] = regs[0] + regs[1];
                    this_rowsumexp[1] = regs[2] + regs[3];
                } else {
                    this_rowsumexp[0] += regs[0] + regs[1];
                    this_rowsumexp[1] += regs[2] + regs[3];
                }

                /* Pack to P registers (m16n8 -> m16k16 layout) */
                nv_bfloat162 *this_P =
                    reinterpret_cast<nv_bfloat162 *>(
                        P_rmem[mma_id_q][mma_id_kv / 2]);
                this_P[(mma_id_kv % 2) * 2] =
                    __float22bfloat162_rn({regs[0], regs[1]});
                this_P[(mma_id_kv % 2) * 2 + 1] =
                    __float22bfloat162_rn({regs[2], regs[3]});
            }

            /* Butterfly reduction for sumexp */
            this_rowsumexp[0] +=
                __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 1);
            this_rowsumexp[0] +=
                __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 2);
            this_rowsumexp[1] +=
                __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 1);
            this_rowsumexp[1] +=
                __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 2);

            rowsumexp[mma_id_q][0] =
                rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
            rowsumexp[mma_id_q][1] =
                rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];

        } /* end per-mma_id_q QK+softmax */

        /* Wait for DMA to signal V is ready */
        bar_sync(BAR_V_FULL, BAR_THREADS);

        /* O += P @ V  [BLOCK_Q, DIM] */
        {
            /* Prefetch V for the very first step (kv=0, d=0) */
            uint32_t V_cur[2];
            {
                uint32_t addr = V_smem_thread;
                ldmatrix_x2_trans(V_cur, addr);
            }

            #pragma unroll
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++) {
                #pragma unroll
                for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
                    /* Prefetch V for next step */
                    uint32_t V_next[2];
                    const bool has_next_d = (mma_id_d + 1 < DIM / MMA_N);
                    const bool has_next_kv = (mma_id_kv + 1 < BLOCK_KV / MMA_K);
                    if (has_next_d) {
                        uint32_t addr = V_smem_thread;
                        addr += mma_id_kv * MMA_K * DIM * sizeof(nv_bfloat16);
                        addr ^= (mma_id_d + 1) * MMA_N * sizeof(nv_bfloat16);
                        ldmatrix_x2_trans(V_next, addr);
                    } else if (has_next_kv) {
                        uint32_t addr = V_smem_thread;
                        addr += (mma_id_kv + 1) * MMA_K * DIM * sizeof(nv_bfloat16);
                        ldmatrix_x2_trans(V_next, addr);
                    }
                    #pragma unroll
                    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
                        mma_m16n8k16(P_rmem[mma_id_q][mma_id_kv],
                                     V_cur,
                                     O_rmem[mma_id_q][mma_id_d]);
                    /* Swap current and next */
                    if (has_next_d || has_next_kv) {
                        V_cur[0] = V_next[0]; V_cur[1] = V_next[1];
                    }
                }
            }
        }

        /* Signal DMA that V buffer is free for next load */
        bar_arrive(BAR_V_EMPTY, BAR_THREADS);

    } /* end kv_id loop */

    /* ---- Write O to global memory (divide by softmax denominator) ---- */
    #pragma unroll
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        #pragma unroll
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
            const int row = mma_warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

            float *regs = O_rmem[mma_id_q][mma_id_d];
            regs[0] /= rowsumexp[mma_id_q][0];
            regs[1] /= rowsumexp[mma_id_q][0];
            regs[2] /= rowsumexp[mma_id_q][1];
            regs[3] /= rowsumexp[mma_id_q][1];

            reinterpret_cast<nv_bfloat162 *>(O_base + (row + 0) * seq_stride + col)[0] =
                __float22bfloat162_rn({regs[0], regs[1]});
            reinterpret_cast<nv_bfloat162 *>(O_base + (row + 8) * seq_stride + col)[0] =
                __float22bfloat162_rn({regs[2], regs[3]});
        }
}

/* ======================================================================
 *  Warp-specialized Flash Attention kernel
 *
 *  Each thread-block handles one BLOCK_Q chunk of one (batch, head).
 *  5 warps: warp 0 = DMA, warps 1-4 = MMA.
 * ====================================================================== */

template<int BLOCK_Q, int BLOCK_KV, int DIM>
__launch_bounds__(160, 1)  /* 1 block/SM for full 255 regs */
__global__
void flash_attention_kernel_ws(
    const nv_bfloat16 *Q,   /* [B, S, H, D] */
    const nv_bfloat16 *K,   /* [B, S, H, D] */
    const nv_bfloat16 *V,   /* [B, S, H, D] */
    nv_bfloat16 *O,         /* [B, S, H, D] */
    int B, int S, int H,
    int len_q,
    int len_kv,
    int is_causal)
{
    constexpr int NUM_MMA_WARPS = 4;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int num_q_blocks = cdiv(len_q, BLOCK_Q);
    const int bs_id       = bid / num_q_blocks;
    const int q_block_id  = bid % num_q_blocks;

    const int batch_id = bs_id / H;
    const int head_id  = bs_id % H;

    /* seq_stride: elements between consecutive sequence positions */
    const int seq_stride = H * DIM;

    /* Base pointers */
    const nv_bfloat16 *Q_base = Q + batch_id * S * seq_stride + head_id * DIM
                                   + q_block_id * BLOCK_Q * seq_stride;
    const nv_bfloat16 *K_base_ptr = K + batch_id * S * seq_stride + head_id * DIM;
    const nv_bfloat16 *V_base_ptr = V + batch_id * S * seq_stride + head_id * DIM;
    nv_bfloat16       *O_base = O + batch_id * S * seq_stride + head_id * DIM
                                  + q_block_id * BLOCK_Q * seq_stride;

    /* Shared memory layout:
     *   Q region:  [BLOCK_Q, DIM] = 32KB (persistent)
     *   K region:  [2, BLOCK_KV, DIM] = 32KB (double-buffered)
     *   V region:  [BLOCK_KV, DIM] = 16KB (single-buffered)
     *   Total: 32KB + 32KB + 16KB = 80KB (within 228KB SM90 limit)
     */
    extern __shared__ nv_bfloat16 smem[];
    const uint32_t smem_base = __cvta_generic_to_shared(smem);
    const uint32_t Q_smem  = smem_base;
    const uint32_t KV_base = smem_base + BLOCK_Q * DIM * sizeof(nv_bfloat16);
    const uint32_t K_smem  = KV_base;
    const uint32_t V_smem  = KV_base + 2 * BLOCK_KV * DIM * sizeof(nv_bfloat16);

    /* ---- Load Q [BLOCK_Q, DIM] from global -> shared (ALL threads) ---- */
    if (tid < 128) {
        global_to_shared_swizzle<BLOCK_Q, DIM, 128>(Q_smem, Q_base, seq_stride, tid);
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    const int num_kv_iter = cdiv(len_kv, BLOCK_KV);

    /* Causal early-exit bound */
    const int max_kv_iter = is_causal
        ? min(num_kv_iter, cdiv((q_block_id + 1) * BLOCK_Q, BLOCK_KV))
        : num_kv_iter;

    /* ---- Warp split ---- */
    if (warp_id == 0) {
        dma_warp_fn<BLOCK_KV, DIM>(
            K_base_ptr, V_base_ptr,
            seq_stride, max_kv_iter,
            K_smem, V_smem,
            tid);
    } else {
        const int mma_warp_id = warp_id - 1;
        mma_warp_fn<BLOCK_Q, BLOCK_KV, DIM, NUM_MMA_WARPS>(
            O_base, seq_stride,
            max_kv_iter, q_block_id, is_causal,
            Q_smem, K_smem, V_smem,
            mma_warp_id, lane_id);
    }
}

/* ======================================================================
 *  Simple integer parser helpers
 * ====================================================================== */

static int parse_int_env(const char *name, int fallback) {
    const char *v = getenv(name);
    if (v && *v) return atoi(v);
    return fallback;
}

static bool parse_bool_env(const char *name, bool fallback) {
    const char *v = getenv(name);
    if (!v) return fallback;
    return (strcmp(v, "true") == 0 || strcmp(v, "1") == 0 ||
            strcmp(v, "True") == 0 || strcmp(v, "TRUE") == 0);
}

static int json_int(const char *json, const char *key, int fallback) {
    if (!json) return fallback;
    const char *p = strstr(json, key);
    if (!p) return fallback;
    p += strlen(key);
    while (*p && (*p == '"' || *p == ':' || *p == ' ' || *p == '\t')) ++p;
    if (*p < '0' || *p > '9') return fallback;
    int val = 0;
    while (*p >= '0' && *p <= '9') { val = val * 10 + (*p - '0'); ++p; }
    return val;
}

static bool json_bool(const char *json, const char *key, bool fallback) {
    if (!json) return fallback;
    const char *p = strstr(json, key);
    if (!p) return fallback;
    p += strlen(key);
    while (*p && (*p == '"' || *p == ':' || *p == ' ' || *p == '\t')) ++p;
    if (strncmp(p, "true", 4) == 0) return true;
    if (strncmp(p, "false", 5) == 0) return false;
    return fallback;
}

/* ======================================================================
 *  kernel_run — entry point called by the cuda_exec harness
 *
 *   inputs[0] = Q [B, S, H, D]
 *   inputs[1] = K [B, S, H, D]
 *   inputs[2] = V [B, S, H, D]
 *   outputs[0] = O [B, S, H, D]
 *   n = total elements per buffer = B * S * H * D
 * ====================================================================== */

extern "C" int kernel_run(
    __nv_bfloat16 **inputs,  int num_inputs,
    __nv_bfloat16 **outputs, int num_outputs,
    int n, cudaStream_t stream)
{
    const char *config_json = getenv("CUDA_EXEC_CONFIG_JSON");

    int B = parse_int_env("CUDA_EXEC_PARAM_BATCH_SIZE", 0);
    int S = parse_int_env("CUDA_EXEC_PARAM_SEQ_LEN",    0);
    int H = parse_int_env("CUDA_EXEC_PARAM_NUM_HEADS",  0);
    int D = parse_int_env("CUDA_EXEC_PARAM_HEAD_DIM",   0);

    if (B == 0) B = json_int(config_json, "batch_size", 0);
    if (S == 0) S = json_int(config_json, "seq_len",    0);
    if (H == 0) H = json_int(config_json, "num_heads",  0);
    if (D == 0) D = json_int(config_json, "head_dim",   0);

    if (D == 0) D = 128;
    if (H == 0) H = 16;
    if (S == 0 && B == 0 && n > 0) {
        int total_tokens = n / (H * D);
        B = 1;
        S = total_tokens / B;
    }
    if (B == 0 || S == 0) return -1;

    bool causal = parse_bool_env("CUDA_EXEC_PARAM_CAUSAL", false);
    if (!causal && config_json) {
        causal = json_bool(config_json, "causal", false);
    }

    if (D != 128) return -2;    /* kernel only supports DIM=128 */

    {
        const int BLOCK_Q   = 128;
        const int BLOCK_KV  = 64;
        const int DIM_CONST = 128;
        const int TB_SIZE   = 160;  /* 5 warps: 1 DMA + 4 MMA */

        int effective_bs = B * H;
        int num_blocks = effective_bs * cdiv(S, BLOCK_Q);

        /* SMEM budget: Q (persistent) + K (double-buffered) + V (single-buffered)
         *   Q:  128*128*2 = 32768
         *   K:  2*64*128*2 = 32768
         *   V:  64*128*2 = 16384
         *   Total: 81920 (80KB, within 228KB SM90 limit)
         */
        int smem_q  = BLOCK_Q * DIM_CONST * (int)sizeof(nv_bfloat16);
        int smem_k  = 2 * BLOCK_KV * DIM_CONST * (int)sizeof(nv_bfloat16);
        int smem_v  = BLOCK_KV * DIM_CONST * (int)sizeof(nv_bfloat16);
        int smem_size = smem_q + smem_k + smem_v;

        auto kernel = flash_attention_kernel_ws<BLOCK_Q, BLOCK_KV, DIM_CONST>;
        if (smem_size > 48000)
            cudaFuncSetAttribute(kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        kernel<<<num_blocks, TB_SIZE, smem_size, stream>>>(
            inputs[0], inputs[1], inputs[2], outputs[0],
            B, S, H, S, S, causal ? 1 : 0);
    }

    return 0;
}
