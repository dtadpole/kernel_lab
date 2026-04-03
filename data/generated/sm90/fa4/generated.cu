/*
 * Flash Attention forward pass — BF16, SM90 WGMMA kernel (Phase 3).
 * Unified 128 threads (1 warp group) + double-buffered K pipeline.
 *
 * Architecture: 128 threads, all threads cooperate on both cp.async
 * loads and WGMMA compute. No warp specialization.
 *
 * Pipeline: K[i+1] loads overlap with QK[i] compute + softmax.
 * V reuses the freed K buffer after QK consumes it.
 *
 * Constants: BLOCK_Q=64, BLOCK_KV=64, DIM=128, 128 threads.
 * SMEM: Q(16KB) + K[0](16KB) + K[1](16KB) + P(8KB) = 56KB total.
 *       V reuses K[cur_stage] after QK is done.
 *
 * Target: NVIDIA H100 (SM90a, GH100). Compile with -arch=sm_90a.
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

__device__ __host__ constexpr
int cdiv(int a, int b) { return (a + b - 1) / b; }

/* --- Fast math: inline PTX to avoid FCHK + slowpath overhead --------- */

__device__ __forceinline__
float fast_exp2f(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__
float fast_rcp(float x) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

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
    constexpr int num_elems = 16 / sizeof(nv_bfloat16);
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

/* --- WGMMA synchronization ------------------------------------------ */

__device__ __forceinline__
void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__
void wgmma_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__
void wgmma_wait_group() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

/* --- WGMMA descriptor construction and advancement ------------------- */

__device__ __forceinline__
uint64_t make_wgmma_desc(uint32_t smem_addr, int stride_bytes) {
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_addr >> 4) & 0x3FFF);            // [0:14)  start_address
    desc |= (uint64_t)(1) << 16;                               // [16:30) leading_byte_off = 1
    desc |= (uint64_t)(((stride_bytes >> 4) & 0x3FFF)) << 32;  // [32:46) stride_byte_off
    desc |= (uint64_t)(1) << 62;                                // [62:64) SWIZZLE_128B
    return desc;
}

__device__ __forceinline__
uint64_t gmma_desc_advance(uint64_t desc, int offset_16B) {
    uint32_t lo = (uint32_t)desc + (uint32_t)offset_16B;
    uint32_t hi = (uint32_t)(desc >> 32);
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

/* --- WGMMA m64n64k16 (QK GEMM) -------------------------------------- */

__device__ __forceinline__
void wgmma_m64n64k16_f32_bf16(float acc[32], uint64_t desc_a, uint64_t desc_b,
                               int scale_D) {
    asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, %34, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
    "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
    " %8,  %9,  %10, %11, %12, %13, %14, %15, "
    " %16, %17, %18, %19, %20, %21, %22, %23, "
    " %24, %25, %26, %27, %28, %29, %30, %31},"
    " %32, %33, p, 1, 1, 0, 0;\n"
    "}\n"
    : "+f"(acc[0]),  "+f"(acc[1]),  "+f"(acc[2]),  "+f"(acc[3]),
      "+f"(acc[4]),  "+f"(acc[5]),  "+f"(acc[6]),  "+f"(acc[7]),
      "+f"(acc[8]),  "+f"(acc[9]),  "+f"(acc[10]), "+f"(acc[11]),
      "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]),
      "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]),
      "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]),
      "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]),
      "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31])
    : "l"(desc_a), "l"(desc_b), "r"(scale_D));
}

/* --- WGMMA m64n128k16 (PV GEMM) ------------------------------------- */

__device__ __forceinline__
void wgmma_m64n128k16_f32_bf16(float acc[64], uint64_t desc_a, uint64_t desc_b,
                                int scale_D) {
    asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, %66, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
    "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
    " %8,  %9,  %10, %11, %12, %13, %14, %15, "
    " %16, %17, %18, %19, %20, %21, %22, %23, "
    " %24, %25, %26, %27, %28, %29, %30, %31, "
    " %32, %33, %34, %35, %36, %37, %38, %39, "
    " %40, %41, %42, %43, %44, %45, %46, %47, "
    " %48, %49, %50, %51, %52, %53, %54, %55, "
    " %56, %57, %58, %59, %60, %61, %62, %63},"
    " %64, %65, p, 1, 1, 0, 1;\n"
    "}\n"
    : "+f"(acc[0]),  "+f"(acc[1]),  "+f"(acc[2]),  "+f"(acc[3]),
      "+f"(acc[4]),  "+f"(acc[5]),  "+f"(acc[6]),  "+f"(acc[7]),
      "+f"(acc[8]),  "+f"(acc[9]),  "+f"(acc[10]), "+f"(acc[11]),
      "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]),
      "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]),
      "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]),
      "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]),
      "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31]),
      "+f"(acc[32]), "+f"(acc[33]), "+f"(acc[34]), "+f"(acc[35]),
      "+f"(acc[36]), "+f"(acc[37]), "+f"(acc[38]), "+f"(acc[39]),
      "+f"(acc[40]), "+f"(acc[41]), "+f"(acc[42]), "+f"(acc[43]),
      "+f"(acc[44]), "+f"(acc[45]), "+f"(acc[46]), "+f"(acc[47]),
      "+f"(acc[48]), "+f"(acc[49]), "+f"(acc[50]), "+f"(acc[51]),
      "+f"(acc[52]), "+f"(acc[53]), "+f"(acc[54]), "+f"(acc[55]),
      "+f"(acc[56]), "+f"(acc[57]), "+f"(acc[58]), "+f"(acc[59]),
      "+f"(acc[60]), "+f"(acc[61]), "+f"(acc[62]), "+f"(acc[63])
    : "l"(desc_a), "l"(desc_b), "r"(scale_D));
}

/* --- fence_view_async_shared ----------------------------------------- */

__device__ __forceinline__
void fence_view_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

/* ======================================================================
 *  WGMMA Flash Attention kernel — Phase 3 (unified + pipeline)
 *
 *  128 threads = 1 warp group (warps 0-3).
 *  Double-buffered K: loads K[i+1] during QK[i] + softmax.
 *  V reuses K[cur_stage] buffer after QK is done.
 * ====================================================================== */

template<int BLOCK_Q, int BLOCK_KV, int DIM>
__launch_bounds__(128, 1)   /* 128 threads, 1 block/SM */
__global__
void flash_attention_wgmma(
    const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    nv_bfloat16 *O,
    int B, int S, int H,
    int len_q, int len_kv,
    int is_causal)
{
    /* Constants */
    constexpr int TB_SIZE = 128;

    /* SMEM layout:
     *   Q:    BLOCK_Q * DIM * 2B    = 16KB  (persistent)
     *   K[0]: BLOCK_KV * DIM * 2B   = 16KB  (stage 0)
     *   K[1]: BLOCK_KV * DIM * 2B   = 16KB  (stage 1)
     *   P:    BLOCK_Q * BLOCK_KV * 2B = 8KB  (softmax output for PV)
     *   Total: 56KB
     *   V reuses K[cur_stage] after QK consumes it. */
    extern __shared__ nv_bfloat16 smem[];
    const uint32_t smem_base = __cvta_generic_to_shared(smem);
    const uint32_t Q_smem   = smem_base;
    const uint32_t K_smem_0 = Q_smem + BLOCK_Q * DIM * sizeof(nv_bfloat16);
    const uint32_t K_smem_1 = K_smem_0 + BLOCK_KV * DIM * sizeof(nv_bfloat16);
    const uint32_t P_smem   = K_smem_1 + BLOCK_KV * DIM * sizeof(nv_bfloat16);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;         /* 0-3 */
    const int lane_id = tid % 32;

    /* Block -> (batch, head, q_block) mapping */
    const int bid = blockIdx.x;
    const int num_q_blocks = cdiv(len_q, BLOCK_Q);
    const int bs_id       = bid / num_q_blocks;
    const int q_block_id  = bid % num_q_blocks;
    const int batch_id    = bs_id / H;
    const int head_id     = bs_id % H;
    const int seq_stride  = H * DIM;

    const nv_bfloat16 *Q_base = Q + batch_id * S * seq_stride + head_id * DIM
                                   + q_block_id * BLOCK_Q * seq_stride;
    const nv_bfloat16 *K_base = K + batch_id * S * seq_stride + head_id * DIM;
    const nv_bfloat16 *V_base = V + batch_id * S * seq_stride + head_id * DIM;
    nv_bfloat16       *O_base = O + batch_id * S * seq_stride + head_id * DIM
                                  + q_block_id * BLOCK_Q * seq_stride;

    /* KV iteration bounds */
    const int num_kv_iter = cdiv(len_kv, BLOCK_KV);
    const int max_kv_iter = is_causal
        ? min(num_kv_iter, cdiv((q_block_id + 1) * BLOCK_Q, BLOCK_KV))
        : num_kv_iter;

    /* ---- Load Q to shared memory (all 128 threads) ---- */
    global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE>(
        Q_smem, Q_base, seq_stride, tid);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    /* WGMMA descriptor stride constants */
    constexpr int QKV_stride_bytes = 8 * DIM * sizeof(nv_bfloat16);      /* 2048 */
    constexpr int P_stride_bytes   = 8 * BLOCK_KV * sizeof(nv_bfloat16); /* 1024 */

    const float softmax_scale_log2 =
        rsqrtf(static_cast<float>(DIM)) * 1.4426950408889634f;

    /* Initialize accumulators */
    float O_acc[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) O_acc[i] = 0.0f;

    float rowmax[2] = {-FLT_MAX, -FLT_MAX};
    float rowsumexp[2] = {0.0f, 0.0f};

    /* Precompute base descriptors for Q and P (invariant across iterations) */
    const uint64_t desc_q_base = make_wgmma_desc(Q_smem, QKV_stride_bytes);
    const uint64_t desc_p_base = make_wgmma_desc(P_smem, P_stride_bytes);

    /* Precompute P SMEM store addresses per thread.
     * Each thread stores 8 packed b32 values per half (16 total).
     * p_row: determined by warp_id, half, lane_id.
     * p_col_base: determined by p4, lane_id.
     * With 128B swizzle on BLOCK_KV=64 width (128B stride),
     * addresses are just base + offset since stride == 128B = 1 swizzle line. */
    uint32_t p_addrs[2][8];  /* [half][p4] */
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        const int p_row = warp_id * 16 + half * 8 + (lane_id / 4);
        #pragma unroll
        for (int p4 = 0; p4 < 8; p4++) {
            const int p_col_base = p4 * 8 + (lane_id % 4) * 2;
            p_addrs[half][p4] = swizzle<BLOCK_KV * (int)sizeof(nv_bfloat16)>(
                P_smem + (p_row * BLOCK_KV + p_col_base) * sizeof(nv_bfloat16));
        }
    }

    /* ---- Prelude: load K[0] into stage 0 ---- */
    if (max_kv_iter > 0) {
        global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(
            K_smem_0, K_base, seq_stride, tid);
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncthreads();
    }

    /* ---- Main KV loop ---- */
    for (int kv_id = 0; kv_id < max_kv_iter; kv_id++) {
        const int cur_stage  = kv_id & 1;
        const uint32_t cur_K_smem  = cur_stage ? K_smem_1 : K_smem_0;
        const uint32_t next_K_smem = cur_stage ? K_smem_0 : K_smem_1;

        /* == Step 1: Start loading K[kv_id+1] (non-blocking) == */
        if (kv_id + 1 < max_kv_iter) {
            const nv_bfloat16 *K_next = K_base + (kv_id + 1) * BLOCK_KV * seq_stride;
            global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(
                next_K_smem, K_next, seq_stride, tid);
            asm volatile("cp.async.commit_group;");
        }

        /* == Step 2: QK GEMM using K_smem[cur_stage] == */
        float S_acc[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) S_acc[i] = 0.0f;

        const uint64_t desc_k_base = make_wgmma_desc(cur_K_smem, QKV_stride_bytes);

        wgmma_fence();

        #pragma unroll
        for (int ks = 0; ks < DIM / 16; ks++) {
            uint64_t desc_q = gmma_desc_advance(desc_q_base, ks * 2);
            uint64_t desc_k = gmma_desc_advance(desc_k_base, ks * 2);
            wgmma_m64n64k16_f32_bf16(S_acc, desc_q, desc_k,
                                      (ks == 0) ? 0 : 1);
        }

        wgmma_commit_group();
        wgmma_wait_group<0>();
        wgmma_fence();  /* fence before reading accumulators */

        /* == Step 3: Softmax on S_acc -> P_smem == */
        /* (runs while cp.async K[kv_id+1] may still be in flight) */
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            float row_vals[16];
            #pragma unroll
            for (int p4 = 0; p4 < 8; p4++) {
                row_vals[p4 * 2 + 0] = S_acc[(p4 << 2) | (half << 1) | 0];
                row_vals[p4 * 2 + 1] = S_acc[(p4 << 2) | (half << 1) | 1];
            }

            /* Scale */
            #pragma unroll
            for (int c = 0; c < 16; c++)
                row_vals[c] *= softmax_scale_log2;

            /* Causal mask */
            if (is_causal) {
                const int row = warp_id * 16 + half * 8 + (lane_id / 4);
                const int q_pos = q_block_id * BLOCK_Q + row;
                #pragma unroll
                for (int p4 = 0; p4 < 8; p4++) {
                    #pragma unroll
                    for (int s2 = 0; s2 < 2; s2++) {
                        const int col = p4 * 8 + s2 + (lane_id % 4) * 2;
                        const int kv_pos = kv_id * BLOCK_KV + col;
                        if (kv_pos > q_pos)
                            row_vals[p4 * 2 + s2] = -FLT_MAX;
                    }
                }
            }

            /* Local max across 16 values */
            float local_max = row_vals[0];
            #pragma unroll
            for (int c = 1; c < 16; c++)
                local_max = fmaxf(local_max, row_vals[c]);

            /* Warp reduce max across 4 threads (lane%4 groups -> full 64-col) */
            local_max = fmaxf(local_max,
                __shfl_xor_sync(0xFFFFFFFF, local_max, 1));
            local_max = fmaxf(local_max,
                __shfl_xor_sync(0xFFFFFFFF, local_max, 2));

            /* Update running max */
            float new_max = fmaxf(local_max, rowmax[half]);
            float rescale = fast_exp2f(rowmax[half] - new_max);
            rowmax[half] = new_max;

            /* Rescale O accumulator for this row */
            #pragma unroll
            for (int p4 = 0; p4 < 16; p4++) {
                O_acc[(p4 << 2) | (half << 1) | 0] *= rescale;
                O_acc[(p4 << 2) | (half << 1) | 1] *= rescale;
            }

            /* Compute exp2(x - max) and sum */
            float local_sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < 16; c++) {
                row_vals[c] = fast_exp2f(row_vals[c] - new_max);
                local_sum += row_vals[c];
            }

            /* Warp reduce sum */
            local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, 1);
            local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, 2);

            /* Update running sum */
            rowsumexp[half] = rowsumexp[half] * rescale + local_sum;

            /* Store P to SMEM — packed b32 stores using precomputed addresses */
            #pragma unroll
            for (int p4 = 0; p4 < 8; p4++) {
                nv_bfloat16 v0 = __float2bfloat16(row_vals[p4 * 2 + 0]);
                nv_bfloat16 v1 = __float2bfloat16(row_vals[p4 * 2 + 1]);
                uint32_t packed = (uint32_t)(*(uint16_t*)&v0) |
                                  ((uint32_t)(*(uint16_t*)&v1) << 16);
                asm volatile("st.shared.b32 [%0], %1;"
                             :: "r"(p_addrs[half][p4]), "r"(packed));
            }
        }

        /* Ensure P writes visible to WGMMA */
        fence_view_async_shared();

        /* == Step 4: Wait for K[kv_id+1] load to complete == */
        asm volatile("cp.async.wait_all;");
        __syncthreads();

        /* == Step 5: Load V[kv_id] into K_smem[cur_stage] (reuse freed buffer) == */
        const nv_bfloat16 *V_cur = V_base + kv_id * BLOCK_KV * seq_stride;
        global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(
            cur_K_smem, V_cur, seq_stride, tid);
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncthreads();

        /* == Step 6: PV GEMM using P_smem x K_smem[cur_stage] (now holds V) == */
        const uint64_t desc_v_base = make_wgmma_desc(cur_K_smem, QKV_stride_bytes);

        wgmma_fence();

        #pragma unroll
        for (int ks = 0; ks < BLOCK_KV / 16; ks++) {
            uint64_t desc_p = gmma_desc_advance(desc_p_base, ks * 2);
            uint64_t desc_v = gmma_desc_advance(desc_v_base, ks * 2);
            wgmma_m64n128k16_f32_bf16(O_acc, desc_p, desc_v, 1);
        }

        wgmma_commit_group();
        wgmma_wait_group<0>();

        /* wgmma_wait_group<0> synchronizes all warps in the warp group.
         * Next iteration's cp.async writes to next_K_smem (different
         * buffer from cur_K_smem that PV just read). No sync needed. */

    } /* end kv_id loop */

    /* ---- Epilogue: finalize O and store to gmem ---- */
    wgmma_fence();  /* fence before reading O_acc */

    #pragma unroll
    for (int half = 0; half < 2; half++) {
        float inv_sum = fast_rcp(rowsumexp[half]);
        #pragma unroll
        for (int p4 = 0; p4 < 16; p4++) {
            O_acc[(p4 << 2) | (half << 1) | 0] *= inv_sum;
            O_acc[(p4 << 2) | (half << 1) | 1] *= inv_sum;
        }
    }

    /* Write O to global memory */
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        const int row = warp_id * 16 + half * 8 + (lane_id / 4);
        if (row < len_q - q_block_id * BLOCK_Q) {
            #pragma unroll
            for (int p4 = 0; p4 < 16; p4++) {
                #pragma unroll
                for (int s2 = 0; s2 < 2; s2++) {
                    const int col = p4 * 8 + s2 + (lane_id % 4) * 2;
                    const int idx = (p4 << 2) | (half << 1) | s2;
                    nv_bfloat16 val = __float2bfloat16(O_acc[idx]);
                    O_base[row * seq_stride + col] = val;
                }
            }
        }
    }
}

/* ======================================================================
 *  Helpers + kernel_run
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
    if (!causal && config_json)
        causal = json_bool(config_json, "causal", false);

    if (D != 128) return -2;

    {
        const int BLOCK_Q   = 64;
        const int BLOCK_KV  = 64;
        const int DIM_CONST = 128;
        const int TB_SIZE   = 128;

        int num_blocks = B * H * cdiv(S, BLOCK_Q);

        /* SMEM: Q(16KB) + K[0](16KB) + K[1](16KB) + P(8KB) = 56KB */
        int smem_size = BLOCK_Q * DIM_CONST * (int)sizeof(nv_bfloat16)      /* Q: 16KB */
                      + 2 * BLOCK_KV * DIM_CONST * (int)sizeof(nv_bfloat16) /* K[0]+K[1]: 32KB */
                      + BLOCK_Q * BLOCK_KV * (int)sizeof(nv_bfloat16);      /* P: 8KB */

        auto kernel = flash_attention_wgmma<BLOCK_Q, BLOCK_KV, DIM_CONST>;
        cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        kernel<<<num_blocks, TB_SIZE, smem_size, stream>>>(
            inputs[0], inputs[1], inputs[2], outputs[0],
            B, S, H, S, S, causal ? 1 : 0);
    }

    return 0;
}
