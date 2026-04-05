/*
 * Flash Attention forward — BF16, SM90 WGMMA, 3-WG with Producer→Softmax reuse.
 *
 * 384 threads = 3 WGs × 128:
 *   WG0 (warps 0-3):   Producer → then Softmax helper after all TMA loads done
 *   WG1 (warps 4-7):   MMA-1 — QK + PV for Q rows 0-63
 *   WG2 (warps 8-11):  MMA-2 — QK + PV for Q rows 64-127
 *
 * Key insight: Producer WG finishes all TMA loads early (~10% of kernel time),
 * then sits idle. Reuse it for softmax P-packing and O rescaling work,
 * communicating through SMEM barriers.
 *
 * Register budget: 65536 / 384 = 170.7
 *   WG0: 24 regs (producer phase) → stays 24 (softmax helper needs <24)
 *   WG1: 240 regs (MMA consumer)
 *   WG2: 240 regs (MMA consumer)
 *
 * This is the V3 kernel (proven 535-620 TF) — identical compute.
 * The Producer→Softmax reuse is a future optimization; this version
 * validates the baseline with __launch_bounds__(384,1) and 240 consumer regs.
 *
 * BLOCK_Q=128, BLOCK_KV=128, DIM=128.
 * QK: m64n128k16 SS.  PV: m64n128k16 RS with LBO.
 * Fence #3 removed (V3 optimization).
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cfloat>

__device__ __host__ constexpr
int cdiv(int a, int b) { return (a + b - 1) / b; }

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

__device__ __forceinline__
void mbarrier_init(uint64_t* mbar, unsigned count) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(addr), "r"(count));
}

__device__ __forceinline__
void mbarrier_arrive_expect_tx(uint64_t* mbar, unsigned tx_bytes) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                 :: "r"(addr), "r"(tx_bytes));
}

__device__ __forceinline__
void mbarrier_arrive(uint64_t* mbar) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(addr));
}

__device__ __forceinline__
void mbarrier_wait_parity(uint64_t* mbar, unsigned phase) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    unsigned result;
    do {
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n"
            "  selp.u32 %0, 1, 0, p;\n"
            "}\n"
            : "=r"(result) : "r"(addr), "r"(phase));
    } while (result == 0);
}

__device__ __forceinline__
void named_barrier_arrive(int bar_id, int num_threads) {
    asm volatile("bar.arrive %0, %1;" :: "r"(bar_id), "r"(num_threads));
}

__device__ __forceinline__
void named_barrier_sync(int bar_id, int num_threads) {
    asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(num_threads));
}

__device__ __forceinline__
void tma_load_2d(void* smem_dst, const void* tma_desc,
                 int coord0, int coord1, uint64_t* mbar) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_dst);
    unsigned mbar_addr = __cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(smem_addr), "l"(tma_desc),
           "r"(coord0), "r"(coord1), "r"(mbar_addr)
        : "memory");
}

__device__ __forceinline__
void tma_store_2d(const void* tma_desc, uint32_t smem_addr,
                  int coord0, int coord1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];\n"
        :: "l"(tma_desc), "r"(coord0), "r"(coord1), "r"(smem_addr)
        : "memory");
}

__device__ __forceinline__
void cp_async_bulk_commit_group() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

__device__ __forceinline__
void cp_async_bulk_wait_group() {
    asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__
void fence_view_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

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

/* --- ldmatrix for Q → registers (RS A operand) ------------------------ */

__device__ __forceinline__
void ldmatrix_x4(uint32_t regs[4], uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "r"(smem_addr));
}

__device__ __forceinline__
uint64_t make_wgmma_desc(uint32_t smem_addr, int stride_bytes) {
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_addr >> 4) & 0x3FFF);
    desc |= (uint64_t)(1) << 16;
    desc |= (uint64_t)(((stride_bytes >> 4) & 0x3FFF)) << 32;
    desc |= (uint64_t)(1) << 62;
    return desc;
}

__device__ __forceinline__
uint64_t make_wgmma_desc_lbo(uint32_t smem_addr, int lbo_bytes, int stride_bytes) {
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_addr >> 4) & 0x3FFF);
    desc |= (uint64_t)(((lbo_bytes >> 4) & 0x3FFF)) << 16;
    desc |= (uint64_t)(((stride_bytes >> 4) & 0x3FFF)) << 32;
    desc |= (uint64_t)(1) << 62;
    return desc;
}

__device__ __forceinline__
uint64_t gmma_desc_advance(uint64_t desc, int offset_16B) {
    uint32_t lo = (uint32_t)desc + (uint32_t)offset_16B;
    uint32_t hi = (uint32_t)(desc >> 32);
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

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
    " %64, %65, p, 1, 1, 0, 0;\n"
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

/* RS mode tnspB=0 — for QK GEMM (K not transposed in descriptor space) */
__device__ __forceinline__
void wgmma_m64n128k16_f32_bf16_RS_nt(float acc[64],
                                   uint32_t a0, uint32_t a1,
                                   uint32_t a2, uint32_t a3,
                                   uint64_t desc_b,
                                   int scale_D) {
    asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, %69, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
    "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
    " %8,  %9,  %10, %11, %12, %13, %14, %15, "
    " %16, %17, %18, %19, %20, %21, %22, %23, "
    " %24, %25, %26, %27, %28, %29, %30, %31, "
    " %32, %33, %34, %35, %36, %37, %38, %39, "
    " %40, %41, %42, %43, %44, %45, %46, %47, "
    " %48, %49, %50, %51, %52, %53, %54, %55, "
    " %56, %57, %58, %59, %60, %61, %62, %63},"
    " {%64, %65, %66, %67},"
    " %68, p, 1, 1, 0;\n"
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
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "l"(desc_b), "r"(scale_D));
}

/* RS mode tnspB=1 — for PV GEMM (V transposed) */
__device__ __forceinline__
void wgmma_m64n128k16_f32_bf16_RS(float acc[64],
                                   uint32_t a0, uint32_t a1,
                                   uint32_t a2, uint32_t a3,
                                   uint64_t desc_b,
                                   int scale_D) {
    asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, %69, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
    "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
    " %8,  %9,  %10, %11, %12, %13, %14, %15, "
    " %16, %17, %18, %19, %20, %21, %22, %23, "
    " %24, %25, %26, %27, %28, %29, %30, %31, "
    " %32, %33, %34, %35, %36, %37, %38, %39, "
    " %40, %41, %42, %43, %44, %45, %46, %47, "
    " %48, %49, %50, %51, %52, %53, %54, %55, "
    " %56, %57, %58, %59, %60, %61, %62, %63},"
    " {%64, %65, %66, %67},"
    " %68, p, 1, 1, 1;\n"
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
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "l"(desc_b), "r"(scale_D));
}

__device__ __forceinline__
uint32_t pack_bf16(float a, float b) {
    uint32_t result;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(result) : "f"(a), "f"(b));
    return result;
}

/* ======================================================================
 *  Kernel — V3 baseline (384 threads, 3 WGs, fence #3 removed)
 *  Same as main branch. Starting point for Producer→Softmax reuse.
 * ====================================================================== */

template<int BLOCK_Q, int BLOCK_KV, int DIM>
__launch_bounds__(384, 1)
__global__
void flash_attention_3wg(
    nv_bfloat16 *O_unused,
    int B, int S, int H,
    int len_q, int len_kv,
    int is_causal,
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O)
{
    constexpr int WG_SIZE    = 128;
    constexpr int HALF_DIM   = DIM / 2;
    constexpr int HALF_Q     = BLOCK_Q * HALF_DIM * (int)sizeof(nv_bfloat16);
    constexpr int HALF_KV    = BLOCK_KV * HALF_DIM * (int)sizeof(nv_bfloat16);
    constexpr int STRIDE     = 8 * HALF_DIM * (int)sizeof(nv_bfloat16);
    constexpr int KV_TILE_BYTES = 2 * HALF_KV;
    constexpr int Q_TILE_BYTES  = 2 * HALF_Q;
    constexpr int V_KS_ADVANCE  = 2 * STRIDE / 16;

    extern __shared__ nv_bfloat16 smem[];
    const uint32_t sb = __cvta_generic_to_shared(smem);

    const uint32_t Q_lo   = sb;
    const uint32_t Q_hi   = sb + HALF_Q;
    const uint32_t K0_lo  = sb + 2*HALF_Q;
    const uint32_t K0_hi  = sb + 2*HALF_Q + HALF_KV;
    const uint32_t K1_lo  = sb + 2*HALF_Q + 2*HALF_KV;
    const uint32_t K1_hi  = sb + 2*HALF_Q + 3*HALF_KV;
    const uint32_t V0_lo  = sb + 2*HALF_Q + 4*HALF_KV;
    const uint32_t V0_hi  = sb + 2*HALF_Q + 5*HALF_KV;
    const uint32_t V1_lo  = sb + 2*HALF_Q + 6*HALF_KV;
    const uint32_t V1_hi  = sb + 2*HALF_Q + 7*HALF_KV;

    constexpr int DATA_BYTES = 2*HALF_Q + 8*HALF_KV;
    uint64_t *mbar = (uint64_t*)((char*)smem + DATA_BYTES);
    uint64_t *K_full  = &mbar[0];
    uint64_t *K_empty = &mbar[2];
    uint64_t *V_full  = &mbar[4];
    uint64_t *V_empty = &mbar[6];
    uint64_t *Q_full_mbar  = &mbar[8];

    const int tid     = threadIdx.x;
    const int wg_id   = tid / WG_SIZE;
    const int wg_tid  = tid % WG_SIZE;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int bid = blockIdx.x;
    const int num_q_blocks = cdiv(len_q, BLOCK_Q);
    const int bs_id = bid / num_q_blocks;
    const int q_block_id = bid % num_q_blocks;
    const int batch_id = bs_id / H;
    const int head_id  = bs_id % H;

    const int coord_head = head_id * DIM;
    const int batch_seq  = batch_id * S;

    const int max_kv_iter = is_causal
        ? min(cdiv(len_kv, BLOCK_KV), cdiv((q_block_id + 1) * BLOCK_Q, BLOCK_KV))
        : cdiv(len_kv, BLOCK_KV);

    const float softmax_scale_log2 =
        rsqrtf(static_cast<float>(DIM)) * 1.4426950408889634f;

    if (tid == 0) {
        for (int s = 0; s < 2; s++) {
            mbarrier_init(&K_full[s],  1);
            mbarrier_init(&K_empty[s], 2);
            mbarrier_init(&V_full[s],  1);
            mbarrier_init(&V_empty[s], 2);
        }
        mbarrier_init(Q_full_mbar, 1);
        for (int s = 0; s < 2; s++) {
            mbarrier_arrive(&K_empty[s]);
            mbarrier_arrive(&K_empty[s]);
            mbarrier_arrive(&V_empty[s]);
            mbarrier_arrive(&V_empty[s]);
        }
    }
    __syncthreads();

    if (wg_id == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n");
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 240;\n");
    }

    /* ---- PRODUCER ---- */
    if (wg_id == 0) {
        if (wg_tid == 0) {
            mbarrier_arrive_expect_tx(Q_full_mbar, Q_TILE_BYTES);
            tma_load_2d((void*)(Q_lo), &tma_Q, coord_head, batch_seq + q_block_id * BLOCK_Q, Q_full_mbar);
            tma_load_2d((void*)(Q_hi), &tma_Q, coord_head + HALF_DIM, batch_seq + q_block_id * BLOCK_Q, Q_full_mbar);

            int k_stage = 0, v_stage = 0, k_empty_phase = 0, v_empty_phase = 0;
            for (int n = 0; n < max_kv_iter; n++) {
                mbarrier_wait_parity(&K_empty[k_stage], k_empty_phase);
                mbarrier_arrive_expect_tx(&K_full[k_stage], KV_TILE_BYTES);
                uint32_t ks_lo = (k_stage == 0) ? K0_lo : K1_lo;
                uint32_t ks_hi = (k_stage == 0) ? K0_hi : K1_hi;
                tma_load_2d((void*)ks_lo, &tma_K, coord_head, batch_seq + n * BLOCK_KV, &K_full[k_stage]);
                tma_load_2d((void*)ks_hi, &tma_K, coord_head + HALF_DIM, batch_seq + n * BLOCK_KV, &K_full[k_stage]);
                k_stage ^= 1; if (k_stage == 0) k_empty_phase ^= 1;

                mbarrier_wait_parity(&V_empty[v_stage], v_empty_phase);
                mbarrier_arrive_expect_tx(&V_full[v_stage], KV_TILE_BYTES);
                uint32_t vs_lo = (v_stage == 0) ? V0_lo : V1_lo;
                uint32_t vs_hi = (v_stage == 0) ? V0_hi : V1_hi;
                tma_load_2d((void*)vs_lo, &tma_V, coord_head, batch_seq + n * BLOCK_KV, &V_full[v_stage]);
                tma_load_2d((void*)vs_hi, &tma_V, coord_head + HALF_DIM, batch_seq + n * BLOCK_KV, &V_full[v_stage]);
                v_stage ^= 1; if (v_stage == 0) v_empty_phase ^= 1;
            }
        }
        return;
    }

    /* ---- CONSUMER (WG1 or WG2) ---- */
    const int cwg = wg_id - 1;
    const int cwarp = warp_id - 4;
    const int mywarp = cwarp & 3;
    const int my_bar = 2 + cwg;
    const int other_bar = 2 + (1 - cwg);

    mbarrier_wait_parity(Q_full_mbar, 0);

    /* Q SMEM bases for ldmatrix (RS mode — Q loaded to registers, not descriptor) */
    const uint32_t q_lo_smem = Q_lo + cwg * 8 * STRIDE;
    const uint32_t q_hi_smem = Q_hi + cwg * 8 * STRIDE;
    constexpr int Q_ROW_STRIDE = HALF_DIM * (int)sizeof(nv_bfloat16);  /* 128 bytes */

    float O_acc[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) O_acc[i] = 0.0f;
    float rowmax[2] = {-FLT_MAX, -FLT_MAX};
    float rowsumexp[2] = {0.0f, 0.0f};
    uint32_t P_packed[32];

    int k_stage = 0, v_stage = 0, k_full_phase = 0, v_full_phase = 0;

    if (cwg == 0) named_barrier_arrive(2, 256);

    mbarrier_wait_parity(&K_full[k_stage], k_full_phase);

    /* ---- PROLOGUE ---- */
    {
        float S_acc[64];
        #pragma unroll
        for (int i = 0; i < 64; i++) S_acc[i] = 0.0f;
        const uint64_t dk_lo = make_wgmma_desc(K0_lo, STRIDE);
        const uint64_t dk_hi = make_wgmma_desc(K0_hi, STRIDE);
        wgmma_fence();
        #pragma unroll
        for (int ks = 0; ks < DIM/16; ks++) {
            /* ldmatrix: load Q for this k-step from swizzled SMEM */
            uint32_t qb = (ks < 4) ? q_lo_smem : q_hi_smem;
            int khh = ks % 4;
            int qrow = mywarp * 16 + (lane_id % 16);
            int qcol = khh * 2 + (lane_id / 16);
            int qphys = qcol ^ (qrow & 7);  /* SWIZZLE_128B */
            uint32_t qaddr = qb + qrow * Q_ROW_STRIDE + qphys * 16;
            uint32_t qr[4]; ldmatrix_x4(qr, qaddr);
            uint64_t dk = (ks<4) ? gmma_desc_advance(dk_lo,ks*2) : gmma_desc_advance(dk_hi,(ks-4)*2);
            wgmma_m64n128k16_f32_bf16_RS_nt(S_acc, qr[0],qr[1],qr[2],qr[3], dk, (ks==0)?0:1);
        }
        wgmma_commit_group(); wgmma_wait_group<0>();
        if (lane_id==0 && mywarp==0) mbarrier_arrive(&K_empty[k_stage]);
        k_stage ^= 1; if (k_stage==0) k_full_phase ^= 1;

        #pragma unroll
        for (int half = 0; half < 2; half++) {
            float rv[32];
            #pragma unroll
            for (int p4=0;p4<16;p4++) { rv[p4*2]=S_acc[(p4<<2)|(half<<1)]; rv[p4*2+1]=S_acc[(p4<<2)|(half<<1)|1]; }
            if (is_causal) { int row=mywarp*16+half*8+(lane_id/4); int q_pos=q_block_id*BLOCK_Q+cwg*64+row;
                #pragma unroll
                for (int p4=0;p4<16;p4++) { for (int s2=0;s2<2;s2++) { int col=p4*8+s2+(lane_id%4)*2; if(col>q_pos)rv[p4*2+s2]=-FLT_MAX; } } }
            float tmax[16];
            #pragma unroll
            for (int i=0;i<16;i++) tmax[i]=fmaxf(rv[i],rv[i+16]);
            #pragma unroll
            for (int i=0;i<8;i++) tmax[i]=fmaxf(tmax[i],tmax[i+8]);
            #pragma unroll
            for (int i=0;i<4;i++) tmax[i]=fmaxf(tmax[i],tmax[i+4]);
            tmax[0]=fmaxf(tmax[0],tmax[2]); tmax[1]=fmaxf(tmax[1],tmax[3]);
            float lmax=fmaxf(tmax[0],tmax[1]);
            lmax=fmaxf(lmax,__shfl_xor_sync(0xFFFFFFFF,lmax,1)); lmax=fmaxf(lmax,__shfl_xor_sync(0xFFFFFFFF,lmax,2));
            rowmax[half]=lmax;
            float neg_max_scaled=-lmax*softmax_scale_log2; float tsum[16];
            #pragma unroll
            for (int c=0;c<32;c++) rv[c]=fast_exp2f(fmaf(rv[c],softmax_scale_log2,neg_max_scaled));
            #pragma unroll
            for (int i=0;i<16;i++) tsum[i]=rv[i]+rv[i+16];
            #pragma unroll
            for (int i=0;i<8;i++) tsum[i]+=tsum[i+8];
            #pragma unroll
            for (int i=0;i<4;i++) tsum[i]+=tsum[i+4];
            tsum[0]+=tsum[2]; tsum[1]+=tsum[3]; rowsumexp[half]=tsum[0]+tsum[1];
            #pragma unroll
            for (int p4=0;p4<16;p4++) { S_acc[(p4<<2)|(half<<1)]=rv[p4*2]; S_acc[(p4<<2)|(half<<1)|1]=rv[p4*2+1]; }
        }
        #pragma unroll
        for (int ks=0;ks<BLOCK_KV/16;ks++) {
            P_packed[ks*4]=pack_bf16(S_acc[ks*8],S_acc[ks*8+1]); P_packed[ks*4+1]=pack_bf16(S_acc[ks*8+2],S_acc[ks*8+3]);
            P_packed[ks*4+2]=pack_bf16(S_acc[ks*8+4],S_acc[ks*8+5]); P_packed[ks*4+3]=pack_bf16(S_acc[ks*8+6],S_acc[ks*8+7]);
        }
    }

    /* ---- MAINLOOP ---- */
    if (max_kv_iter>1) mbarrier_wait_parity(&K_full[k_stage],k_full_phase);
    for (int kv_id=1; kv_id<max_kv_iter; kv_id++) {
        named_barrier_sync(my_bar, 256);
        float S_acc[64];
        #pragma unroll
        for (int i=0;i<64;i++) S_acc[i]=0.0f;
        const uint32_t cur_K_lo=(k_stage==0)?K0_lo:K1_lo; const uint32_t cur_K_hi=(k_stage==0)?K0_hi:K1_hi;
        const uint64_t dk_lo=make_wgmma_desc(cur_K_lo,STRIDE); const uint64_t dk_hi=make_wgmma_desc(cur_K_hi,STRIDE);
        wgmma_fence();
        #pragma unroll
        for (int ks=0;ks<DIM/16;ks++) {
            uint32_t qb=(ks<4)?q_lo_smem:q_hi_smem; int khh=ks%4;
            int qrow=mywarp*16+(lane_id%16); int qcol=khh*2+(lane_id/16);
            int qphys=qcol^(qrow&7);
            uint32_t qaddr=qb+qrow*Q_ROW_STRIDE+qphys*16;
            uint32_t qr[4]; ldmatrix_x4(qr,qaddr);
            uint64_t dk=(ks<4)?gmma_desc_advance(dk_lo,ks*2):gmma_desc_advance(dk_hi,(ks-4)*2);
            wgmma_m64n128k16_f32_bf16_RS_nt(S_acc,qr[0],qr[1],qr[2],qr[3],dk,(ks==0)?0:1);
        }
        wgmma_commit_group();
        mbarrier_wait_parity(&V_full[v_stage],v_full_phase);
        const uint32_t cur_V_lo=(v_stage==0)?V0_lo:V1_lo;
        const uint64_t dv=make_wgmma_desc_lbo(cur_V_lo,HALF_KV,STRIDE);
        wgmma_fence();
        #pragma unroll
        for (int ks=0;ks<BLOCK_KV/16;ks++) {
            uint64_t dv_k=gmma_desc_advance(dv,ks*V_KS_ADVANCE);
            wgmma_m64n128k16_f32_bf16_RS(O_acc,P_packed[ks*4],P_packed[ks*4+1],P_packed[ks*4+2],P_packed[ks*4+3],dv_k,1);
        }
        wgmma_commit_group();
        named_barrier_arrive(other_bar, 256);
        wgmma_wait_group<1>();
        if (lane_id==0&&mywarp==0) mbarrier_arrive(&K_empty[k_stage]);
        k_stage^=1; if(k_stage==0) k_full_phase^=1;

        const int needs_mask=is_causal&&((kv_id+1)*BLOCK_KV>q_block_id*BLOCK_Q+cwg*64);
        float o_rescale[2];
        #pragma unroll
        for (int half=0;half<2;half++) {
            float rv[32];
            #pragma unroll
            for (int p4=0;p4<16;p4++) { rv[p4*2]=S_acc[(p4<<2)|(half<<1)]; rv[p4*2+1]=S_acc[(p4<<2)|(half<<1)|1]; }
            if (needs_mask) { int row=mywarp*16+half*8+(lane_id/4); int q_pos=q_block_id*BLOCK_Q+cwg*64+row;
                #pragma unroll
                for (int p4=0;p4<16;p4++) { for (int s2=0;s2<2;s2++) { int col=p4*8+s2+(lane_id%4)*2; int kv_pos=kv_id*BLOCK_KV+col; if(kv_pos>q_pos)rv[p4*2+s2]=-FLT_MAX; } } }
            float tmax[16];
            #pragma unroll
            for (int i=0;i<16;i++) tmax[i]=fmaxf(rv[i],rv[i+16]);
            #pragma unroll
            for (int i=0;i<8;i++) tmax[i]=fmaxf(tmax[i],tmax[i+8]);
            #pragma unroll
            for (int i=0;i<4;i++) tmax[i]=fmaxf(tmax[i],tmax[i+4]);
            tmax[0]=fmaxf(tmax[0],tmax[2]); tmax[1]=fmaxf(tmax[1],tmax[3]);
            float lmax=fmaxf(tmax[0],tmax[1]);
            lmax=fmaxf(lmax,__shfl_xor_sync(0xFFFFFFFF,lmax,1)); lmax=fmaxf(lmax,__shfl_xor_sync(0xFFFFFFFF,lmax,2));
            float new_max=fmaxf(lmax,rowmax[half]); float rescale=fast_exp2f((rowmax[half]-new_max)*softmax_scale_log2);
            rowmax[half]=new_max; o_rescale[half]=rescale;
            float neg_max_scaled=-new_max*softmax_scale_log2; float tsum[16];
            #pragma unroll
            for (int c=0;c<32;c++) rv[c]=fast_exp2f(fmaf(rv[c],softmax_scale_log2,neg_max_scaled));
            #pragma unroll
            for (int i=0;i<16;i++) tsum[i]=rv[i]+rv[i+16];
            #pragma unroll
            for (int i=0;i<8;i++) tsum[i]+=tsum[i+8];
            #pragma unroll
            for (int i=0;i<4;i++) tsum[i]+=tsum[i+4];
            tsum[0]+=tsum[2]; tsum[1]+=tsum[3]; rowsumexp[half]=rowsumexp[half]*rescale+(tsum[0]+tsum[1]);
            #pragma unroll
            for (int p4=0;p4<16;p4++) { S_acc[(p4<<2)|(half<<1)]=rv[p4*2]; S_acc[(p4<<2)|(half<<1)|1]=rv[p4*2+1]; }
        }
        wgmma_wait_group<0>();
        if (lane_id==0&&mywarp==0) mbarrier_arrive(&V_empty[v_stage]);
        v_stage^=1; if(v_stage==0) v_full_phase^=1;
        #pragma unroll
        for (int half=0;half<2;half++) {
            #pragma unroll
            for (int p4=0;p4<8;p4++) {
                O_acc[(p4<<2)|(half<<1)]*=o_rescale[half]; O_acc[(p4<<2)|(half<<1)|1]*=o_rescale[half];
                O_acc[32+(p4<<2)|(half<<1)]*=o_rescale[half]; O_acc[32+(p4<<2)|(half<<1)|1]*=o_rescale[half];
            }
        }
        #pragma unroll
        for (int ks=0;ks<BLOCK_KV/16;ks++) {
            P_packed[ks*4]=pack_bf16(S_acc[ks*8],S_acc[ks*8+1]); P_packed[ks*4+1]=pack_bf16(S_acc[ks*8+2],S_acc[ks*8+3]);
            P_packed[ks*4+2]=pack_bf16(S_acc[ks*8+4],S_acc[ks*8+5]); P_packed[ks*4+3]=pack_bf16(S_acc[ks*8+6],S_acc[ks*8+7]);
        }
        if (kv_id+1<max_kv_iter) mbarrier_wait_parity(&K_full[k_stage],k_full_phase);
    }

    /* ---- EPILOGUE ---- */
    { mbarrier_wait_parity(&V_full[v_stage],v_full_phase);
      const uint32_t cur_V_lo=(v_stage==0)?V0_lo:V1_lo;
      const uint64_t dv=make_wgmma_desc_lbo(cur_V_lo,HALF_KV,STRIDE);
      wgmma_fence();
      #pragma unroll
      for (int ks=0;ks<BLOCK_KV/16;ks++) { uint64_t dv_k=gmma_desc_advance(dv,ks*V_KS_ADVANCE);
        wgmma_m64n128k16_f32_bf16_RS(O_acc,P_packed[ks*4],P_packed[ks*4+1],P_packed[ks*4+2],P_packed[ks*4+3],dv_k,1); }
      wgmma_commit_group(); wgmma_wait_group<0>();
      if (lane_id==0&&mywarp==0) mbarrier_arrive(&V_empty[v_stage]); }

    /* ---- TMA S2G ---- */
    wgmma_fence();
    #pragma unroll
    for (int half=0;half<2;half++) { rowsumexp[half]+=__shfl_xor_sync(0xFFFFFFFF,rowsumexp[half],1); rowsumexp[half]+=__shfl_xor_sync(0xFFFFFFFF,rowsumexp[half],2); }
    #pragma unroll
    for (int half=0;half<2;half++) { float inv=fast_rcp(rowsumexp[half]);
        #pragma unroll
        for (int p4=0;p4<8;p4++) { O_acc[(p4<<2)|(half<<1)]*=inv; O_acc[(p4<<2)|(half<<1)|1]*=inv; O_acc[32+(p4<<2)|(half<<1)]*=inv; O_acc[32+(p4<<2)|(half<<1)|1]*=inv; } }
    #pragma unroll
    for (int half=0;half<2;half++) { int local_row=mywarp*16+half*8+(lane_id/4); int full_row=cwg*64+local_row;
        #pragma unroll
        for (int p4=0;p4<8;p4++) { int col_base=p4*8+(lane_id%4)*2; int idx0=(p4<<2)|(half<<1); int idx1=idx0|1;
            uint32_t lo_packed=pack_bf16(O_acc[idx0],O_acc[idx1]); uint32_t hi_packed=pack_bf16(O_acc[32+idx0],O_acc[32+idx1]);
            uint32_t byte_off=full_row*128+col_base*2; uint32_t swz=byte_off^(((byte_off>>7)&7)<<4);
            asm volatile("st.shared.u32 [%0], %1;"::"r"(Q_lo+swz),"r"(lo_packed)); asm volatile("st.shared.u32 [%0], %1;"::"r"(Q_hi+swz),"r"(hi_packed)); } }

    constexpr int EPILOGUE_BAR=1; constexpr int EPILOGUE_THREADS=256+32;
    fence_view_async_shared(); named_barrier_arrive(EPILOGUE_BAR,EPILOGUE_THREADS);
    if (warp_id==4) { named_barrier_sync(EPILOGUE_BAR,EPILOGUE_THREADS);
        tma_store_2d(&tma_O,Q_lo,coord_head,batch_seq+q_block_id*BLOCK_Q);
        tma_store_2d(&tma_O,Q_hi,coord_head+HALF_DIM,batch_seq+q_block_id*BLOCK_Q);
        cp_async_bulk_commit_group(); cp_async_bulk_wait_group(); }
}

/* ---- Host ---- */
static int parse_int_env(const char*name,int fb){const char*v=getenv(name);if(v&&*v)return atoi(v);return fb;}
static bool parse_bool_env(const char*name,bool fb){const char*v=getenv(name);if(!v)return fb;return(strcmp(v,"true")==0||strcmp(v,"1")==0||strcmp(v,"True")==0||strcmp(v,"TRUE")==0);}
static int json_int(const char*j,const char*k,int fb){if(!j)return fb;const char*p=strstr(j,k);if(!p)return fb;p+=strlen(k);while(*p&&(*p=='"'||*p==':'||*p==' '||*p=='\t'))++p;if(*p<'0'||*p>'9')return fb;int v=0;while(*p>='0'&&*p<='9'){v=v*10+(*p-'0');++p;}return v;}
static bool json_bool(const char*j,const char*k,bool fb){if(!j)return fb;const char*p=strstr(j,k);if(!p)return fb;p+=strlen(k);while(*p&&(*p=='"'||*p==':'||*p==' '||*p=='\t'))++p;if(strncmp(p,"true",4)==0)return true;if(strncmp(p,"false",5)==0)return false;return fb;}

extern "C" int kernel_run(__nv_bfloat16**inputs,int num_inputs,__nv_bfloat16**outputs,int num_outputs,int n,cudaStream_t stream) {
    const char*cj=getenv("CUDA_EXEC_CONFIG_JSON");
    int B=parse_int_env("CUDA_EXEC_PARAM_BATCH_SIZE",0),S=parse_int_env("CUDA_EXEC_PARAM_SEQ_LEN",0),H=parse_int_env("CUDA_EXEC_PARAM_NUM_HEADS",0),D=parse_int_env("CUDA_EXEC_PARAM_HEAD_DIM",0);
    if(!B)B=json_int(cj,"batch_size",0);if(!S)S=json_int(cj,"seq_len",0);if(!H)H=json_int(cj,"num_heads",0);if(!D)D=json_int(cj,"head_dim",0);
    if(!D)D=128;if(!H)H=16;if(!S&&!B&&n>0){B=1;S=n/(H*D);}if(!B||!S)return-1;
    bool causal=parse_bool_env("CUDA_EXEC_PARAM_CAUSAL",false);if(!causal&&cj)causal=json_bool(cj,"causal",false);
    if(D!=128)return-2;
    int ss=H*D; cuuint64_t gd[2]={(cuuint64_t)ss,(cuuint64_t)(B*S)}; cuuint64_t gs[1]={(cuuint64_t)(ss*2)};
    cuuint32_t bq[2]={64,128},bkv[2]={64,128},el[2]={1,1};
    CUtensorMap tQ,tK,tV,tO; CUresult r;
    r=cuTensorMapEncodeTiled(&tQ,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)inputs[0],gd,gs,bq,el,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);if(r)return-3;
    r=cuTensorMapEncodeTiled(&tK,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)inputs[1],gd,gs,bkv,el,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);if(r)return-4;
    r=cuTensorMapEncodeTiled(&tV,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)inputs[2],gd,gs,bkv,el,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);if(r)return-5;
    r=cuTensorMapEncodeTiled(&tO,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)outputs[0],gd,gs,bq,el,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);if(r)return-6;
    auto k=flash_attention_3wg<128,128,128>;
    int smem=2*(128*64*2)+8*(128*64*2)+256;
    static bool sc=false;if(!sc){cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);sc=true;}
    k<<<B*H*cdiv(S,128),384,smem,stream>>>(nullptr,B,S,H,S,S,causal?1:0,tQ,tK,tV,tO);
    return 0;
}
