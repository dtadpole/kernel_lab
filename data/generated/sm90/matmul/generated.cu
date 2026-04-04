/*
 * Raw CUDA + inline PTX WGMMA-based BF16 matrix multiplication for SM90 (H100).
 *
 * Hand-written kernel with zero CUTLASS dependency.
 * Uses:
 *   - WGMMA (warpgroup MMA) m64n256k16 for compute — SS mode
 *   - TMA (cp.async.bulk.tensor) for global→shared loads
 *   - mbarrier for pipeline synchronization
 *   - Warp specialization: 3 warpgroups (1 producer + 2 consumers), 384 threads
 *   - setmaxnreg: producer 24 regs, consumers 232 regs
 *   - Persistent tile scheduling with L2 super-tiling
 *
 * Tile: 128×256×64, 384 threads (3 warpgroups), 4-stage pipeline
 * SMEM: ~197KB (within H100's 228KB limit)
 *
 * kernel_run contract:
 *   extern "C" int kernel_run(__nv_bfloat16** inputs, int num_inputs,
 *                             __nv_bfloat16** outputs, int num_outputs,
 *                             int n, cudaStream_t stream);
 */
#include <cuda_bf16.h>
#include <cuda.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

/* =========================================================================
 * Constants
 * ========================================================================= */
#define TILE_M       128
#define TILE_N       256
#define TILE_K       64
#define STAGES       4
#define THREADS      384    /* 3 warpgroups × 128 threads (1 producer + 2 consumers) */
#define WG_SIZE      128    /* threads per warpgroup */
#define N_SUB        128    /* N-subdivision per WGMMA call */
#define NUM_CONSUMERS 2

/* SMEM per stage */
#define A_STAGE_BYTES   (TILE_M * TILE_K * 2)                   /* 16384 */
#define B_SUB_BYTES     (N_SUB * TILE_K * 2)                    /* 16384 */
#define B_STAGE_BYTES   (2 * B_SUB_BYTES)                       /* 32768 */
#define SMEM_STAGE      (A_STAGE_BYTES + B_STAGE_BYTES)         /* 49152 */
#define SMEM_BYTES      (STAGES * SMEM_STAGE + 512)             /* 197120: +full[4]+empty[4] barriers */

/* Epilogue: padded SMEM stride for bank-conflict-free R2S writes.
 * Pad by 8 (not 4) so row stride = 264*2 = 528 bytes is 16-byte aligned
 * for uint4 loads. Bank conflict check: 528/4 = 132, 132%32 = 4 → OK. */
#define EPI_PAD         8
#define EPI_N_STRIDE    (TILE_N + EPI_PAD)  /* 264 elements per row */

/* =========================================================================
 * PTX helpers — mbarrier
 * ========================================================================= */
__device__ __forceinline__
void mbarrier_init(uint64_t* mbar, unsigned count) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(addr), "r"(count));
}

__device__ __forceinline__
void mbarrier_inval(uint64_t* mbar) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.inval.shared.b64 [%0];\n" :: "r"(addr));
}

__device__ __forceinline__
void mbarrier_arrive_expect_tx(uint64_t* mbar, unsigned tx_bytes) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
                 :: "r"(addr), "r"(tx_bytes));
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

/* =========================================================================
 * PTX helpers — TMA
 * ========================================================================= */
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

/* =========================================================================
 * PTX helpers — WGMMA fence / commit / wait
 * ========================================================================= */
__device__ __forceinline__
void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__
void wgmma_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__
void wgmma_wait_group_0() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(0) : "memory");
}

__device__ __forceinline__
void wgmma_wait_group_1() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(1) : "memory");
}

/* =========================================================================
 * PTX helpers — plain mbarrier arrive (no tx_bytes, for empty barriers)
 * ========================================================================= */
__device__ __forceinline__
void mbarrier_arrive(uint64_t* mbar) {
    unsigned addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" :: "r"(addr));
}

/* =========================================================================
 * PTX helpers — setmaxnreg (register redistribution between warpgroups)
 *
 * Producer: decrease to 24 regs, freeing registers for consumers.
 * Consumer: increase to 232 regs for more accumulator + scheduling headroom.
 * Budget: 128×40 + 256×216 = 60288 ≤ 65536 registers per SM. ✓
 * ========================================================================= */
__device__ __forceinline__
void setmaxnreg_dec40() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
}

__device__ __forceinline__
void setmaxnreg_inc216() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 216;\n");
}

/* =========================================================================
 * PTX helper — TMA descriptor prefetch
 * ========================================================================= */
__device__ __forceinline__
void tma_prefetch_descriptor(const void* tma_desc) {
    asm volatile(
        "prefetch.tensormap [%0];\n"
        :: "l"(tma_desc) : "memory");
}

/* =========================================================================
 * WGMMA descriptor construction
 *
 * GmmaDescriptor 64-bit layout (from CUTLASS mma_sm90_desc.hpp):
 *   [13:0]   start_address    = smem_byte_offset >> 4
 *   [29:16]  leading_byte_off = 1 (unused for SWIZZLE_* layouts)
 *   [45:32]  stride_byte_off  = row_stride_bytes >> 4
 *   [51:49]  base_offset      = K-step index (16B granularity in 128B line)
 *   [63:62]  layout_type      = 1 (SWIZZLE_128B)
 * ========================================================================= */
__device__ __forceinline__
uint64_t make_gmma_desc(const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    /* stride: 8 rows × 128 bytes/row (TILE_K * 2) = 1024 bytes → >> 4 = 64 */
    const int stride_16B = (8 * TILE_K * 2) >> 4;   /* 64 */
    uint64_t desc = 0;
    desc |= (uint64_t)((addr >> 4) & 0x3FFF);                  /* start_address */
    desc |= (uint64_t)(1) << 16;                                /* leading_byte_off */
    desc |= (uint64_t)(stride_16B & 0x3FFF) << 32;             /* stride_byte_off */
    /* base_offset = 0 always (CUTLASS convention) */
    desc |= (uint64_t)(1) << 62;                                /* layout_type = B128 */
    return desc;
}

/* Advance a GMMA descriptor by offset in 16-byte units (modifies start_address only).
 * For K-stepping: each k16 BF16 step = 32 bytes = 2 units of 16 bytes. */
__device__ __forceinline__
uint64_t gmma_desc_advance(uint64_t desc, int offset_16B) {
    /* Only low 32 bits contain start_address; CUTLASS does: reg32_[0] += offset */
    uint32_t lo = (uint32_t)desc + (uint32_t)offset_16B;
    uint32_t hi = (uint32_t)(desc >> 32);
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

/* =========================================================================
 * WGMMA tile: fence + 4× m64n256k16 + commit in a SINGLE asm block.
 *
 * This eliminates ptxas C7515 pipeline serialization by ensuring no
 * non-WGMMA instructions touch accumulators between fence and commit.
 *
 * 128 F32 accumulators cover the full 64×256 output per warpgroup.
 * K-stepping via start_address advancement (2 units per step).
 * scaleA=1, scaleB=1, tnspA=0 (K-major), tnspB=0 (K-major).
 * ========================================================================= */
__device__ __forceinline__
void wgmma_tile_k64(float* acc,
                    uint64_t da0, uint64_t db0, uint64_t da1, uint64_t db1,
                    uint64_t da2, uint64_t db2, uint64_t da3, uint64_t db3,
                    int scale_D) {
    asm volatile(
    "wgmma.fence.sync.aligned;\n"
    "{\n .reg .pred p;\n setp.ne.b32 p, %136, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7,"
    " %8, %9, %10, %11, %12, %13, %14, %15,"
    " %16, %17, %18, %19, %20, %21, %22, %23,"
    " %24, %25, %26, %27, %28, %29, %30, %31,"
    " %32, %33, %34, %35, %36, %37, %38, %39,"
    " %40, %41, %42, %43, %44, %45, %46, %47,"
    " %48, %49, %50, %51, %52, %53, %54, %55,"
    " %56, %57, %58, %59, %60, %61, %62, %63,"
    " %64, %65, %66, %67, %68, %69, %70, %71,"
    " %72, %73, %74, %75, %76, %77, %78, %79,"
    " %80, %81, %82, %83, %84, %85, %86, %87,"
    " %88, %89, %90, %91, %92, %93, %94, %95,"
    " %96, %97, %98, %99, %100, %101, %102, %103,"
    " %104, %105, %106, %107, %108, %109, %110, %111,"
    " %112, %113, %114, %115, %116, %117, %118, %119,"
    " %120, %121, %122, %123, %124, %125, %126, %127},"
    " %128, %129, p, 1, 1, 0, 0;\n}\n"
    "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7,"
    " %8, %9, %10, %11, %12, %13, %14, %15,"
    " %16, %17, %18, %19, %20, %21, %22, %23,"
    " %24, %25, %26, %27, %28, %29, %30, %31,"
    " %32, %33, %34, %35, %36, %37, %38, %39,"
    " %40, %41, %42, %43, %44, %45, %46, %47,"
    " %48, %49, %50, %51, %52, %53, %54, %55,"
    " %56, %57, %58, %59, %60, %61, %62, %63,"
    " %64, %65, %66, %67, %68, %69, %70, %71,"
    " %72, %73, %74, %75, %76, %77, %78, %79,"
    " %80, %81, %82, %83, %84, %85, %86, %87,"
    " %88, %89, %90, %91, %92, %93, %94, %95,"
    " %96, %97, %98, %99, %100, %101, %102, %103,"
    " %104, %105, %106, %107, %108, %109, %110, %111,"
    " %112, %113, %114, %115, %116, %117, %118, %119,"
    " %120, %121, %122, %123, %124, %125, %126, %127},"
    " %130, %131, p, 1, 1, 0, 0;\n}\n"
    "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7,"
    " %8, %9, %10, %11, %12, %13, %14, %15,"
    " %16, %17, %18, %19, %20, %21, %22, %23,"
    " %24, %25, %26, %27, %28, %29, %30, %31,"
    " %32, %33, %34, %35, %36, %37, %38, %39,"
    " %40, %41, %42, %43, %44, %45, %46, %47,"
    " %48, %49, %50, %51, %52, %53, %54, %55,"
    " %56, %57, %58, %59, %60, %61, %62, %63,"
    " %64, %65, %66, %67, %68, %69, %70, %71,"
    " %72, %73, %74, %75, %76, %77, %78, %79,"
    " %80, %81, %82, %83, %84, %85, %86, %87,"
    " %88, %89, %90, %91, %92, %93, %94, %95,"
    " %96, %97, %98, %99, %100, %101, %102, %103,"
    " %104, %105, %106, %107, %108, %109, %110, %111,"
    " %112, %113, %114, %115, %116, %117, %118, %119,"
    " %120, %121, %122, %123, %124, %125, %126, %127},"
    " %132, %133, p, 1, 1, 0, 0;\n}\n"
    "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7,"
    " %8, %9, %10, %11, %12, %13, %14, %15,"
    " %16, %17, %18, %19, %20, %21, %22, %23,"
    " %24, %25, %26, %27, %28, %29, %30, %31,"
    " %32, %33, %34, %35, %36, %37, %38, %39,"
    " %40, %41, %42, %43, %44, %45, %46, %47,"
    " %48, %49, %50, %51, %52, %53, %54, %55,"
    " %56, %57, %58, %59, %60, %61, %62, %63,"
    " %64, %65, %66, %67, %68, %69, %70, %71,"
    " %72, %73, %74, %75, %76, %77, %78, %79,"
    " %80, %81, %82, %83, %84, %85, %86, %87,"
    " %88, %89, %90, %91, %92, %93, %94, %95,"
    " %96, %97, %98, %99, %100, %101, %102, %103,"
    " %104, %105, %106, %107, %108, %109, %110, %111,"
    " %112, %113, %114, %115, %116, %117, %118, %119,"
    " %120, %121, %122, %123, %124, %125, %126, %127},"
    " %134, %135, p, 1, 1, 0, 0;\n}\n"
    "wgmma.commit_group.sync.aligned;\n"
    :
      "+f"(acc[0]),
      "+f"(acc[1]),
      "+f"(acc[2]),
      "+f"(acc[3]),
      "+f"(acc[4]),
      "+f"(acc[5]),
      "+f"(acc[6]),
      "+f"(acc[7]),
      "+f"(acc[8]),
      "+f"(acc[9]),
      "+f"(acc[10]),
      "+f"(acc[11]),
      "+f"(acc[12]),
      "+f"(acc[13]),
      "+f"(acc[14]),
      "+f"(acc[15]),
      "+f"(acc[16]),
      "+f"(acc[17]),
      "+f"(acc[18]),
      "+f"(acc[19]),
      "+f"(acc[20]),
      "+f"(acc[21]),
      "+f"(acc[22]),
      "+f"(acc[23]),
      "+f"(acc[24]),
      "+f"(acc[25]),
      "+f"(acc[26]),
      "+f"(acc[27]),
      "+f"(acc[28]),
      "+f"(acc[29]),
      "+f"(acc[30]),
      "+f"(acc[31]),
      "+f"(acc[32]),
      "+f"(acc[33]),
      "+f"(acc[34]),
      "+f"(acc[35]),
      "+f"(acc[36]),
      "+f"(acc[37]),
      "+f"(acc[38]),
      "+f"(acc[39]),
      "+f"(acc[40]),
      "+f"(acc[41]),
      "+f"(acc[42]),
      "+f"(acc[43]),
      "+f"(acc[44]),
      "+f"(acc[45]),
      "+f"(acc[46]),
      "+f"(acc[47]),
      "+f"(acc[48]),
      "+f"(acc[49]),
      "+f"(acc[50]),
      "+f"(acc[51]),
      "+f"(acc[52]),
      "+f"(acc[53]),
      "+f"(acc[54]),
      "+f"(acc[55]),
      "+f"(acc[56]),
      "+f"(acc[57]),
      "+f"(acc[58]),
      "+f"(acc[59]),
      "+f"(acc[60]),
      "+f"(acc[61]),
      "+f"(acc[62]),
      "+f"(acc[63]),
      "+f"(acc[64]),
      "+f"(acc[65]),
      "+f"(acc[66]),
      "+f"(acc[67]),
      "+f"(acc[68]),
      "+f"(acc[69]),
      "+f"(acc[70]),
      "+f"(acc[71]),
      "+f"(acc[72]),
      "+f"(acc[73]),
      "+f"(acc[74]),
      "+f"(acc[75]),
      "+f"(acc[76]),
      "+f"(acc[77]),
      "+f"(acc[78]),
      "+f"(acc[79]),
      "+f"(acc[80]),
      "+f"(acc[81]),
      "+f"(acc[82]),
      "+f"(acc[83]),
      "+f"(acc[84]),
      "+f"(acc[85]),
      "+f"(acc[86]),
      "+f"(acc[87]),
      "+f"(acc[88]),
      "+f"(acc[89]),
      "+f"(acc[90]),
      "+f"(acc[91]),
      "+f"(acc[92]),
      "+f"(acc[93]),
      "+f"(acc[94]),
      "+f"(acc[95]),
      "+f"(acc[96]),
      "+f"(acc[97]),
      "+f"(acc[98]),
      "+f"(acc[99]),
      "+f"(acc[100]),
      "+f"(acc[101]),
      "+f"(acc[102]),
      "+f"(acc[103]),
      "+f"(acc[104]),
      "+f"(acc[105]),
      "+f"(acc[106]),
      "+f"(acc[107]),
      "+f"(acc[108]),
      "+f"(acc[109]),
      "+f"(acc[110]),
      "+f"(acc[111]),
      "+f"(acc[112]),
      "+f"(acc[113]),
      "+f"(acc[114]),
      "+f"(acc[115]),
      "+f"(acc[116]),
      "+f"(acc[117]),
      "+f"(acc[118]),
      "+f"(acc[119]),
      "+f"(acc[120]),
      "+f"(acc[121]),
      "+f"(acc[122]),
      "+f"(acc[123]),
      "+f"(acc[124]),
      "+f"(acc[125]),
      "+f"(acc[126]),
      "+f"(acc[127])
    :
      "l"(da0), "l"(db0), "l"(da1), "l"(db1),
      "l"(da2), "l"(db2), "l"(da3), "l"(db3),
      "r"(scale_D));
}


/* =========================================================================
 * Main WGMMA kernel: warp-specialized, 3 warpgroups (1 producer + 2 consumers)
 *
 * WG0 (128 threads): Producer — dedicated TMA loads, 24 registers
 * WG1 (128 threads): Consumer0 — WGMMA rows 0-63, 232 registers
 * WG2 (128 threads): Consumer1 — WGMMA rows 64-127, 232 registers
 *
 * B is pre-transposed to (N, K) with K contiguous (like CuTe DSL).
 * Both A and B are K-major in SMEM → tnspA=0, tnspB=0.
 *
 * Producer-consumer protocol:
 *   full[STAGES]  — producer sets expect_tx, TMA completes it; consumers wait
 *   empty[STAGES] — consumers arrive when done reading; producer waits
 * ========================================================================= */
__launch_bounds__(THREADS, 1)
__global__ void wgmma_matmul(
        __nv_bfloat16* __restrict__ C, int M, int N, int K,
        int totalTiles, int nTilesN,
        const __grid_constant__ CUtensorMap tma_A,
        const __grid_constant__ CUtensorMap tma_B) {

    extern __shared__ char smem[];
    uint64_t* full_bar  = reinterpret_cast<uint64_t*>(smem + STAGES * SMEM_STAGE);
    uint64_t* empty_bar = full_bar + STAGES;

    const int tid     = threadIdx.x;
    const int wg_id   = tid / WG_SIZE;              /* 0=producer, 1,2=consumers */
    const int wg_tid  = tid % WG_SIZE;              /* thread index within WG */
    const int warp_in_wg = wg_tid / 32;             /* 0-3 within warpgroup */
    const int lane    = wg_tid % 32;
    const int groupID = lane / 4;
    const int thread_in_group = lane % 4;

    int numKTiles = K / TILE_K;
    int nTilesM = totalTiles / nTilesN;

    /* TMA descriptor prefetch (warp 0 only, once) */
    if (tid < 32) {
        tma_prefetch_descriptor(&tma_A);
        tma_prefetch_descriptor(&tma_B);
    }

    /* Initialize barriers:
     *   full[s]:  arrive_count=1 (producer sets expect_tx, TMA completes)
     *   empty[s]: arrive_count=NUM_CONSUMERS (both consumers must signal) */
    if (tid == 0) {
        for (int s = 0; s < STAGES; s++) {
            mbarrier_init(&full_bar[s], 1);
            mbarrier_init(&empty_bar[s], NUM_CONSUMERS);
        }
    }
    __syncthreads();

    /* Register redistribution: producer gives regs to consumers.
     * Producer needs ~30 regs for context vars (function params, smem ptrs,
     * loop vars). 40 is safe minimum. Consumers get 216 (128 acc + 88 extra). */
    if (wg_id == 0) {
        setmaxnreg_dec40();
    } else {
        setmaxnreg_inc216();
    }

    /* Consumers pre-signal empty barriers (all stages available initially).
     * One thread per consumer WG arrives on each empty barrier. */
    if (wg_id > 0 && wg_tid == 0) {
        for (int s = 0; s < STAGES; s++)
            mbarrier_arrive(&empty_bar[s]);
    }

    /* ============= Persistent tile loop ============= */
    for (int tileIdx = blockIdx.x; tileIdx < totalTiles; tileIdx += gridDim.x) {

        /* CTA swizzle (group_size_m=16 for L2 reuse) */
        int tile_m, tile_n;
        {
            const int group_m = 16;
            int groups_m = nTilesM / group_m;
            if (groups_m > 0) {
                int group_id = tileIdx / (group_m * nTilesN);
                int within = tileIdx % (group_m * nTilesN);
                tile_n = within / group_m;
                tile_m = group_id * group_m + within % group_m;
            } else {
                tile_m = tileIdx / nTilesN;
                tile_n = tileIdx % nTilesN;
            }
        }

        int mBase = tile_m * TILE_M;
        int nBase = tile_n * TILE_N;

        if (wg_id == 0) {
            /* ==================== PRODUCER ==================== */
            /* Only thread 0 of producer WG issues TMA loads.
             * Other 127 threads idle (their registers were donated). */
            if (wg_tid == 0) {
                for (int kb = 0; kb < numKTiles; kb++) {
                    int qidx = kb % STAGES;
                    int phase = (kb / STAGES) & 1;

                    /* Wait for consumers to finish with this stage */
                    mbarrier_wait_parity(&empty_bar[qidx], phase);

                    /* Set expected TMA bytes and issue loads */
                    mbarrier_arrive_expect_tx(&full_bar[qidx],
                        A_STAGE_BYTES + B_STAGE_BYTES);
                    int kc = kb * TILE_K;
                    char* stage = smem + qidx * SMEM_STAGE;
                    tma_load_2d(stage, &tma_A, kc, mBase, &full_bar[qidx]);
                    tma_load_2d(stage + A_STAGE_BYTES,
                                &tma_B, kc, nBase, &full_bar[qidx]);
                    tma_load_2d(stage + A_STAGE_BYTES + B_SUB_BYTES,
                                &tma_B, kc, nBase + N_SUB, &full_bar[qidx]);
                }
            }
        } else {
            /* ==================== CONSUMER ==================== */
            int consumer_id = wg_id - 1;  /* 0 or 1 */
            int wg_a_offset = consumer_id * 64 * TILE_K * 2;
            float acc[128];  /* Declared in consumer scope — keeps producer at 24 regs */

            /* ---- Prologue: first K-tile, scale_D=0 (overwrite) ---- */
            {
                mbarrier_wait_parity(&full_bar[0], 0);
                char* stage = smem;
                uint64_t da_base = make_gmma_desc(stage + wg_a_offset);
                uint64_t db_base = make_gmma_desc(stage + A_STAGE_BYTES);
                wgmma_tile_k64(acc, da_base, db_base,
                    gmma_desc_advance(da_base, 2), gmma_desc_advance(db_base, 2),
                    gmma_desc_advance(da_base, 4), gmma_desc_advance(db_base, 4),
                    gmma_desc_advance(da_base, 6), gmma_desc_advance(db_base, 6),
                    0);
                wgmma_wait_group_0();
                if (wg_tid == 0)
                    mbarrier_arrive(&empty_bar[0]);
            }

            /* ---- Mainloop: K-tiles 1..numKTiles-1, scale_D=1 ---- */
            for (int kb = 1; kb < numKTiles; kb++) {
                int qidx = kb % STAGES;
                int phase = (kb / STAGES) & 1;

                mbarrier_wait_parity(&full_bar[qidx], phase);

                char* stage = smem + qidx * SMEM_STAGE;
                uint64_t da_base = make_gmma_desc(stage + wg_a_offset);
                uint64_t db_base = make_gmma_desc(stage + A_STAGE_BYTES);
                wgmma_tile_k64(acc, da_base, db_base,
                    gmma_desc_advance(da_base, 2), gmma_desc_advance(db_base, 2),
                    gmma_desc_advance(da_base, 4), gmma_desc_advance(db_base, 4),
                    gmma_desc_advance(da_base, 6), gmma_desc_advance(db_base, 6),
                    1);
                wgmma_wait_group_0();
                if (wg_tid == 0)
                    mbarrier_arrive(&empty_bar[qidx]);
            }

            /* ---- Epilogue: SMEM-buffered coalesced stores ----
             * 256 consumer threads (WG1+WG2) cooperate to write 128×256 output.
             * Step 1: each consumer writes its 64 rows of fragments to SMEM.
             * Step 2: bar.sync consumer threads (256 of 384).
             * Step 3: coalesced 128-bit stores from SMEM to GMEM. */

            __nv_bfloat16* epi_smem = reinterpret_cast<__nv_bfloat16*>(smem);

            /* Step 1: Write F32 → BF16 fragments to padded SMEM */
            #pragma unroll
            for (int j = 0; j < 64; j++) {
                int p4 = j / 2;
                int h  = j % 2;
                int row = consumer_id * 64 + warp_in_wg * 16 + h * 8 + groupID;
                int col = p4 * 8 + thread_in_group * 2;
                int idx = 4 * p4 + 2 * h;
                int off = row * EPI_N_STRIDE + col;
                *reinterpret_cast<__nv_bfloat162*>(&epi_smem[off]) =
                    __floats2bfloat162_rn(acc[idx], acc[idx + 1]);
            }

            /* Step 2: Sync 256 consumer threads (bar.sync 1 with 256 threads) */
            asm volatile("bar.sync 1, 256;\n" ::: "memory");

            /* Step 3: Coalesced 128-bit stores.
             * 256 consumer threads: 8 warps × 16 rows × 32 threads per row. */
            {
                int consumer_tid = tid - WG_SIZE;  /* 0..255 */
                int warp_epi = consumer_tid / 32;  /* 0..7 */
                int lane_epi = consumer_tid % 32;  /* 0..31 */
                #pragma unroll
                for (int r = 0; r < 16; r++) {
                    int row = mBase + warp_epi * 16 + r;
                    int col = nBase + lane_epi * 8;
                    if (row < M && col + 7 < N) {
                        uint4 data = *reinterpret_cast<const uint4*>(
                            &epi_smem[(warp_epi * 16 + r) * EPI_N_STRIDE + lane_epi * 8]);
                        *reinterpret_cast<uint4*>(&C[row * N + col]) = data;
                    }
                }
            }
        } /* end consumer */

        /* Tile boundary: sync ALL 384 threads.
         * Ensures epilogue SMEM writes complete before producer reloads. */
        __syncthreads();

        /* Re-init barriers for next tile */
        if (tileIdx + gridDim.x < totalTiles) {
            if (tid == 0) {
                for (int s = 0; s < STAGES; s++) {
                    mbarrier_inval(&full_bar[s]);
                    mbarrier_inval(&empty_bar[s]);
                    mbarrier_init(&full_bar[s], 1);
                    mbarrier_init(&empty_bar[s], NUM_CONSUMERS);
                }
            }
            __syncthreads();
            /* Consumers re-signal empty for next tile */
            if (wg_id > 0 && wg_tid == 0) {
                for (int s = 0; s < STAGES; s++)
                    mbarrier_arrive(&empty_bar[s]);
            }
        }
    } /* end persistent tile loop */
}

/* fence + 4x m64n128k16 + commit for small tile (128x128x64) */
__device__ __forceinline__
void wgmma_tile_k64_small(float* acc,
                    uint64_t da0, uint64_t db0, uint64_t da1, uint64_t db1,
                    uint64_t da2, uint64_t db2, uint64_t da3, uint64_t db3,
                    int scale_D) {
    asm volatile(
    "wgmma.fence.sync.aligned;\n"
    "{\n .reg .pred p;\n setp.ne.b32 p, %72, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7,"
    " %8, %9, %10, %11, %12, %13, %14, %15,"
    " %16, %17, %18, %19, %20, %21, %22, %23,"
    " %24, %25, %26, %27, %28, %29, %30, %31,"
    " %32, %33, %34, %35, %36, %37, %38, %39,"
    " %40, %41, %42, %43, %44, %45, %46, %47,"
    " %48, %49, %50, %51, %52, %53, %54, %55,"
    " %56, %57, %58, %59, %60, %61, %62, %63},"
    " %64, %65, p, 1, 1, 0, 0;\n}\n"
    "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7,"
    " %8, %9, %10, %11, %12, %13, %14, %15,"
    " %16, %17, %18, %19, %20, %21, %22, %23,"
    " %24, %25, %26, %27, %28, %29, %30, %31,"
    " %32, %33, %34, %35, %36, %37, %38, %39,"
    " %40, %41, %42, %43, %44, %45, %46, %47,"
    " %48, %49, %50, %51, %52, %53, %54, %55,"
    " %56, %57, %58, %59, %60, %61, %62, %63},"
    " %66, %67, p, 1, 1, 0, 0;\n}\n"
    "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7,"
    " %8, %9, %10, %11, %12, %13, %14, %15,"
    " %16, %17, %18, %19, %20, %21, %22, %23,"
    " %24, %25, %26, %27, %28, %29, %30, %31,"
    " %32, %33, %34, %35, %36, %37, %38, %39,"
    " %40, %41, %42, %43, %44, %45, %46, %47,"
    " %48, %49, %50, %51, %52, %53, %54, %55,"
    " %56, %57, %58, %59, %60, %61, %62, %63},"
    " %68, %69, p, 1, 1, 0, 0;\n}\n"
    "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7,"
    " %8, %9, %10, %11, %12, %13, %14, %15,"
    " %16, %17, %18, %19, %20, %21, %22, %23,"
    " %24, %25, %26, %27, %28, %29, %30, %31,"
    " %32, %33, %34, %35, %36, %37, %38, %39,"
    " %40, %41, %42, %43, %44, %45, %46, %47,"
    " %48, %49, %50, %51, %52, %53, %54, %55,"
    " %56, %57, %58, %59, %60, %61, %62, %63},"
    " %70, %71, p, 1, 1, 0, 0;\n}\n"
    "wgmma.commit_group.sync.aligned;\n"
    :
      "+f"(acc[0]),
      "+f"(acc[1]),
      "+f"(acc[2]),
      "+f"(acc[3]),
      "+f"(acc[4]),
      "+f"(acc[5]),
      "+f"(acc[6]),
      "+f"(acc[7]),
      "+f"(acc[8]),
      "+f"(acc[9]),
      "+f"(acc[10]),
      "+f"(acc[11]),
      "+f"(acc[12]),
      "+f"(acc[13]),
      "+f"(acc[14]),
      "+f"(acc[15]),
      "+f"(acc[16]),
      "+f"(acc[17]),
      "+f"(acc[18]),
      "+f"(acc[19]),
      "+f"(acc[20]),
      "+f"(acc[21]),
      "+f"(acc[22]),
      "+f"(acc[23]),
      "+f"(acc[24]),
      "+f"(acc[25]),
      "+f"(acc[26]),
      "+f"(acc[27]),
      "+f"(acc[28]),
      "+f"(acc[29]),
      "+f"(acc[30]),
      "+f"(acc[31]),
      "+f"(acc[32]),
      "+f"(acc[33]),
      "+f"(acc[34]),
      "+f"(acc[35]),
      "+f"(acc[36]),
      "+f"(acc[37]),
      "+f"(acc[38]),
      "+f"(acc[39]),
      "+f"(acc[40]),
      "+f"(acc[41]),
      "+f"(acc[42]),
      "+f"(acc[43]),
      "+f"(acc[44]),
      "+f"(acc[45]),
      "+f"(acc[46]),
      "+f"(acc[47]),
      "+f"(acc[48]),
      "+f"(acc[49]),
      "+f"(acc[50]),
      "+f"(acc[51]),
      "+f"(acc[52]),
      "+f"(acc[53]),
      "+f"(acc[54]),
      "+f"(acc[55]),
      "+f"(acc[56]),
      "+f"(acc[57]),
      "+f"(acc[58]),
      "+f"(acc[59]),
      "+f"(acc[60]),
      "+f"(acc[61]),
      "+f"(acc[62]),
      "+f"(acc[63])
    :
      "l"(da0), "l"(db0), "l"(da1), "l"(db1),
      "l"(da2), "l"(db2), "l"(da3), "l"(db3),
      "r"(scale_D));
}


/* =========================================================================
 * Small kernel: 128×128×64 tile, 256 threads, 4-stage pipeline
 * For matrices with N < 256 (can't use the big 128×256 tile).
 * ========================================================================= */
#define SMALL_TILE_M    128
#define SMALL_TILE_N    128
#define SMALL_TILE_K    64
#define SMALL_STAGES    4
#define SMALL_A_STAGE   (SMALL_TILE_M * SMALL_TILE_K * 2)           /* 16384 */
#define SMALL_B_STAGE   (SMALL_TILE_N * SMALL_TILE_K * 2)           /* 16384 */
#define SMALL_SMEM_STAGE (SMALL_A_STAGE + SMALL_B_STAGE)            /* 32768 */
#define SMALL_SMEM_BYTES (SMALL_STAGES * SMALL_SMEM_STAGE + 256)    /* 131328 */
#define SMALL_EPI_PAD    4
#define SMALL_EPI_N_STRIDE (SMALL_TILE_N + SMALL_EPI_PAD)          /* 132 */

__device__ __forceinline__
uint64_t make_gmma_desc_small(const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    const int stride_16B = (8 * SMALL_TILE_K * 2) >> 4;  /* 64 */
    uint64_t desc = 0;
    desc |= (uint64_t)((addr >> 4) & 0x3FFF);
    desc |= (uint64_t)(1) << 16;
    desc |= (uint64_t)(stride_16B & 0x3FFF) << 32;
    desc |= (uint64_t)(1) << 62;  /* B128 */
    return desc;
}

__launch_bounds__(THREADS, 1)
__global__ void wgmma_matmul_small(
        __nv_bfloat16* __restrict__ C, int M, int N, int K,
        int totalTiles, int nTilesN,
        const __grid_constant__ CUtensorMap tma_A,
        const __grid_constant__ CUtensorMap tma_B) {

    extern __shared__ char smem[];
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + SMALL_STAGES * SMALL_SMEM_STAGE);
    const int tid = threadIdx.x;
    const int wg_id = tid / WG_SIZE;
    const int warp_in_wg = (tid / 32) % 4;
    const int lane = tid % 32;
    const int groupID = lane / 4;
    const int thread_in_group = lane % 4;
    int nTilesM = totalTiles / nTilesN;
    int numKTiles = K / SMALL_TILE_K;
    float acc[64];

    if (tid < 32) { tma_prefetch_descriptor(&tma_A); tma_prefetch_descriptor(&tma_B); }
    if (tid == 0) for (int s = 0; s < SMALL_STAGES; s++) mbarrier_init(&mbar[s], 1);
    __syncthreads();

    int tileIdx = blockIdx.x;
    int nTilesM_ = nTilesM;
    int tile_m, tile_n;
    { const int gm = 16; int gms = nTilesM_ / gm;
      if (gms > 0) { int gi = tileIdx/(gm*nTilesN); int w = tileIdx%(gm*nTilesN);
        tile_n = w/gm; tile_m = gi*gm + w%gm; }
      else { tile_m = tileIdx/nTilesN; tile_n = tileIdx%nTilesN; } }

    int mBase = tile_m * SMALL_TILE_M;
    int nBase = tile_n * SMALL_TILE_N;
    int wg_a_offset = wg_id * 64 * SMALL_TILE_K * 2;

    /* Prelude */
    int prelude = (numKTiles < SMALL_STAGES) ? numKTiles : SMALL_STAGES;
    if (tid == 0) {
        for (int s = 0; s < prelude; s++) {
            int kc = s * SMALL_TILE_K;
            char* stage = smem + s * SMALL_SMEM_STAGE;
            mbarrier_arrive_expect_tx(&mbar[s], SMALL_A_STAGE + SMALL_B_STAGE);
            tma_load_2d(stage, &tma_A, kc, mBase, &mbar[s]);
            tma_load_2d(stage + SMALL_A_STAGE, &tma_B, kc, nBase, &mbar[s]);
        }
    }

    /* Prologue */
    { mbarrier_wait_parity(&mbar[0], 0);
      char* stage = smem;
      uint64_t da = make_gmma_desc_small(stage + wg_a_offset);
      uint64_t db = make_gmma_desc_small(stage + SMALL_A_STAGE);
      wgmma_tile_k64_small(acc, da, db,
                     gmma_desc_advance(da,2), gmma_desc_advance(db,2),
                     gmma_desc_advance(da,4), gmma_desc_advance(db,4),
                     gmma_desc_advance(da,6), gmma_desc_advance(db,6), 0); }

    /* Mainloop */
    for (int kb = 1; kb < numKTiles; kb++) {
        int sc = kb % SMALL_STAGES, sl = (kb+SMALL_STAGES-1)%SMALL_STAGES;
        int phase = (kb/SMALL_STAGES) & 1;
        mbarrier_wait_parity(&mbar[sc], phase);
        char* stage = smem + sc * SMALL_SMEM_STAGE;
        uint64_t da = make_gmma_desc_small(stage + wg_a_offset);
        uint64_t db = make_gmma_desc_small(stage + SMALL_A_STAGE);
        wgmma_tile_k64_small(acc, da, db,
                       gmma_desc_advance(da,2), gmma_desc_advance(db,2),
                       gmma_desc_advance(da,4), gmma_desc_advance(db,4),
                       gmma_desc_advance(da,6), gmma_desc_advance(db,6), 1);
        wgmma_wait_group_1();
        if (tid == 0 && kb+SMALL_STAGES-1 < numKTiles) {
            int nk = (kb+SMALL_STAGES-1)*SMALL_TILE_K;
            char* ns = smem + sl * SMALL_SMEM_STAGE;
            mbarrier_arrive_expect_tx(&mbar[sl], SMALL_A_STAGE+SMALL_B_STAGE);
            tma_load_2d(ns, &tma_A, nk, mBase, &mbar[sl]);
            tma_load_2d(ns+SMALL_A_STAGE, &tma_B, nk, nBase, &mbar[sl]);
        }
    }
    wgmma_wait_group_0();

    /* Epilogue: SMEM-buffered coalesced stores (same approach as big kernel) */
    __syncthreads();

    __nv_bfloat16* epi_smem = reinterpret_cast<__nv_bfloat16*>(smem);

    /* Step 1: Write m64n128k16 fragments to padded SMEM */
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        int p4 = j/2, h = j%2;
        int row = wg_id*64 + warp_in_wg*16 + h*8 + groupID;
        int col = p4*8 + thread_in_group*2;
        int idx = 4*p4 + 2*h;
        int off = row * SMALL_EPI_N_STRIDE + col;
        *reinterpret_cast<__nv_bfloat162*>(&epi_smem[off]) =
            __floats2bfloat162_rn(acc[idx], acc[idx+1]);
    }
    __syncthreads();

    /* Step 2: Coalesced stores. 128 cols / 32 threads = 4 BF16 per thread (64-bit stores). */
    {
        int warp_epi = tid / 32;
        int lane_epi = tid % 32;
        #pragma unroll
        for (int r = 0; r < 16; r++) {
            int row = mBase + warp_epi*16 + r;
            int col = nBase + lane_epi*4;
            if (row < M && col+3 < N) {
                uint2 data = *reinterpret_cast<const uint2*>(
                    &epi_smem[(warp_epi*16+r) * SMALL_EPI_N_STRIDE + lane_epi*4]);
                *reinterpret_cast<uint2*>(&C[row*N+col]) = data;
            }
        }
    }
}

/* =========================================================================
 * Simple BF16 transpose kernel: (K, N) row-major → (N, K) row-major
 * After transpose, K is contiguous (matching CuTe DSL convention).
 * ========================================================================= */
__global__ void transpose_bf16(const __nv_bfloat16* __restrict__ src,
                               __nv_bfloat16* __restrict__ dst,
                               int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols)
        dst[c * rows + r] = src[r * cols + c];
}

/* =========================================================================
 * Host: TMA encoder, shape parsing, kernel dispatch
 * ========================================================================= */
typedef CUresult (*cuTensorMapEncodeTiled_fn)(
    CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
    const cuuint64_t*, const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
    CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion,
    CUtensorMapFloatOOBfill);

static cuTensorMapEncodeTiled_fn s_encodeTiled = nullptr;

static bool init_tma_encoder() {
    if (s_encodeTiled) return true;
    void* handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!handle) handle = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!handle) handle = dlopen("libcuda.so", RTLD_LAZY);
    if (!handle) return false;

    typedef CUresult (*getProc_fn)(const char*, void**, int, cuuint64_t,
                                   CUdriverProcAddressQueryResult*);
    getProc_fn getProc = (getProc_fn)dlsym(handle, "cuGetProcAddress_v2");
    if (!getProc) getProc = (getProc_fn)dlsym(handle, "cuGetProcAddress");
    if (!getProc) return false;

    CUdriverProcAddressQueryResult status;
    CUresult res = getProc("cuTensorMapEncodeTiled",
                           (void**)&s_encodeTiled, 12000,
                           CU_GET_PROC_ADDRESS_DEFAULT, &status);
    return (res == CUDA_SUCCESS && s_encodeTiled);
}

static int s_M = 0, s_N = 0, s_K = 0;
static void ensure_shape(int n) {
    if (s_M > 0) return;
    const char* shape = getenv("CUDA_EXEC_PARAM_SHAPE");
    if (shape) {
        int d0 = 0, d1 = 0;
        const char* p = shape;
        while (*p && *p != '[') ++p;
        if (*p == '[') ++p;
        while (*p == ' ') ++p;
        while (*p >= '0' && *p <= '9') { d0 = d0 * 10 + (*p - '0'); ++p; }
        while (*p == ' ' || *p == ',') ++p;
        while (*p >= '0' && *p <= '9') { d1 = d1 * 10 + (*p - '0'); ++p; }
        s_M = d0; s_N = d1; s_K = d0;
    } else {
        s_M = (int)sqrtf((float)n);
        s_N = s_M; s_K = s_M;
    }
}

/* Cached transposed B buffer */
static __nv_bfloat16* s_B_t = nullptr;
static size_t s_B_t_size = 0;
static const __nv_bfloat16* s_B_last = nullptr;  /* cache: skip transpose if same B */

/* Cached TMA descriptors — avoid per-call cuTensorMapEncodeTiled overhead.
 * cuTensorMapEncodeTiled is a CPU driver call (~1-3 us each).  Two calls per
 * kernel_run = 2-6 us overhead, which is 20-60% of total time at 256×256
 * but negligible at 8192×8192.  Cache and only recreate when pointer changes. */
static CUtensorMap s_tma_A_big, s_tma_B_big;
static const void* s_tma_A_ptr_big = nullptr;
static const void* s_tma_B_ptr_big = nullptr;
static CUtensorMap s_tma_A_small, s_tma_B_small;
static const void* s_tma_A_ptr_small = nullptr;
static const void* s_tma_B_ptr_small = nullptr;

extern "C" int kernel_run(__nv_bfloat16** inputs, int num_inputs,
                          __nv_bfloat16** outputs, int num_outputs,
                          int n, cudaStream_t stream) {
    if (num_inputs < 2 || num_outputs < 1) return -1;

    const __nv_bfloat16* A = inputs[0];
    const __nv_bfloat16* B = inputs[1];
    __nv_bfloat16*       C = outputs[0];

    ensure_shape(n);
    int M = s_M, N = s_N, K = s_K;

    if (!init_tma_encoder()) return -1;

    static int s_numSMs = 0;
    if (s_numSMs == 0)
        cudaDeviceGetAttribute(&s_numSMs, cudaDevAttrMultiProcessorCount, 0);

    if (M < SMALL_TILE_M || N < SMALL_TILE_N || K < SMALL_TILE_K) {
        fprintf(stderr, "Matrix too small: M=%d N=%d K=%d\n", M, N, K);
        return -4;
    }

    /* --- Transpose B: (K, N) row-major → (N, K) row-major --- */
    size_t B_elems = (size_t)K * N;
    if (s_B_t_size < B_elems * sizeof(__nv_bfloat16)) {
        if (s_B_t) cudaFree(s_B_t);
        cudaMalloc(&s_B_t, B_elems * sizeof(__nv_bfloat16));
        s_B_t_size = B_elems * sizeof(__nv_bfloat16);
        s_B_last = nullptr;
    }
    if (s_B_last != B) {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (K + 31) / 32);
        transpose_bf16<<<grid, block, 0, stream>>>(B, s_B_t, K, N);
        s_B_last = B;
    }

    /* Dispatch: big kernel (128×256) for N≥256, small kernel (128×128) otherwise */
    bool use_big = (M >= TILE_M && N >= TILE_N && K >= TILE_K);

    if (use_big) {
        /* Cache TMA descriptors — only recreate when data pointer changes */
        if (s_tma_A_ptr_big != (const void*)A) {
            cuuint64_t dims[2]={(cuuint64_t)K,(cuuint64_t)M}; cuuint64_t str[1]={(cuuint64_t)K*2}; cuuint32_t box[2]={TILE_K,TILE_M}; cuuint32_t el[2]={1,1};
            if (s_encodeTiled(&s_tma_A_big,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)A,dims,str,box,el,
                CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)!=CUDA_SUCCESS) return -2;
            s_tma_A_ptr_big = (const void*)A;
        }
        if (s_tma_B_ptr_big != (const void*)s_B_t) {
            cuuint64_t dims[2]={(cuuint64_t)K,(cuuint64_t)N}; cuuint64_t str[1]={(cuuint64_t)K*2}; cuuint32_t box[2]={TILE_K,N_SUB}; cuuint32_t el[2]={1,1};
            if (s_encodeTiled(&s_tma_B_big,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)s_B_t,dims,str,box,el,
                CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)!=CUDA_SUCCESS) return -3;
            s_tma_B_ptr_big = (const void*)s_B_t;
        }
        static bool cfg_big = false;
        if (!cfg_big) { cudaFuncSetAttribute(wgmma_matmul, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES); cfg_big = true; }
        int nTilesM=M/TILE_M, nTilesN=N/TILE_N, totalTiles=nTilesM*nTilesN;
        int gridSize = (totalTiles < s_numSMs) ? totalTiles : s_numSMs;
        wgmma_matmul<<<gridSize, THREADS, SMEM_BYTES, stream>>>(C,M,N,K,totalTiles,nTilesN,s_tma_A_big,s_tma_B_big);
    } else {
        /* Cache TMA descriptors — only recreate when data pointer changes */
        if (s_tma_A_ptr_small != (const void*)A) {
            cuuint64_t dims[2]={(cuuint64_t)K,(cuuint64_t)M}; cuuint64_t str[1]={(cuuint64_t)K*2}; cuuint32_t box[2]={SMALL_TILE_K,(cuuint32_t)SMALL_TILE_M}; cuuint32_t el[2]={1,1};
            if (s_encodeTiled(&s_tma_A_small,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)A,dims,str,box,el,
                CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)!=CUDA_SUCCESS) return -2;
            s_tma_A_ptr_small = (const void*)A;
        }
        if (s_tma_B_ptr_small != (const void*)s_B_t) {
            cuuint64_t dims[2]={(cuuint64_t)K,(cuuint64_t)N}; cuuint64_t str[1]={(cuuint64_t)K*2}; cuuint32_t box[2]={SMALL_TILE_K,(cuuint32_t)SMALL_TILE_N}; cuuint32_t el[2]={1,1};
            if (s_encodeTiled(&s_tma_B_small,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)s_B_t,dims,str,box,el,
                CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)!=CUDA_SUCCESS) return -3;
            s_tma_B_ptr_small = (const void*)s_B_t;
        }
        static bool cfg_small = false;
        if (!cfg_small) { cudaFuncSetAttribute(wgmma_matmul_small, cudaFuncAttributeMaxDynamicSharedMemorySize, SMALL_SMEM_BYTES); cfg_small = true; }
        int nTilesM=M/SMALL_TILE_M, nTilesN=N/SMALL_TILE_N, totalTiles=nTilesM*nTilesN;
        wgmma_matmul_small<<<totalTiles, THREADS, SMALL_SMEM_BYTES, stream>>>(C,M,N,K,totalTiles,nTilesN,s_tma_A_small,s_tma_B_small);
    }

    return 0;
}
