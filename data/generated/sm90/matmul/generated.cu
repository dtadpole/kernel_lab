/*
 * Raw CUDA + inline PTX WGMMA-based BF16 matrix multiplication for SM90 (H100).
 *
 * Hand-written kernel with zero CUTLASS dependency.
 * Uses:
 *   - WGMMA (warpgroup MMA) m64n256k16 for compute — 2 warpgroups, SS mode
 *   - TMA (cp.async.bulk.tensor) for global→shared loads
 *   - mbarrier for pipeline synchronization
 *   - Persistent tile scheduling with L2 super-tiling
 *
 * Tile: 128×256×64, 256 threads (2 warpgroups), 4-stage pipeline
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
#define THREADS      256    /* 2 warpgroups × 128 threads */
#define WG_SIZE      128    /* threads per warpgroup */
#define N_SUB        128    /* N-subdivision per WGMMA call */

/* SMEM per stage */
#define A_STAGE_BYTES   (TILE_M * TILE_K * 2)                   /* 16384 */
#define B_SUB_BYTES     (N_SUB * TILE_K * 2)                    /* 16384 */
#define B_STAGE_BYTES   (2 * B_SUB_BYTES)                       /* 32768 */
#define SMEM_STAGE      (A_STAGE_BYTES + B_STAGE_BYTES)         /* 49152 */
#define SMEM_BYTES      (STAGES * SMEM_STAGE + 256)             /* 196864 */

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
 * Main WGMMA kernel: 128×256×64 tile, 256 threads, 4-stage pipeline
 *
 * B is pre-transposed to (N, K) with K contiguous (like CuTe DSL).
 * Both A and B are K-major in SMEM → tnspA=0, tnspB=0.
 * ========================================================================= */
__launch_bounds__(THREADS, 1)
__global__ void wgmma_matmul(
        __nv_bfloat16* __restrict__ C, int M, int N, int K,
        int totalTiles, int nTilesN,
        const __grid_constant__ CUtensorMap tma_A,
        const __grid_constant__ CUtensorMap tma_B) {

    extern __shared__ char smem[];
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + STAGES * SMEM_STAGE);

    const int tid = threadIdx.x;
    const int wg_id = tid / WG_SIZE;              /* 0 or 1 */
    const int warp_in_wg = (tid / 32) % 4;       /* 0-3 within warpgroup */
    const int lane = tid % 32;
    const int groupID = lane / 4;                 /* 0-7 */
    const int thread_in_group = lane % 4;         /* 0-3 */

    int numKTiles = K / TILE_K;
    int nTilesM = totalTiles / nTilesN;

    /* 128 F32 accumulators for full 64×256 output per warpgroup (m64n256k16). */
    float acc[128];
    int tileIdx = blockIdx.x;

    /* CTA swizzle (CuTe DSL pattern): group_size_m=8 for L2 reuse */
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
    int wg_a_offset = wg_id * 64 * TILE_K * 2;

    /* TMA descriptor prefetch (warp 0 only, like CuTe DSL) */
    if (tid < 32) {
        tma_prefetch_descriptor(&tma_A);
        tma_prefetch_descriptor(&tma_B);
    }

    /* Initialize mbarriers (once per block, no re-init needed) */
    if (tid == 0)
        for (int s = 0; s < STAGES; s++)
            mbarrier_init(&mbar[s], 1);
    __syncthreads();

    /* ---- Prelude: pre-fill pipeline (warp 0 issues TMA) ---- */
    int prelude = (numKTiles < STAGES) ? numKTiles : STAGES;
    if (tid == 0) {
        for (int s = 0; s < prelude; s++) {
            int kc = s * TILE_K;
            char* stage = smem + s * SMEM_STAGE;
            mbarrier_arrive_expect_tx(&mbar[s],
                A_STAGE_BYTES + B_STAGE_BYTES);
            tma_load_2d(stage, &tma_A, kc, mBase, &mbar[s]);
            tma_load_2d(stage + A_STAGE_BYTES,
                        &tma_B, kc, nBase, &mbar[s]);
            tma_load_2d(stage + A_STAGE_BYTES + B_SUB_BYTES,
                        &tma_B, kc, nBase + N_SUB, &mbar[s]);
        }
    }

    /* ---- Prologue: first K-tile with scale_D=0 (overwrite) ---- */
    {
        mbarrier_wait_parity(&mbar[0], 0);
        char* stage = smem;
        uint64_t da_base = make_gmma_desc(stage + wg_a_offset);
        uint64_t db_base = make_gmma_desc(stage + A_STAGE_BYTES);
        wgmma_tile_k64(acc, da_base, db_base,
                       gmma_desc_advance(da_base, 2), gmma_desc_advance(db_base, 2),
                       gmma_desc_advance(da_base, 4), gmma_desc_advance(db_base, 4),
                       gmma_desc_advance(da_base, 6), gmma_desc_advance(db_base, 6),
                       0);  /* scale_D = 0: overwrite accumulators */
    }

    /* ---- Mainloop: K-tiles 1..numKTiles-1, scale_D=1 (accumulate) ----
     *
     * CuTe DSL pattern: wait_data → WGMMA → commit → wait_group(1) → TMA
     * wait_group(1) ensures PREVIOUS group's SMEM reads are done before TMA
     * overwrites that buffer. Current group's WGMMA overlaps with next TMA.
     */
    for (int kb = 1; kb < numKTiles; kb++) {
        int sc = kb % STAGES;
        int sl = (kb + STAGES - 1) % STAGES;
        int phase = (kb / STAGES) & 1;

        /* Wait for TMA data for current K-tile */
        mbarrier_wait_parity(&mbar[sc], phase);

        /* WGMMA: always accumulate (scale_D=1) */
        char* stage = smem + sc * SMEM_STAGE;
        uint64_t da_base = make_gmma_desc(stage + wg_a_offset);
        uint64_t db_base = make_gmma_desc(stage + A_STAGE_BYTES);
        wgmma_tile_k64(acc, da_base, db_base,
                       gmma_desc_advance(da_base, 2), gmma_desc_advance(db_base, 2),
                       gmma_desc_advance(da_base, 4), gmma_desc_advance(db_base, 4),
                       gmma_desc_advance(da_base, 6), gmma_desc_advance(db_base, 6),
                       1);  /* scale_D = 1: accumulate */

        /* Wait for PREVIOUS group — ensures its SMEM reads are done */
        wgmma_wait_group_1();

        /* Producer: TMA for next stage (safe now — previous group done) */
        if (tid == 0 && kb + STAGES - 1 < numKTiles) {
            int nk = (kb + STAGES - 1) * TILE_K;
            char* next_stage = smem + sl * SMEM_STAGE;
            mbarrier_arrive_expect_tx(&mbar[sl],
                A_STAGE_BYTES + B_STAGE_BYTES);
            tma_load_2d(next_stage, &tma_A, nk, mBase, &mbar[sl]);
            tma_load_2d(next_stage + A_STAGE_BYTES,
                        &tma_B, nk, nBase, &mbar[sl]);
            tma_load_2d(next_stage + A_STAGE_BYTES + B_SUB_BYTES,
                        &tma_B, nk, nBase + N_SUB, &mbar[sl]);
        }
    }

    /* Wait for last WGMMA group */
    wgmma_wait_group_0();

        /* ---- Epilogue: write accumulators to global memory ---- */
        /*
         * WGMMA m64n256k16 F32 output fragment layout:
         *
         *   For register d[i] (i=0..127):
         *     half  = (i >> 1) & 1       (0 or 1 — top/bottom 8-row group)
         *     pair4 = i >> 2             (0..31 — which 8-column chunk)
         *     sub2  = i & 1              (0 or 1 — even/odd column within pair)
         *
         *     row = warp_in_wg * 16 + half * 8 + groupID
         *     col = pair4 * 8 + sub2 + thread_in_group * 2
         *
         *   Registers d[4*p + 2*h] and d[4*p + 2*h + 1] are adjacent columns.
         */

        /* Vectorized bfloat162 stores */
        #pragma unroll
        for (int j = 0; j < 64; j++) {
            int p4 = j / 2;    /* pair4 index (0..31) — column group */
            int h  = j % 2;    /* half (0 or 1) — row group */

            int row = mBase + wg_id * 64 + warp_in_wg * 16 + h * 8 + groupID;
            int col = nBase + p4 * 8 + thread_in_group * 2;

            int idx = 4 * p4 + 2 * h;  /* base register index */

            if (row < M && col + 1 < N) {
                *reinterpret_cast<__nv_bfloat162*>(&C[row * N + col]) =
                    __floats2bfloat162_rn(acc[idx], acc[idx + 1]);
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

    /* Skip sizes too small for the 128×256 tile */
    if (M < TILE_M || N < TILE_N || K < TILE_K) {
        fprintf(stderr, "Matrix too small for 128x256x64 tile: M=%d N=%d K=%d\n",
                M, N, K);
        return -4;
    }

    /* --- Transpose B: (K, N) row-major → (N, K) row-major --- */
    size_t B_elems = (size_t)K * N;
    if (s_B_t_size < B_elems * sizeof(__nv_bfloat16)) {
        if (s_B_t) cudaFree(s_B_t);
        cudaMalloc(&s_B_t, B_elems * sizeof(__nv_bfloat16));
        s_B_t_size = B_elems * sizeof(__nv_bfloat16);
        s_B_last = nullptr;  /* force re-transpose */
    }
    if (s_B_last != B) {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (K + 31) / 32);
        transpose_bf16<<<grid, block, 0, stream>>>(B, s_B_t, K, N);
        s_B_last = B;
    }

    /* --- TMA descriptors --- */
    CUtensorMap tma_A, tma_B;

    /* A: global (K, M) with K contiguous — row-major A viewed as (K-cols, M-rows) */
    {
        cuuint64_t dims[2]    = {(cuuint64_t)K, (cuuint64_t)M};
        cuuint64_t strides[1] = {(cuuint64_t)K * 2};
        cuuint32_t box[2]     = {(cuuint32_t)TILE_K, (cuuint32_t)TILE_M};
        cuuint32_t elem[2]    = {1, 1};
        if (s_encodeTiled(&tma_A, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)A,
                dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) return -2;
    }

    /* B_t: global (K, N) with K contiguous — transposed B viewed as (K-cols, N-rows) */
    {
        cuuint64_t dims[2]    = {(cuuint64_t)K, (cuuint64_t)N};
        cuuint64_t strides[1] = {(cuuint64_t)K * 2};
        cuuint32_t box[2]     = {(cuuint32_t)TILE_K, (cuuint32_t)N_SUB};
        cuuint32_t elem[2]    = {1, 1};
        if (s_encodeTiled(&tma_B, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)s_B_t,
                dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) return -3;
    }

    /* Configure dynamic SMEM */
    static bool cfg_done = false;
    if (!cfg_done) {
        cudaFuncSetAttribute(wgmma_matmul,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        cfg_done = true;
    }

    /* Launch: non-persistent, one block per output tile */
    int nTilesM = M / TILE_M;
    int nTilesN = N / TILE_N;
    int totalTiles = nTilesM * nTilesN;

    wgmma_matmul<<<totalTiles, THREADS, SMEM_BYTES, stream>>>(
        C, M, N, K, totalTiles, nTilesN, tma_A, tma_B);

    return 0;
}
