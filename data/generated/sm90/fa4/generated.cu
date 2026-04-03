/*
 * Flash Attention forward pass — TMA + wgmma, BF16, SM90a (H100).
 *
 * Architecture:
 *   384 threads = 3 warp groups (WG0=producer, WG1=consumer0, WG2=consumer1)
 *   Producer (warps 0-3, 128 threads, 24 regs): TMA loads Q/K/V
 *   Consumer0 (warps 4-7, 128 threads, 240 regs): wgmma QK/PV + softmax (Q rows 0-63)
 *   Consumer1 (warps 8-11, 128 threads, 240 regs): wgmma QK/PV + softmax (Q rows 64-127)
 *
 * Data movement: TMA (cp.async.bulk.tensor) with 128B swizzle
 * Compute: wgmma.mma_async RS variant — m64n128k16 for QK, m64n64k16 for PV
 * Pipeline: 2-stage double-buffered K/V with mbarrier
 *
 * Tile sizes: BLOCK_Q=128, BLOCK_KV=128, DIM=128
 *
 * Inter-WG synchronization: CuTe DSL barrier protocol with intra_wg_overlap.
 *   Named barrier 2 = WarpSchedulerWG0 (consumer0's barrier)
 *   Named barrier 3 = WarpSchedulerWG1 (consumer1's barrier)
 *   Each WG syncs on its OWN barrier, arrives at the OTHER's.
 *   bar.sync/bar.arrive num_threads = 256 (2 consumer WGs × 128 threads).
 *
 * SMEM layout (split-DIM: each [DIM=128,BLOCK] tile → two [64,BLOCK] halves):
 *   Q (BLOCK_Q=128):
 *     Q_lo:    [64, 128] = 16KB at SMEM + 0
 *     Q_hi:    [64, 128] = 16KB at SMEM + 16KB
 *   K stage s:
 *     K_lo:  [64, 128] = 16KB at SMEM + 32KB + s*32KB
 *     K_hi:  [64, 128] = 16KB at SMEM + 32KB + s*32KB + 16KB
 *   V stage s:
 *     V_lo:  [64, 128] = 16KB at SMEM + 96KB + s*32KB
 *     V_hi:  [64, 128] = 16KB at SMEM + 96KB + s*32KB + 16KB
 *   mbarriers: 9 × 8B aligned at 160KB
 *
 * TMA uses boxDim=[64, BLOCK] with SWIZZLE_128B.
 * Each row = 64 bf16 = 128 bytes = exactly one 128B swizzle unit.
 *
 * kernel_run contract:
 *   extern "C" int kernel_run(__nv_bfloat16** inputs, int num_inputs,
 *                             __nv_bfloat16** outputs, int num_outputs,
 *                             int n, cudaStream_t stream);
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
/* dlfcn.h no longer needed — using direct -lcuda linking */
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cfloat>

/* ======================================================================
 *  Constants
 * ====================================================================== */

static constexpr int WARP_SIZE   = 32;
static constexpr int BLOCK_Q     = 128;
static constexpr int BLOCK_KV    = 128;
static constexpr int DIM_CONST   = 128;
static constexpr int HALF_DIM    = 64;
static constexpr int NUM_STAGES  = 2;
static constexpr int TB_SIZE     = 384;   /* 3 warp groups */
static constexpr int WG_SIZE     = 128;   /* threads per warp group */

/* Row stride within a half-tile: 64 bf16 = 128 bytes */
static constexpr int HALF_ROW_STRIDE = HALF_DIM * 2;  /* 128 bytes */

/* Q half-tile: [64 cols, BLOCK_Q rows] = [64, 128] = 16KB */
static constexpr int Q_HALF_TILE_BYTES = HALF_DIM * BLOCK_Q * 2;   /* 16384 = 16KB */
static constexpr int Q_FULL_TILE_BYTES = 2 * Q_HALF_TILE_BYTES;    /* 32768 = 32KB */

/* K/V half-tile: [64 cols, BLOCK_KV rows] = [64, 128] = 16KB */
static constexpr int KV_HALF_TILE_BYTES = HALF_DIM * BLOCK_KV * 2; /* 16384 = 16KB */
static constexpr int KV_FULL_TILE_BYTES = 2 * KV_HALF_TILE_BYTES;  /* 32768 = 32KB */

/* Offset to second 64 rows within a KV half-tile (for PV GEMM split) */
static constexpr int KV_ROW_OFFSET_64  = 64 * HALF_ROW_STRIDE;    /* 8192 bytes */

/* Per-WG Q half-tile size: each WG processes 64 of 128 Q rows */
static constexpr int Q_WG_HALF_TILE_BYTES = HALF_DIM * (BLOCK_Q / 2) * 2;  /* 8192 = 8KB */

/* SMEM layout offsets */
static constexpr int SMEM_Q_OFFSET   = 0;                                              /* 0 */
static constexpr int SMEM_K_OFFSET   = Q_FULL_TILE_BYTES;                              /* 32KB */
static constexpr int SMEM_V_OFFSET   = SMEM_K_OFFSET + NUM_STAGES * KV_FULL_TILE_BYTES; /* 96KB */
static constexpr int SMEM_BAR_OFFSET = SMEM_V_OFFSET + NUM_STAGES * KV_FULL_TILE_BYTES; /* 160KB */

/*
 * mbarrier layout (9 mbarriers, 8 bytes each, in SMEM):
 *   0: Q_full          (producer arrives with tx=Q_FULL_TILE_BYTES, consumers wait)
 *   1: K_full[0]       (producer arrives with tx=KV_FULL_TILE_BYTES, consumers wait)
 *   2: K_full[1]
 *   3: V_full[0]       (producer arrives with tx=KV_FULL_TILE_BYTES, consumers wait)
 *   4: V_full[1]
 *   5: K_empty[0]      (consumer arrives, producer waits) — only WG0 signals
 *   6: K_empty[1]
 *   7: V_empty[0]      (consumer arrives, producer waits) — only WG0 signals
 *   8: V_empty[1]
 *
 * Named barrier layout (bar.sync/bar.arrive, separate HW namespace):
 *   0: reserved (__syncthreads)
 *   1: reserved (Epilogue)
 *   2: WarpSchedulerWG0 (consumer0 syncs, consumer1 arrives) — 256 threads
 *   3: WarpSchedulerWG1 (consumer1 syncs, consumer0 arrives) — 256 threads
 */
static constexpr int NUM_BARRIERS   = 9;
static constexpr int SMEM_TOTAL     = SMEM_BAR_OFFSET + NUM_BARRIERS * 8 + 128;

static constexpr int BAR_Q_FULL     = 0;
static constexpr int BAR_K_FULL     = 1;  /* + stage */
static constexpr int BAR_V_FULL     = 3;  /* + stage */
static constexpr int BAR_K_EMPTY    = 5;  /* + stage */
static constexpr int BAR_V_EMPTY    = 7;  /* + stage */

/* Named barrier IDs (bar.sync/bar.arrive hardware, NOT mbarrier).
 * Use high IDs (14,15) to avoid conflicts with compiler-injected
 * warpgroup.arrive barriers which use low IDs. */
static constexpr int BAR_SCHED_WG0  = 14;  /* consumer0's scheduler barrier */
static constexpr int BAR_SCHED_WG1  = 15;  /* consumer1's scheduler barrier */

__device__ __host__ constexpr
int cdiv(int a, int b) { return (a + b - 1) / b; }

/* ======================================================================
 *  Fast math helpers
 * ====================================================================== */

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

/* ======================================================================
 *  mbarrier helpers
 * ====================================================================== */

__device__ __forceinline__
void mbarrier_init(uint64_t* mbar, unsigned count) {
    uint32_t addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(addr), "r"(count));
}

__device__ __forceinline__
void mbarrier_arrive_expect_tx(uint64_t* mbar, unsigned tx_bytes) {
    uint32_t addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                 :: "r"(addr), "r"(tx_bytes));
}

__device__ __forceinline__
void mbarrier_arrive(uint64_t* mbar) {
    uint32_t addr = __cvta_generic_to_shared(mbar);
    asm volatile(
        "{\n"
        "  .reg .b64 state;\n"
        "  mbarrier.arrive.shared.b64 state, [%0];\n"
        "}\n" :: "r"(addr));
}

__device__ __forceinline__
void mbarrier_wait_parity(uint64_t* mbar, unsigned phase) {
    uint32_t addr = __cvta_generic_to_shared(mbar);
    uint32_t result;
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
void mbarrier_inval(uint64_t* mbar) {
    uint32_t addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(addr));
}

/* ======================================================================
 *  Named barrier helpers for inter-WG scheduler synchronization
 *
 *  bar.sync:   block calling threads until `num_threads` have arrived at barrier `id`
 *  bar.arrive: contribute `num_threads` arrival count to barrier `id` without blocking
 *
 *  For 2 consumer WGs (256 threads total), num_threads = 256.
 *  Named barrier IDs 2 (BAR_SCHED_WG0) and 3 (BAR_SCHED_WG1).
 * ====================================================================== */

__device__ __forceinline__
void named_barrier_sync(int barrier_id, int num_threads) {
    asm volatile("bar.sync %0, %1;" :: "r"(barrier_id), "r"(num_threads));
}

__device__ __forceinline__
void named_barrier_arrive(int barrier_id, int num_threads) {
    asm volatile("bar.arrive %0, %1;" :: "r"(barrier_id), "r"(num_threads));
}

/* ======================================================================
 *  TMA load helper
 * ====================================================================== */

__device__ __forceinline__
void tma_load_2d(void* smem_dst, const void* tma_desc,
                 int coord0, int coord1, uint64_t* mbar) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_dst);
    uint32_t mbar_addr = __cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(smem_addr), "l"(tma_desc),
           "r"(coord0), "r"(coord1), "r"(mbar_addr)
        : "memory");
}

/* ======================================================================
 *  wgmma descriptor construction for 128B-swizzled SMEM half-tiles
 *
 *  Each half-tile is [64 cols, BLOCK rows] bf16 with SWIZZLE_128B.
 *  Row stride = 64 bf16 = 128 bytes = one swizzle unit.
 *
 *  GmmaDescriptor bitfield (CUTLASS):
 *    [0:14)   start_address >> 4
 *    [16:30)  leading_byte_offset >> 4
 *    [32:46)  stride_byte_offset >> 4
 *    [49:52)  base_offset (3 bits)
 *    [62:64)  layout_type (1 = SWIZZLE_128B)
 *
 *  For half-tiles [64 cols, N rows]:
 *    leading_byte_offset = 128 (row stride in bytes) -> >> 4 = 8
 *    stride_byte_offset = 8 * 128 = 1024 (stride between groups of 8 rows) -> >> 4 = 64
 * ====================================================================== */

__device__ __forceinline__
uint64_t make_wgmma_desc(uint32_t smem_addr, int base_offset) {
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_addr >> 4) & 0x3FFF);          /* [0:14)  start_address >> 4 */
    desc |= (uint64_t)(HALF_ROW_STRIDE >> 4) << 16;         /* [16:30) LBO >> 4 = 8 */
    desc |= (uint64_t)((8 * HALF_ROW_STRIDE) >> 4) << 32;   /* [32:46) SBO >> 4 = 64 */
    desc |= (uint64_t)(base_offset & 7) << 49;              /* [49:52) base_offset */
    desc |= (uint64_t)(1) << 62;                             /* [62:64) SWIZZLE_128B */
    return desc;
}

/* ======================================================================
 *  wgmma PTX helpers — RS variant m64n64k16
 *
 *  RS variant trailing params: p, scaleA, scaleB, tnspB (4 params)
 *    p=1 (enabled): D += scaleA*A @ scaleB*B (accumulate)
 *    p=0: D = A @ B (overwrite)
 *    scaleA, scaleB: must be 1 or -1
 *    tnspB: 0=non-transposed, 1=transposed
 * ====================================================================== */

/* Accumulate mode: D += A @ B, tnspB=1 (for QK: K transposed) */
__device__ __forceinline__
void wgmma_m64n64k16_acc(float d[32], uint32_t a[4], uint64_t desc_b) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, 1, 0;\n"  /* p=true -> accumulate */
        "  wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "  {%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
        "   %8,  %9,  %10, %11, %12, %13, %14, %15,"
        "   %16, %17, %18, %19, %20, %21, %22, %23,"
        "   %24, %25, %26, %27, %28, %29, %30, %31},"
        "  {%32, %33, %34, %35},"
        "   %36,"
        "   p, 1, 1, 1;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "l"(desc_b)
    );
}

/* Overwrite mode: D = A @ B, tnspB=1 (for first QK k-step) */
__device__ __forceinline__
void wgmma_m64n64k16_new(float d[32], uint32_t a[4], uint64_t desc_b) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, 0, 0;\n"  /* p=false -> overwrite D */
        "  wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "  {%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
        "   %8,  %9,  %10, %11, %12, %13, %14, %15,"
        "   %16, %17, %18, %19, %20, %21, %22, %23,"
        "   %24, %25, %26, %27, %28, %29, %30, %31},"
        "  {%32, %33, %34, %35},"
        "   %36,"
        "   p, 1, 1, 1;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "l"(desc_b)
    );
}

/* Overwrite mode, tnspB=0: D = A @ B (for first PV k-step of each V half) */
__device__ __forceinline__
void wgmma_m64n64k16_new_nt(float d[32], uint32_t a[4], uint64_t desc_b) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, 0, 0;\n"  /* p=false -> overwrite D */
        "  wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "  {%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
        "   %8,  %9,  %10, %11, %12, %13, %14, %15,"
        "   %16, %17, %18, %19, %20, %21, %22, %23,"
        "   %24, %25, %26, %27, %28, %29, %30, %31},"
        "  {%32, %33, %34, %35},"
        "   %36,"
        "   p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "l"(desc_b)
    );
}

/* Accumulate mode, tnspB=0: D += A @ B (for PV: V non-transposed) */
__device__ __forceinline__
void wgmma_m64n64k16_acc_nt(float d[32], uint32_t a[4], uint64_t desc_b) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, 1, 0;\n"  /* p=true -> accumulate */
        "  wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "  {%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
        "   %8,  %9,  %10, %11, %12, %13, %14, %15,"
        "   %16, %17, %18, %19, %20, %21, %22, %23,"
        "   %24, %25, %26, %27, %28, %29, %30, %31},"
        "  {%32, %33, %34, %35},"
        "   %36,"
        "   p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "l"(desc_b)
    );
}

/* ======================================================================
 *  wgmma PTX helpers — RS variant m64n128k16
 *
 *  Same as m64n64k16 but with 64 output registers (16 groups of 4).
 *  Used for QK GEMM where N=128 (full BLOCK_KV in one wgmma call).
 * ====================================================================== */

/* Overwrite mode: D = A @ B, tnspB=1 (for first QK k-step) */
__device__ __forceinline__
void wgmma_m64n128k16_new(float d[64], uint32_t a[4], uint64_t desc_b) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, 0, 0;\n"  /* p=false -> overwrite D */
        "  wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "  {%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
        "   %8,  %9,  %10, %11, %12, %13, %14, %15,"
        "   %16, %17, %18, %19, %20, %21, %22, %23,"
        "   %24, %25, %26, %27, %28, %29, %30, %31,"
        "   %32, %33, %34, %35, %36, %37, %38, %39,"
        "   %40, %41, %42, %43, %44, %45, %46, %47,"
        "   %48, %49, %50, %51, %52, %53, %54, %55,"
        "   %56, %57, %58, %59, %60, %61, %62, %63},"
        "  {%64, %65, %66, %67},"
        "   %68,"
        "   p, 1, 1, 1;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
          "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
          "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
          "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
          "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "l"(desc_b)
    );
}

/* Accumulate mode: D += A @ B, tnspB=1 (for QK: K transposed) */
__device__ __forceinline__
void wgmma_m64n128k16_acc(float d[64], uint32_t a[4], uint64_t desc_b) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, 1, 0;\n"  /* p=true -> accumulate */
        "  wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "  {%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
        "   %8,  %9,  %10, %11, %12, %13, %14, %15,"
        "   %16, %17, %18, %19, %20, %21, %22, %23,"
        "   %24, %25, %26, %27, %28, %29, %30, %31,"
        "   %32, %33, %34, %35, %36, %37, %38, %39,"
        "   %40, %41, %42, %43, %44, %45, %46, %47,"
        "   %48, %49, %50, %51, %52, %53, %54, %55,"
        "   %56, %57, %58, %59, %60, %61, %62, %63},"
        "  {%64, %65, %66, %67},"
        "   %68,"
        "   p, 1, 1, 1;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
          "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
          "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
          "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
          "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "l"(desc_b)
    );
}

/* wgmma fence, commit, wait */
__device__ __forceinline__
void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n");
}

__device__ __forceinline__
void wgmma_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n");
}

__device__ __forceinline__
void wgmma_wait_group_0() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n");
}

__device__ __forceinline__
void wgmma_wait_group_1() {
    asm volatile("wgmma.wait_group.sync.aligned 1;\n");
}

/* ======================================================================
 *  setmaxnreg for warp specialization
 * ====================================================================== */

__device__ __forceinline__
void setmaxnreg_dec_24() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n");
}

__device__ __forceinline__
void setmaxnreg_inc_240() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 240;\n");
}

/* ======================================================================
 *  ldmatrix for Q -> registers (wgmma RS A operand)
 * ====================================================================== */

__device__ __forceinline__
void ldmatrix_x4(uint32_t regs[4], uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "r"(smem_addr));
}

/* ======================================================================
 *  Producer warp group (warps 0-3, threads 0-127)
 *
 *  Loads Q once via TMA (4 half-loads for BLOCK_Q=128), then loops K/V
 *  with 2-stage pipeline. Each tile load = 2 TMA calls (lo + hi halves).
 *  Only thread 0 (elected leader) issues TMA loads.
 * ====================================================================== */

__device__ __noinline__
void producer_warp_group(
    char* smem_base,
    uint64_t* barriers,
    const CUtensorMap* tma_Q,
    const CUtensorMap* tma_K,
    const CUtensorMap* tma_V,
    int q_coord_seq,    /* batch_id * S + q_block_id * BLOCK_Q */
    int batch_seq_off,  /* batch_id * S */
    int coord_head,     /* head_id * DIM_CONST */
    int max_kv_iter,
    int tid_in_wg)
{
    setmaxnreg_dec_24();

    const bool is_leader = (tid_in_wg == 0);

    /* Load Q tile via TMA: 4 half-loads for BLOCK_Q=128
     *   Q_lo rows 0-63:    boxDim=[64,64] at coord (head, q_coord_seq)
     *   Q_lo rows 64-127:  boxDim=[64,64] at coord (head, q_coord_seq+64)
     *   Q_hi rows 0-63:    boxDim=[64,64] at coord (head+64, q_coord_seq)
     *   Q_hi rows 64-127:  boxDim=[64,64] at coord (head+64, q_coord_seq+64)
     *
     * TMA box is [64, 64] (matching boxDim in TMA descriptor).
     * Q_lo covers DIM cols [0:64), Q_hi covers DIM cols [64:128).
     * Within each half, rows 0-63 are WG0's, rows 64-127 are WG1's.
     */
    if (is_leader) {
        mbarrier_arrive_expect_tx(&barriers[BAR_Q_FULL], Q_FULL_TILE_BYTES);

        /* Q_lo rows 0-63: [64, 64] at offset 0 */
        tma_load_2d(smem_base + SMEM_Q_OFFSET,
                    tma_Q, coord_head, q_coord_seq,
                    &barriers[BAR_Q_FULL]);
        /* Q_lo rows 64-127: [64, 64] at offset 8KB */
        tma_load_2d(smem_base + SMEM_Q_OFFSET + Q_WG_HALF_TILE_BYTES,
                    tma_Q, coord_head, q_coord_seq + 64,
                    &barriers[BAR_Q_FULL]);
        /* Q_hi rows 0-63: [64, 64] at offset 16KB */
        tma_load_2d(smem_base + SMEM_Q_OFFSET + Q_HALF_TILE_BYTES,
                    tma_Q, coord_head + HALF_DIM, q_coord_seq,
                    &barriers[BAR_Q_FULL]);
        /* Q_hi rows 64-127: [64, 64] at offset 24KB */
        tma_load_2d(smem_base + SMEM_Q_OFFSET + Q_HALF_TILE_BYTES + Q_WG_HALF_TILE_BYTES,
                    tma_Q, coord_head + HALF_DIM, q_coord_seq + 64,
                    &barriers[BAR_Q_FULL]);
    }

    /* K/V pipeline loop */
    for (int kv_id = 0; kv_id < max_kv_iter; kv_id++) {
        int stage = kv_id % NUM_STAGES;
        int kv_coord_seq = batch_seq_off + kv_id * BLOCK_KV;

        /* Wait for consumer to finish reading previous data in this stage */
        if (kv_id >= NUM_STAGES) {
            int prev_phase = ((kv_id - NUM_STAGES) / NUM_STAGES) & 1;
            mbarrier_wait_parity(&barriers[BAR_K_EMPTY + stage], prev_phase);
        }

        /* Load K tile: 2 half-loads */
        if (is_leader) {
            mbarrier_arrive_expect_tx(&barriers[BAR_K_FULL + stage], KV_FULL_TILE_BYTES);

            /* K_lo: columns [0:64) */
            tma_load_2d(smem_base + SMEM_K_OFFSET + stage * KV_FULL_TILE_BYTES,
                        tma_K, coord_head, kv_coord_seq,
                        &barriers[BAR_K_FULL + stage]);
            /* K_hi: columns [64:128) */
            tma_load_2d(smem_base + SMEM_K_OFFSET + stage * KV_FULL_TILE_BYTES + KV_HALF_TILE_BYTES,
                        tma_K, coord_head + HALF_DIM, kv_coord_seq,
                        &barriers[BAR_K_FULL + stage]);
        }

        /* Wait for consumer to finish reading V in this stage */
        if (kv_id >= NUM_STAGES) {
            int prev_phase = ((kv_id - NUM_STAGES) / NUM_STAGES) & 1;
            mbarrier_wait_parity(&barriers[BAR_V_EMPTY + stage], prev_phase);
        }

        /* Load V tile: 2 half-loads */
        if (is_leader) {
            mbarrier_arrive_expect_tx(&barriers[BAR_V_FULL + stage], KV_FULL_TILE_BYTES);

            /* V_lo: columns [0:64) */
            tma_load_2d(smem_base + SMEM_V_OFFSET + stage * KV_FULL_TILE_BYTES,
                        tma_V, coord_head, kv_coord_seq,
                        &barriers[BAR_V_FULL + stage]);
            /* V_hi: columns [64:128) */
            tma_load_2d(smem_base + SMEM_V_OFFSET + stage * KV_FULL_TILE_BYTES + KV_HALF_TILE_BYTES,
                        tma_V, coord_head + HALF_DIM, kv_coord_seq,
                        &barriers[BAR_V_FULL + stage]);
        }
    }
}

/* ======================================================================
 *  Consumer warp group (2 instances: WG1 handles Q rows 0-63, WG2 handles 64-127)
 *
 *  wgmma.mma_async for QK^T and PV, with online softmax.
 *  Q from SMEM via ldmatrix (RS variant: A=registers).
 *  K, V from SMEM descriptors (B operand).
 *
 *  Split-DIM approach:
 *    QK: S[64,128] = Q[64,128] @ K^T[128,128]
 *      Single n-block using wgmma.m64n128k16 (64 output registers = 128 KV positions).
 *      8 k-steps (4 from Q_lo×K_lo + 4 from Q_hi×K_hi).
 *      S_acc[64] covers all 128 KV positions in 16 groups of 4 registers.
 *
 *    PV: O[64,128] = P[64,128] @ V[128,128]
 *      = [P @ V_lo, P @ V_hi] where each V half is [128,64].
 *      Each V half split into top[64,64] + bot[64,64] (KV rows 0-63 and 64-127).
 *      4 k-steps per KV-half, 2 KV-halves per DIM-half, 2 DIM-halves = 16 wgmma calls.
 *      Uses tnspB=0 with column-advance (ks*32), bot offset = 8192 bytes.
 *
 *  Inter-WG barrier protocol (CuTe DSL style):
 *    - First half-block: NO sync, NO arrive
 *    - Middle blocks: sync own barrier BEFORE QK, arrive other's barrier AFTER PV issued
 *    - Last half-block: NO sync, NO arrive
 *    - mma_init: WG1 primes WG0's barrier once before the KV loop
 * ====================================================================== */

__device__ __forceinline__
void consumer_warp_group(
    char* smem_base,
    uint64_t* barriers,
    nv_bfloat16* O_base,
    int seq_stride,
    int max_kv_iter,
    int q_block_id,
    int is_causal,
    int tid_in_wg,
    int lane_id,
    int wg_idx)       /* 0 for consumer0 (Q rows 0-63), 1 for consumer1 (Q rows 64-127) */
{
    setmaxnreg_inc_240();

    const int warp_in_wg = tid_in_wg / WARP_SIZE;  /* 0..3 */

    /* Scheduler barrier IDs for this WG */
    const int my_sched_bar    = (wg_idx == 0) ? BAR_SCHED_WG0 : BAR_SCHED_WG1;
    const int other_sched_bar = (wg_idx == 0) ? BAR_SCHED_WG1 : BAR_SCHED_WG0;
    const int sched_num_threads = 2 * WG_SIZE;  /* 256: both consumer WGs */

    /* O accumulator: 2 n-halves of m64n64 for DIM=128 output */
    float O_lo[32];  /* columns 0..63 */
    float O_hi[32];  /* columns 64..127 */
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        O_lo[i] = 0.0f;
        O_hi[i] = 0.0f;
    }

    /* Online softmax state per row.
     * Each thread owns 2 rows: row0 = warp*16+lane/4, row1 = row0+8. */
    float rowmax[2]    = {-FLT_MAX, -FLT_MAX};
    float rowsumexp[2] = {0.0f, 0.0f};
    float saved_rescale[2] = {1.0f, 1.0f};  /* deferred O rescale for overlap */

    const float softmax_scale_log2 =
        rsqrtf(static_cast<float>(DIM_CONST)) * 1.4426950408889634f;

    /* Wait for Q to be loaded */
    mbarrier_wait_parity(&barriers[BAR_Q_FULL], 0);

    /* Q SMEM base addresses for this WG's 64 Q rows.
     * Q layout: Q_lo[64, 128] at SMEM+0, Q_hi[64, 128] at SMEM+16KB.
     * WG0 uses rows 0-63 (offset 0), WG1 uses rows 64-127 (offset 8KB within each half).
     */
    int q_smem_wg_offset = wg_idx * Q_WG_HALF_TILE_BYTES;  /* 0 or 8192 */
    uint32_t q_lo_smem = __cvta_generic_to_shared(smem_base + SMEM_Q_OFFSET + q_smem_wg_offset);
    uint32_t q_hi_smem = __cvta_generic_to_shared(smem_base + SMEM_Q_OFFSET + Q_HALF_TILE_BYTES + q_smem_wg_offset);

    /* Causal masking row offset: this WG's Q rows start at wg_idx * 64 within the block */
    const int causal_row_offset = wg_idx * 64;

    /* ================================================================
     * Main KV loop with INTRA-WARPGROUP OVERLAP + INTER-WG BARRIERS
     *
     * Intra-WG overlap: QK GEMM for block N overlaps with PV GEMM for block N-1.
     * Inter-WG barriers: CuTe DSL protocol for 2 consumer WGs.
     *
     * Structure:
     *   mma_init:         WG1 primes WG0's barrier
     *   First half-block: QK[0] + softmax + pack P (NO barriers)
     *   Middle blocks:    sync own, QK[n], rescale, PV[n-1], arrive other, softmax, pack P
     *   Last half-block:  rescale O, PV[last] (NO barriers)
     * ================================================================ */

    /* Saved P registers for PV of the *previous* KV block */
    uint32_t saved_p_regs[8][4];  /* [ks][4] for 8 k-steps (128 KV positions / 16) */
    int prev_v_stage = 0;
    int prev_v_phase = 0;

    /* Helper macros (inlined by compiler) */

    /*
     * QK GEMM: S[m64, n128] = Q[m64, k128] @ K^T[k128, n128]
     */
    #define DO_QK_GEMM(kv_id_arg, stage_arg, is_first)                          \
    {                                                                            \
        uint32_t kl = __cvta_generic_to_shared(                                  \
            smem_base + SMEM_K_OFFSET + (stage_arg) * KV_FULL_TILE_BYTES);      \
        uint32_t kh = kl + KV_HALF_TILE_BYTES;                                  \
        wgmma_fence();                                                           \
        for (int kg = 0; kg < 8; kg++) {                                         \
            int khh = kg % 4;                                                    \
            uint32_t qb = (kg < 4) ? q_lo_smem : q_hi_smem;                     \
            uint32_t r = warp_in_wg * 16 + (lane_id % 16);                      \
            uint32_t lc = khh * 2 + (lane_id / 16);                             \
            uint32_t pc = lc ^ (r & 7);                                          \
            uint32_t qa = qb + r * HALF_ROW_STRIDE + pc * 16;                   \
            uint32_t qr[4]; ldmatrix_x4(qr, qa);                                \
            uint32_t ks = (kg < 4) ? kl : kh;                                   \
            uint32_t ksk = ks + khh * 32;                                        \
            uint64_t kd = make_wgmma_desc(ksk, 0);                              \
            if (kg == 0)                                                         \
                wgmma_m64n128k16_new(S_acc, qr, kd);                            \
            else                                                                 \
                wgmma_m64n128k16_acc(S_acc, qr, kd);                            \
        }                                                                        \
        wgmma_commit_group();                                                    \
    }

    /*
     * PV GEMM: O[m64, n128] = P[m64, k128] @ V[k128, n128]
     */
    #define DO_PV_GEMM(v_stage_arg)                                              \
    {                                                                            \
        uint32_t vl = __cvta_generic_to_shared(                                  \
            smem_base + SMEM_V_OFFSET + (v_stage_arg) * KV_FULL_TILE_BYTES);    \
        uint32_t vh = vl + KV_HALF_TILE_BYTES;                                  \
        wgmma_fence();                                                           \
        /* O_lo: DIM cols 0-63 */                                               \
        /* KV rows 0-63 (P_regs 0-3) */                                        \
        for (int ks = 0; ks < 4; ks++) {                                         \
            uint64_t vd = make_wgmma_desc(vl + ks * 32, 0);                    \
            wgmma_m64n64k16_acc_nt(O_lo, saved_p_regs[ks], vd);               \
        }                                                                        \
        /* KV rows 64-127 (P_regs 4-7) */                                      \
        for (int ks = 0; ks < 4; ks++) {                                         \
            uint64_t vd = make_wgmma_desc(vl + KV_ROW_OFFSET_64 + ks * 32, 0); \
            wgmma_m64n64k16_acc_nt(O_lo, saved_p_regs[ks + 4], vd);           \
        }                                                                        \
        /* O_hi: DIM cols 64-127 */                                             \
        /* KV rows 0-63 (P_regs 0-3) */                                        \
        for (int ks = 0; ks < 4; ks++) {                                         \
            uint64_t vd = make_wgmma_desc(vh + ks * 32, 0);                    \
            wgmma_m64n64k16_acc_nt(O_hi, saved_p_regs[ks], vd);               \
        }                                                                        \
        /* KV rows 64-127 (P_regs 4-7) */                                      \
        for (int ks = 0; ks < 4; ks++) {                                         \
            uint64_t vd = make_wgmma_desc(vh + KV_ROW_OFFSET_64 + ks * 32, 0); \
            wgmma_m64n64k16_acc_nt(O_hi, saved_p_regs[ks + 4], vd);           \
        }                                                                        \
        wgmma_commit_group();                                                    \
    }

    /*
     * Softmax over S_acc[64] = 128 KV positions (16 groups of 4 regs).
     * Each thread owns 2 rows. Per row, 16 groups x 2 values = 32 values.
     * Uses causal_row_offset to adjust row indices for this WG's Q rows.
     */
    #define DO_SOFTMAX(kv_id_arg, is_first)                                      \
    {                                                                            \
        for (int i = 0; i < 64; i++) S_acc[i] *= softmax_scale_log2;            \
        if (is_causal) {                                                         \
            int r0 = q_block_id*BLOCK_Q + causal_row_offset + warp_in_wg*16 + (lane_id/4); \
            int r1 = r0 + 8;                                                    \
            int kvs = (kv_id_arg) * BLOCK_KV;                                   \
            for (int g = 0; g < 16; g++) {                                       \
                int c0 = kvs + g*8 + (lane_id%4)*2;                             \
                int c1 = c0 + 1;                                                \
                if (c0 > r0) S_acc[g*4+0] = -FLT_MAX;                          \
                if (c1 > r0) S_acc[g*4+1] = -FLT_MAX;                          \
                if (c0 > r1) S_acc[g*4+2] = -FLT_MAX;                          \
                if (c1 > r1) S_acc[g*4+3] = -FLT_MAX;                          \
            }                                                                    \
        }                                                                        \
        float nm[2] = {-FLT_MAX, -FLT_MAX};                                    \
        for (int g = 0; g < 16; g++) {                                           \
            nm[0] = fmaxf(nm[0], fmaxf(S_acc[g*4+0], S_acc[g*4+1]));           \
            nm[1] = fmaxf(nm[1], fmaxf(S_acc[g*4+2], S_acc[g*4+3]));           \
        }                                                                        \
        nm[0] = fmaxf(nm[0], __shfl_xor_sync(0xFFFFFFFF, nm[0], 1));           \
        nm[0] = fmaxf(nm[0], __shfl_xor_sync(0xFFFFFFFF, nm[0], 2));           \
        nm[1] = fmaxf(nm[1], __shfl_xor_sync(0xFFFFFFFF, nm[1], 1));           \
        nm[1] = fmaxf(nm[1], __shfl_xor_sync(0xFFFFFFFF, nm[1], 2));           \
        if (!(is_first)) { nm[0] = fmaxf(nm[0], rowmax[0]); nm[1] = fmaxf(nm[1], rowmax[1]); } \
        float rs[2];                                                             \
        rs[0] = (is_first) ? 1.0f : fast_exp2f(rowmax[0] - nm[0]);             \
        rs[1] = (is_first) ? 1.0f : fast_exp2f(rowmax[1] - nm[1]);             \
        if (!(is_first)) {                                                       \
            for (int g = 0; g < 8; g++) {                                        \
                O_lo[g*4+0]*=rs[0]; O_lo[g*4+1]*=rs[0];                        \
                O_lo[g*4+2]*=rs[1]; O_lo[g*4+3]*=rs[1];                        \
                O_hi[g*4+0]*=rs[0]; O_hi[g*4+1]*=rs[0];                        \
                O_hi[g*4+2]*=rs[1]; O_hi[g*4+3]*=rs[1];                        \
            }                                                                    \
        }                                                                        \
        rowmax[0] = nm[0]; rowmax[1] = nm[1];                                   \
        float ns[2] = {0.0f, 0.0f};                                             \
        for (int g = 0; g < 16; g++) {                                           \
            S_acc[g*4+0] = fast_exp2f(S_acc[g*4+0] - rowmax[0]);               \
            S_acc[g*4+1] = fast_exp2f(S_acc[g*4+1] - rowmax[0]);               \
            S_acc[g*4+2] = fast_exp2f(S_acc[g*4+2] - rowmax[1]);               \
            S_acc[g*4+3] = fast_exp2f(S_acc[g*4+3] - rowmax[1]);               \
            ns[0] += S_acc[g*4+0] + S_acc[g*4+1];                              \
            ns[1] += S_acc[g*4+2] + S_acc[g*4+3];                              \
        }                                                                        \
        ns[0] += __shfl_xor_sync(0xFFFFFFFF, ns[0], 1);                         \
        ns[0] += __shfl_xor_sync(0xFFFFFFFF, ns[0], 2);                         \
        ns[1] += __shfl_xor_sync(0xFFFFFFFF, ns[1], 1);                         \
        ns[1] += __shfl_xor_sync(0xFFFFFFFF, ns[1], 2);                         \
        rowsumexp[0] = rowsumexp[0] * rs[0] + ns[0];                            \
        rowsumexp[1] = rowsumexp[1] * rs[1] + ns[1];                            \
    }

    /* Same as DO_SOFTMAX but saves rescale factors to saved_rescale[]
     * instead of immediately applying to O_lo/O_hi. Used in overlap path
     * where O is being written by PV GEMM in tensor cores. */
    #define DO_SOFTMAX_NO_RESCALE(kv_id_arg, is_first)                           \
    {                                                                            \
        for (int i = 0; i < 64; i++) S_acc[i] *= softmax_scale_log2;            \
        if (is_causal) {                                                         \
            int r0 = q_block_id*BLOCK_Q + causal_row_offset + warp_in_wg*16 + (lane_id/4); \
            int r1 = r0 + 8;                                                    \
            int kvs = (kv_id_arg) * BLOCK_KV;                                   \
            for (int g = 0; g < 16; g++) {                                       \
                int c0 = kvs + g*8 + (lane_id%4)*2;                             \
                int c1 = c0 + 1;                                                \
                if (c0 > r0) S_acc[g*4+0] = -FLT_MAX;                          \
                if (c1 > r0) S_acc[g*4+1] = -FLT_MAX;                          \
                if (c0 > r1) S_acc[g*4+2] = -FLT_MAX;                          \
                if (c1 > r1) S_acc[g*4+3] = -FLT_MAX;                          \
            }                                                                    \
        }                                                                        \
        float nm[2] = {-FLT_MAX, -FLT_MAX};                                    \
        for (int g = 0; g < 16; g++) {                                           \
            nm[0] = fmaxf(nm[0], fmaxf(S_acc[g*4+0], S_acc[g*4+1]));           \
            nm[1] = fmaxf(nm[1], fmaxf(S_acc[g*4+2], S_acc[g*4+3]));           \
        }                                                                        \
        nm[0] = fmaxf(nm[0], __shfl_xor_sync(0xFFFFFFFF, nm[0], 1));           \
        nm[0] = fmaxf(nm[0], __shfl_xor_sync(0xFFFFFFFF, nm[0], 2));           \
        nm[1] = fmaxf(nm[1], __shfl_xor_sync(0xFFFFFFFF, nm[1], 1));           \
        nm[1] = fmaxf(nm[1], __shfl_xor_sync(0xFFFFFFFF, nm[1], 2));           \
        nm[0] = fmaxf(nm[0], rowmax[0]); nm[1] = fmaxf(nm[1], rowmax[1]);      \
        float rs[2];                                                             \
        rs[0] = fast_exp2f(rowmax[0] - nm[0]);                                  \
        rs[1] = fast_exp2f(rowmax[1] - nm[1]);                                  \
        saved_rescale[0] = rs[0]; saved_rescale[1] = rs[1];                     \
        rowmax[0] = nm[0]; rowmax[1] = nm[1];                                   \
        float ns[2] = {0.0f, 0.0f};                                             \
        for (int g = 0; g < 16; g++) {                                           \
            S_acc[g*4+0] = fast_exp2f(S_acc[g*4+0] - rowmax[0]);               \
            S_acc[g*4+1] = fast_exp2f(S_acc[g*4+1] - rowmax[0]);               \
            S_acc[g*4+2] = fast_exp2f(S_acc[g*4+2] - rowmax[1]);               \
            S_acc[g*4+3] = fast_exp2f(S_acc[g*4+3] - rowmax[1]);               \
            ns[0] += S_acc[g*4+0] + S_acc[g*4+1];                              \
            ns[1] += S_acc[g*4+2] + S_acc[g*4+3];                              \
        }                                                                        \
        ns[0] += __shfl_xor_sync(0xFFFFFFFF, ns[0], 1);                         \
        ns[0] += __shfl_xor_sync(0xFFFFFFFF, ns[0], 2);                         \
        ns[1] += __shfl_xor_sync(0xFFFFFFFF, ns[1], 1);                         \
        ns[1] += __shfl_xor_sync(0xFFFFFFFF, ns[1], 2);                         \
        rowsumexp[0] = rowsumexp[0] * rs[0] + ns[0];                            \
        rowsumexp[1] = rowsumexp[1] * rs[1] + ns[1];                            \
    }

    /*
     * Pack P registers for wgmma RS A operand.
     * 8 k-steps for K_PV = BLOCK_KV = 128:
     *   ks 0-7: from S_acc groups (ks*2) and (ks*2+1)
     *   S_acc has 16 groups covering 128 KV positions continuously.
     */
    #define PACK_P_REGS()                                                        \
    {                                                                            \
        for (int ks = 0; ks < 8; ks++) {                                         \
            nv_bfloat162 t;                                                      \
            int g0 = ks*2, g1 = ks*2+1;                                         \
            t = __float22bfloat162_rn({S_acc[g0*4+0], S_acc[g0*4+1]});          \
            saved_p_regs[ks][0] = reinterpret_cast<uint32_t&>(t);               \
            t = __float22bfloat162_rn({S_acc[g1*4+0], S_acc[g1*4+1]});          \
            saved_p_regs[ks][1] = reinterpret_cast<uint32_t&>(t);               \
            t = __float22bfloat162_rn({S_acc[g0*4+2], S_acc[g0*4+3]});          \
            saved_p_regs[ks][2] = reinterpret_cast<uint32_t&>(t);               \
            t = __float22bfloat162_rn({S_acc[g1*4+2], S_acc[g1*4+3]});          \
            saved_p_regs[ks][3] = reinterpret_cast<uint32_t&>(t);               \
        }                                                                        \
    }

    float S_acc[64];  /* QK output: 128 KV positions in 16 groups of 4 regs */

    /* No mma_init priming needed — we use bar.sync (symmetric) instead of
     * bar.sync + bar.arrive (asymmetric), which avoids re-entrance races. */

    /* === FIRST HALF-BLOCK: QK[0] + softmax + pack P === */
    /* NO sync, NO arrive — both WGs proceed independently */
    {
        int stage = 0;
        int phase = 0;
        mbarrier_wait_parity(&barriers[BAR_K_FULL + stage], phase);
        DO_QK_GEMM(0, stage, true);
        wgmma_wait_group_0();
        /* Only WG0 signals K_empty (both WGs are roughly synchronized) */
        if (wg_idx == 0 && tid_in_wg == 0)
            mbarrier_arrive(&barriers[BAR_K_EMPTY + stage]);
        DO_SOFTMAX(0, true);
        PACK_P_REGS();
        prev_v_stage = stage;
        prev_v_phase = phase;
    }

    /* === MIDDLE BLOCKS: overlap QK[n] with PV[n-1], inter-WG barriers === */
    for (int kv_id = 1; kv_id < max_kv_iter; kv_id++) {
        int stage = kv_id % NUM_STAGES;
        int phase = (kv_id / NUM_STAGES) & 1;

        /* Clone V state (save for PV), advance K state (for QK) */
        int this_v_stage = prev_v_stage;
        int this_v_phase = prev_v_phase;

        /* Wait for K[cur] to be loaded */
        mbarrier_wait_parity(&barriers[BAR_K_FULL + stage], phase);

        /* Issue QK[cur] (commit group) */
        DO_QK_GEMM(kv_id, stage, false);

        /* === Rescale O BEFORE PV starts (O is idle, safe to modify) === */
        {
            float rs0 = saved_rescale[0], rs1 = saved_rescale[1];
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                O_lo[g*4+0]*=rs0; O_lo[g*4+1]*=rs0;
                O_lo[g*4+2]*=rs1; O_lo[g*4+3]*=rs1;
                O_hi[g*4+0]*=rs0; O_hi[g*4+1]*=rs0;
                O_hi[g*4+2]*=rs1; O_hi[g*4+3]*=rs1;
            }
        }

        /* Wait for V[prev] to be loaded */
        mbarrier_wait_parity(&barriers[BAR_V_FULL + this_v_stage], this_v_phase);

        /* Issue PV[prev] (commit group) — overlaps with QK in tensor cores */
        DO_PV_GEMM(this_v_stage);

        /* Wait for QK to finish (group 1). PV may still be running. */
        wgmma_wait_group_1();

        /* Softmax on QK result — does NOT touch O_lo/O_hi */
        DO_SOFTMAX_NO_RESCALE(kv_id, false);

        /* Wait for PV to finish */
        wgmma_wait_group_0();

        /* --- Inter-WG sync: both WGs synchronize AFTER PV is done.
         * This ensures both WGs have consumed K[cur] and V[prev]
         * before the producer can reuse those buffers.
         * Use a single shared barrier (BAR_SCHED_WG0=14) with bar.sync. */
        named_barrier_sync(BAR_SCHED_WG0, sched_num_threads);

        /* Release K[cur] and V[prev] — only WG0 signals */
        if (wg_idx == 0 && tid_in_wg == 0) {
            mbarrier_arrive(&barriers[BAR_K_EMPTY + stage]);
            mbarrier_arrive(&barriers[BAR_V_EMPTY + this_v_stage]);
        }

        /* Pack P for next PV iteration */
        PACK_P_REGS();
        prev_v_stage = stage;
        prev_v_phase = phase;
    }

    /* === LAST HALF-BLOCK: rescale O, then PV[last] === */
    /* NO sync, NO arrive */
    {
        /* Apply the saved rescale from the last softmax */
        float rs0 = saved_rescale[0], rs1 = saved_rescale[1];
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            O_lo[g*4+0]*=rs0; O_lo[g*4+1]*=rs0;
            O_lo[g*4+2]*=rs1; O_lo[g*4+3]*=rs1;
            O_hi[g*4+0]*=rs0; O_hi[g*4+1]*=rs0;
            O_hi[g*4+2]*=rs1; O_hi[g*4+3]*=rs1;
        }

        mbarrier_wait_parity(&barriers[BAR_V_FULL + prev_v_stage], prev_v_phase);
        DO_PV_GEMM(prev_v_stage);
        wgmma_wait_group_0();
        /* Only WG0 signals V_empty */
        if (wg_idx == 0 && tid_in_wg == 0)
            mbarrier_arrive(&barriers[BAR_V_EMPTY + prev_v_stage]);
    }

    #undef DO_QK_GEMM
    #undef DO_PV_GEMM
    #undef DO_SOFTMAX
    #undef DO_SOFTMAX_NO_RESCALE
    #undef PACK_P_REGS

    /* ---- Normalize O and write to global memory ---- */
    float inv_sum0 = fast_rcp(rowsumexp[0]);
    float inv_sum1 = fast_rcp(rowsumexp[1]);

    /* This WG's rows within the Q block: wg_idx * 64 + warp/lane offsets */
    int base_row0 = causal_row_offset + warp_in_wg * 16 + (lane_id / 4);
    int base_row1 = base_row0 + 8;

    /* Write O_lo (columns 0..63) */
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        int col = g * 8 + (lane_id % 4) * 2;

        float v0 = O_lo[g*4+0] * inv_sum0;
        float v1 = O_lo[g*4+1] * inv_sum0;
        float v2 = O_lo[g*4+2] * inv_sum1;
        float v3 = O_lo[g*4+3] * inv_sum1;

        reinterpret_cast<nv_bfloat162*>(
            O_base + base_row0 * seq_stride + col)[0] =
            __float22bfloat162_rn({v0, v1});
        reinterpret_cast<nv_bfloat162*>(
            O_base + base_row1 * seq_stride + col)[0] =
            __float22bfloat162_rn({v2, v3});
    }

    /* Write O_hi (columns 64..127) */
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        int col = 64 + g * 8 + (lane_id % 4) * 2;

        float v0 = O_hi[g*4+0] * inv_sum0;
        float v1 = O_hi[g*4+1] * inv_sum0;
        float v2 = O_hi[g*4+2] * inv_sum1;
        float v3 = O_hi[g*4+3] * inv_sum1;

        reinterpret_cast<nv_bfloat162*>(
            O_base + base_row0 * seq_stride + col)[0] =
            __float22bfloat162_rn({v0, v1});
        reinterpret_cast<nv_bfloat162*>(
            O_base + base_row1 * seq_stride + col)[0] =
            __float22bfloat162_rn({v2, v3});
    }
}

/* ======================================================================
 *  Kernel entry point
 * ====================================================================== */

__launch_bounds__(TB_SIZE, 1)
__global__
void flash_attention_tma_wgmma(
    nv_bfloat16* O,
    int B_param, int S, int H,
    int len_q, int len_kv,
    int is_causal,
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V)
{
    extern __shared__ char smem[];

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    const int num_q_blocks = cdiv(len_q, BLOCK_Q);
    const int bs_id        = bid / num_q_blocks;
    const int q_block_id   = bid % num_q_blocks;
    const int batch_id     = bs_id / H;
    const int head_id      = bs_id % H;
    const int seq_stride   = H * DIM_CONST;

    /* TMA coordinates: (innermost=col_offset, outermost=row_offset) */
    int coord_head    = head_id * DIM_CONST;            /* column offset */
    int q_coord_seq   = batch_id * S + q_block_id * BLOCK_Q;  /* row offset for Q */
    int batch_seq_off = batch_id * S;                   /* row offset for K/V base */

    nv_bfloat16* O_base = O + (size_t)batch_id * S * seq_stride
                            + head_id * DIM_CONST
                            + (size_t)q_block_id * BLOCK_Q * seq_stride;

    /* Initialize all mbarriers (thread 0 only) */
    uint64_t* barriers = reinterpret_cast<uint64_t*>(smem + SMEM_BAR_OFFSET);
    if (tid == 0) {
        mbarrier_init(&barriers[BAR_Q_FULL], 1);
        for (int s = 0; s < NUM_STAGES; s++) {
            mbarrier_init(&barriers[BAR_K_FULL + s], 1);
            mbarrier_init(&barriers[BAR_V_FULL + s], 1);
        }
        for (int s = 0; s < NUM_STAGES; s++) {
            mbarrier_init(&barriers[BAR_K_EMPTY + s], 1);
            mbarrier_init(&barriers[BAR_V_EMPTY + s], 1);
        }
        /* Scheduler barriers are named barriers (bar.sync/bar.arrive),
         * not mbarriers. They do not need explicit initialization. */
    }
    __syncthreads();

    const int num_kv_iter = cdiv(len_kv, BLOCK_KV);
    const int max_kv_iter = is_causal
        ? min(num_kv_iter, cdiv((q_block_id + 1) * BLOCK_Q, BLOCK_KV))
        : num_kv_iter;

    const int warp_group = tid / WG_SIZE;  /* 0=producer, 1=consumer0, 2=consumer1 */
    const int tid_in_wg  = tid % WG_SIZE;
    const int lane_id    = tid % WARP_SIZE;

    if (warp_group == 0) {
        producer_warp_group(
            smem, barriers,
            &tma_Q, &tma_K, &tma_V,
            q_coord_seq, batch_seq_off, coord_head,
            max_kv_iter, tid_in_wg);
    } else {
        int wg_idx = warp_group - 1;  /* 0 for WG1 (Q rows 0-63), 1 for WG2 (Q rows 64-127) */
        consumer_warp_group(
            smem, barriers,
            O_base, seq_stride,
            max_kv_iter, q_block_id, is_causal,
            tid_in_wg, lane_id,
            wg_idx);
    }
}

/* ======================================================================
 *  Host-side helpers
 * ====================================================================== */

/* Use cuTensorMapEncodeTiled directly — linked with -lcuda */

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

    int B_val = parse_int_env("CUDA_EXEC_PARAM_BATCH_SIZE", 0);
    int S_val = parse_int_env("CUDA_EXEC_PARAM_SEQ_LEN",    0);
    int H_val = parse_int_env("CUDA_EXEC_PARAM_NUM_HEADS",  0);
    int D_val = parse_int_env("CUDA_EXEC_PARAM_HEAD_DIM",   0);

    if (B_val == 0) B_val = json_int(config_json, "batch_size", 0);
    if (S_val == 0) S_val = json_int(config_json, "seq_len",    0);
    if (H_val == 0) H_val = json_int(config_json, "num_heads",  0);
    if (D_val == 0) D_val = json_int(config_json, "head_dim",   0);

    if (D_val == 0) D_val = 128;
    if (H_val == 0) H_val = 16;
    if (S_val == 0 && B_val == 0 && n > 0) {
        int total_tokens = n / (H_val * D_val);
        B_val = 1;
        S_val = total_tokens / B_val;
    }
    if (B_val == 0 || S_val == 0) return -1;

    bool causal = parse_bool_env("CUDA_EXEC_PARAM_CAUSAL", false);
    if (!causal && config_json)
        causal = json_bool(config_json, "causal", false);

    if (D_val != 128) return -2;

    /* CUDA driver is already initialized by CUDA runtime (cudaMalloc in harness).
     * cuTensorMapEncodeTiled is a host-only function that doesn't need extra init. */

    /*
     * Create TMA tensor descriptors for Q, K, V.
     *
     * Global tensor layout: [B*S, H*D] contiguous in H*D
     *   dim0 (innermost) = column index in H*D
     *   dim1 = row index (sequence position across all batches)
     *   stride[0] = row stride in bytes = H*D * sizeof(bf16)
     *
     * SWIZZLE_128B constraint: boxDim[0] * elem_size must equal 128 bytes.
     * For bf16 (2 bytes): boxDim[0] = 64.
     *
     * One TMA descriptor per matrix with box=[64, BLOCK].
     * Producer issues 4 TMA loads for Q (BLOCK_Q=128, split DIM and split rows):
     *   lo rows 0-63:   coord0 = head*DIM,      coord1 = q_seq
     *   lo rows 64-127: coord0 = head*DIM,      coord1 = q_seq+64
     *   hi rows 0-63:   coord0 = head*DIM+64,   coord1 = q_seq
     *   hi rows 64-127: coord0 = head*DIM+64,   coord1 = q_seq+64
     */
    int seq_stride = H_val * D_val;
    cuuint64_t globalDim[2]    = {(cuuint64_t)seq_stride, (cuuint64_t)(B_val * S_val)};
    cuuint64_t globalStride[1] = {(cuuint64_t)(seq_stride * 2)};  /* bytes per row */
    cuuint32_t elemStride[2]   = {1, 1};

    CUtensorMap tma_Q, tma_K, tma_V;

    /* Q: box [64, 64] — each TMA load covers half-DIM x half-BLOCK_Q */
    {
        cuuint32_t boxDim[2] = {(cuuint32_t)HALF_DIM, (cuuint32_t)(BLOCK_Q / 2)};
        CUresult res = cuTensorMapEncodeTiled(
            &tma_Q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
            (void*)inputs[0],
            globalDim, globalStride, boxDim, elemStride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (res != CUDA_SUCCESS) return -4;
    }

    /* K: box [64, BLOCK_KV] */
    {
        cuuint32_t boxDim[2] = {(cuuint32_t)HALF_DIM, (cuuint32_t)BLOCK_KV};
        CUresult res = cuTensorMapEncodeTiled(
            &tma_K, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
            (void*)inputs[1],
            globalDim, globalStride, boxDim, elemStride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (res != CUDA_SUCCESS) return -5;
    }

    /* V: box [64, BLOCK_KV] */
    {
        cuuint32_t boxDim[2] = {(cuuint32_t)HALF_DIM, (cuuint32_t)BLOCK_KV};
        CUresult res = cuTensorMapEncodeTiled(
            &tma_V, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
            (void*)inputs[2],
            globalDim, globalStride, boxDim, elemStride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (res != CUDA_SUCCESS) return -6;
    }

    /* Launch */
    int num_blocks = B_val * H_val * cdiv(S_val, BLOCK_Q);

    static bool smem_configured = false;
    if (!smem_configured) {
        cudaFuncSetAttribute(flash_attention_tma_wgmma,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
        smem_configured = true;
    }

    flash_attention_tma_wgmma<<<num_blocks, TB_SIZE, SMEM_TOTAL, stream>>>(
        outputs[0],
        B_val, S_val, H_val, S_val, S_val,
        causal ? 1 : 0,
        tma_Q, tma_K, tma_V);

    return 0;
}
