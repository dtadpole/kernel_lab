/*
 * Flash Attention forward pass — TMA + wgmma, BF16, SM90a (H100).
 *
 * Architecture:
 *   256 threads = 2 warp groups (WG0=producer, WG1=consumer)
 *   Producer (warps 0-3, 128 threads, 56 regs): TMA loads Q/K/V
 *   Consumer (warps 4-7, 128 threads, 256 regs): wgmma.mma_async QK/PV + softmax
 *
 * Data movement: TMA (cp.async.bulk.tensor) with 128B swizzle
 * Compute: wgmma.mma_async.sync.aligned.m64n64k16 RS variant
 * Pipeline: 2-stage double-buffered K/V with mbarrier
 *
 * Tile sizes: BLOCK_Q=64, BLOCK_KV=64, DIM=128
 *
 * SMEM layout (split-DIM: each [DIM=128,BLOCK] tile → two [64,BLOCK] halves):
 *   Q_lo:    [64, 64] =  8KB at SMEM + 0
 *   Q_hi:    [64, 64] =  8KB at SMEM + 8KB
 *   K stage s:
 *     K_lo:  [64, 64] =  8KB at SMEM + 16KB + s*16KB
 *     K_hi:  [64, 64] =  8KB at SMEM + 16KB + s*16KB + 8KB
 *   V stage s:
 *     V_lo:  [64, 64] =  8KB at SMEM + 48KB + s*16KB
 *     V_hi:  [64, 64] =  8KB at SMEM + 48KB + s*16KB + 8KB
 *   Barriers: 9 × 8B aligned at 80KB
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
static constexpr int BLOCK_Q     = 64;
static constexpr int BLOCK_KV    = 64;
static constexpr int DIM_CONST   = 128;
static constexpr int HALF_DIM    = 64;
static constexpr int NUM_STAGES  = 2;
static constexpr int TB_SIZE     = 256;   /* 2 warp groups */
static constexpr int WG_SIZE     = 128;   /* threads per warp group */

/* Half-tile: [64 cols, BLOCK rows] in SMEM with 128B swizzle.
 * Each row = 64 bf16 = 128 bytes. */
static constexpr int HALF_TILE_BYTES = HALF_DIM * BLOCK_KV * 2;  /* 8192 = 8KB */
static constexpr int FULL_TILE_BYTES = 2 * HALF_TILE_BYTES;      /* 16384 = 16KB */

/* Row stride within a half-tile: 64 bf16 = 128 bytes */
static constexpr int HALF_ROW_STRIDE = HALF_DIM * 2;  /* 128 bytes */

/* SMEM layout offsets */
static constexpr int SMEM_Q_OFFSET   = 0;                                            /* 0 */
static constexpr int SMEM_K_OFFSET   = FULL_TILE_BYTES;                              /* 16KB */
static constexpr int SMEM_V_OFFSET   = SMEM_K_OFFSET + NUM_STAGES * FULL_TILE_BYTES; /* 48KB */
static constexpr int SMEM_BAR_OFFSET = SMEM_V_OFFSET + NUM_STAGES * FULL_TILE_BYTES; /* 80KB */

/*
 * Barrier layout (9 barriers, 8 bytes each):
 *   0: Q_full        (producer arrives with tx=FULL_TILE_BYTES, consumer waits)
 *   1: K_full[0]     (producer arrives with tx=FULL_TILE_BYTES, consumer waits)
 *   2: K_full[1]
 *   3: V_full[0]     (producer arrives with tx=FULL_TILE_BYTES, consumer waits)
 *   4: V_full[1]
 *   5: K_empty[0]    (consumer arrives, producer waits)
 *   6: K_empty[1]
 *   7: V_empty[0]    (consumer arrives, producer waits)
 *   8: V_empty[1]
 */
static constexpr int NUM_BARRIERS   = 9;
static constexpr int SMEM_TOTAL     = SMEM_BAR_OFFSET + NUM_BARRIERS * 8 + 128;

static constexpr int BAR_Q_FULL     = 0;
static constexpr int BAR_K_FULL     = 1;  /* + stage */
static constexpr int BAR_V_FULL     = 3;  /* + stage */
static constexpr int BAR_K_EMPTY    = 5;  /* + stage */
static constexpr int BAR_V_EMPTY    = 7;  /* + stage */

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
void setmaxnreg_dec_56() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 56;\n");
}

__device__ __forceinline__
void setmaxnreg_inc_256() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 256;\n");
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
 *  Loads Q once via TMA (2 half-loads), then loops K/V with 2-stage
 *  pipeline. Each tile load = 2 TMA calls (lo + hi halves).
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
    setmaxnreg_dec_56();

    const bool is_leader = (tid_in_wg == 0);

    /* Load Q tile via TMA: 2 half-loads */
    if (is_leader) {
        mbarrier_arrive_expect_tx(&barriers[BAR_Q_FULL], FULL_TILE_BYTES);

        /* Q_lo: columns [0:64) */
        tma_load_2d(smem_base + SMEM_Q_OFFSET,
                    tma_Q, coord_head, q_coord_seq,
                    &barriers[BAR_Q_FULL]);
        /* Q_hi: columns [64:128) */
        tma_load_2d(smem_base + SMEM_Q_OFFSET + HALF_TILE_BYTES,
                    tma_Q, coord_head + HALF_DIM, q_coord_seq,
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
            mbarrier_arrive_expect_tx(&barriers[BAR_K_FULL + stage], FULL_TILE_BYTES);

            /* K_lo: columns [0:64) */
            tma_load_2d(smem_base + SMEM_K_OFFSET + stage * FULL_TILE_BYTES,
                        tma_K, coord_head, kv_coord_seq,
                        &barriers[BAR_K_FULL + stage]);
            /* K_hi: columns [64:128) */
            tma_load_2d(smem_base + SMEM_K_OFFSET + stage * FULL_TILE_BYTES + HALF_TILE_BYTES,
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
            mbarrier_arrive_expect_tx(&barriers[BAR_V_FULL + stage], FULL_TILE_BYTES);

            /* V_lo: columns [0:64) */
            tma_load_2d(smem_base + SMEM_V_OFFSET + stage * FULL_TILE_BYTES,
                        tma_V, coord_head, kv_coord_seq,
                        &barriers[BAR_V_FULL + stage]);
            /* V_hi: columns [64:128) */
            tma_load_2d(smem_base + SMEM_V_OFFSET + stage * FULL_TILE_BYTES + HALF_TILE_BYTES,
                        tma_V, coord_head + HALF_DIM, kv_coord_seq,
                        &barriers[BAR_V_FULL + stage]);
        }
    }
}

/* ======================================================================
 *  Consumer warp group (warps 4-7, threads 128-255)
 *
 *  wgmma.mma_async for QK^T and PV, with online softmax.
 *  Q from SMEM via ldmatrix (RS variant: A=registers).
 *  K, V from SMEM descriptors (B operand).
 *
 *  Split-DIM approach:
 *    QK: S[64,64] = Q[64,128] @ K^T[128,64]
 *      = Q_lo[64,64] @ K_lo^T[64,64] + Q_hi[64,64] @ K_hi^T[64,64]
 *      4 k-steps from lo half + 4 k-steps from hi half = 8 total
 *
 *    PV: O[64,128] = P[64,64] @ V[64,128]
 *      = [P @ V_lo, P @ V_hi] where each is [64,64]
 *      4 k-steps each
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
    int lane_id)
{
    setmaxnreg_inc_256();

    const int warp_in_wg = tid_in_wg / WARP_SIZE;  /* 0..3 */

    /*
     * wgmma m64n64k16 output register layout (per thread, 32 floats):
     *   8 groups of 4 regs. Group g (0..7) covers N-columns [g*8, g*8+8).
     *   Within group g:
     *     d[g*4+0]: (row = warp*16 + lane/4,     col = g*8 + (lane%4)*2)
     *     d[g*4+1]: (row = warp*16 + lane/4,     col = g*8 + (lane%4)*2 + 1)
     *     d[g*4+2]: (row = warp*16 + lane/4 + 8, col = g*8 + (lane%4)*2)
     *     d[g*4+3]: (row = warp*16 + lane/4 + 8, col = g*8 + (lane%4)*2 + 1)
     */

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

    /* Q SMEM base addresses for lo and hi halves */
    uint32_t q_lo_smem = __cvta_generic_to_shared(smem_base + SMEM_Q_OFFSET);
    uint32_t q_hi_smem = __cvta_generic_to_shared(smem_base + SMEM_Q_OFFSET + HALF_TILE_BYTES);

    /* ================================================================
     * Main KV loop with INTRA-WARPGROUP OVERLAP
     *
     * Key idea: QK GEMM for block N overlaps with PV GEMM for block N-1.
     * Both are async wgmma calls. We use wgmma.wait_group(1) to wait for
     * only the older group (QK), letting PV continue in the background
     * while we do softmax on the QK result.
     *
     * Structure:
     *   First half-block:  QK[0] only (no previous PV)
     *   Middle blocks:     QK[n] + PV[n-1] overlapped
     *   Last half-block:   PV[last] only (no next QK)
     * ================================================================ */

    /* Saved P registers for PV of the *previous* KV block */
    uint32_t saved_p_regs[4][4];  /* [ks][4] for 4 k-steps */
    int prev_v_stage = 0;
    int prev_v_phase = 0;

    /* Helper lambdas (inlined by compiler) */
    #define DO_QK_GEMM(kv_id_arg, stage_arg, is_first)                          \
    {                                                                            \
        float* S_dst = S_acc;                                                    \
        uint32_t kl = __cvta_generic_to_shared(                                  \
            smem_base + SMEM_K_OFFSET + (stage_arg) * FULL_TILE_BYTES);          \
        uint32_t kh = kl + HALF_TILE_BYTES;                                      \
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
                wgmma_m64n64k16_new(S_dst, qr, kd);                             \
            else                                                                 \
                wgmma_m64n64k16_acc(S_dst, qr, kd);                             \
        }                                                                        \
        wgmma_commit_group();                                                    \
    }

    #define DO_PV_GEMM(v_stage_arg)                                              \
    {                                                                            \
        uint32_t vl = __cvta_generic_to_shared(                                  \
            smem_base + SMEM_V_OFFSET + (v_stage_arg) * FULL_TILE_BYTES);        \
        uint32_t vh = vl + HALF_TILE_BYTES;                                      \
        wgmma_fence();                                                           \
        for (int ks = 0; ks < 4; ks++) {                                         \
            uint64_t vd_lo = make_wgmma_desc(vl + ks * 32, 0);                  \
            wgmma_m64n64k16_acc_nt(O_lo, saved_p_regs[ks], vd_lo);             \
        }                                                                        \
        for (int ks = 0; ks < 4; ks++) {                                         \
            uint64_t vd_hi = make_wgmma_desc(vh + ks * 32, 0);                  \
            wgmma_m64n64k16_acc_nt(O_hi, saved_p_regs[ks], vd_hi);             \
        }                                                                        \
        wgmma_commit_group();                                                    \
    }

    #define DO_SOFTMAX(kv_id_arg, is_first)                                      \
    {                                                                            \
        for (int i = 0; i < 32; i++) S_acc[i] *= softmax_scale_log2;            \
        if (is_causal) {                                                         \
            int r0 = q_block_id*BLOCK_Q + warp_in_wg*16 + (lane_id/4);         \
            int r1 = r0 + 8;                                                    \
            int kvs = (kv_id_arg) * BLOCK_KV;                                   \
            for (int g = 0; g < 8; g++) {                                        \
                int c0 = kvs + g*8 + (lane_id%4)*2;                             \
                int c1 = c0 + 1;                                                \
                if (c0 > r0) S_acc[g*4+0] = -FLT_MAX;                          \
                if (c1 > r0) S_acc[g*4+1] = -FLT_MAX;                          \
                if (c0 > r1) S_acc[g*4+2] = -FLT_MAX;                          \
                if (c1 > r1) S_acc[g*4+3] = -FLT_MAX;                          \
            }                                                                    \
        }                                                                        \
        float nm[2] = {-FLT_MAX, -FLT_MAX};                                    \
        for (int g = 0; g < 8; g++) {                                            \
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
        for (int g = 0; g < 8; g++) {                                            \
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
        for (int i = 0; i < 32; i++) S_acc[i] *= softmax_scale_log2;            \
        if (is_causal) {                                                         \
            int r0 = q_block_id*BLOCK_Q + warp_in_wg*16 + (lane_id/4);         \
            int r1 = r0 + 8;                                                    \
            int kvs = (kv_id_arg) * BLOCK_KV;                                   \
            for (int g = 0; g < 8; g++) {                                        \
                int c0 = kvs + g*8 + (lane_id%4)*2;                             \
                int c1 = c0 + 1;                                                \
                if (c0 > r0) S_acc[g*4+0] = -FLT_MAX;                          \
                if (c1 > r0) S_acc[g*4+1] = -FLT_MAX;                          \
                if (c0 > r1) S_acc[g*4+2] = -FLT_MAX;                          \
                if (c1 > r1) S_acc[g*4+3] = -FLT_MAX;                          \
            }                                                                    \
        }                                                                        \
        float nm[2] = {-FLT_MAX, -FLT_MAX};                                    \
        for (int g = 0; g < 8; g++) {                                            \
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
        for (int g = 0; g < 8; g++) {                                            \
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

    #define PACK_P_REGS()                                                        \
    {                                                                            \
        for (int ks = 0; ks < 4; ks++) {                                         \
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

    float S_acc[32];

    /* === FIRST HALF-BLOCK: QK[0] + softmax, save P for later PV === */
    {
        int stage = 0;
        int phase = 0;
        mbarrier_wait_parity(&barriers[BAR_K_FULL + stage], phase);
        DO_QK_GEMM(0, stage, true);
        wgmma_wait_group_0();
        if (tid_in_wg == 0) mbarrier_arrive(&barriers[BAR_K_EMPTY + stage]);
        DO_SOFTMAX(0, true);
        PACK_P_REGS();
        prev_v_stage = stage;
        prev_v_phase = phase;
    }

    /* === MIDDLE BLOCKS: overlap QK[n] with PV[n-1] === */
    for (int kv_id = 1; kv_id < max_kv_iter; kv_id++) {
        int stage = kv_id % NUM_STAGES;
        int phase = (kv_id / NUM_STAGES) & 1;

        /* Wait for V[prev] and K[cur] */
        mbarrier_wait_parity(&barriers[BAR_V_FULL + prev_v_stage], prev_v_phase);
        mbarrier_wait_parity(&barriers[BAR_K_FULL + stage], phase);

        /* === Rescale O BEFORE PV starts (O is idle, safe to modify) === */
        /* Apply the scale from the PREVIOUS softmax iteration.
         * On the first middle iteration (kv_id=1), the scale is from first-block softmax. */
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

        /* Issue QK[cur] (commit group 1) */
        DO_QK_GEMM(kv_id, stage, false);

        /* Issue PV[prev] (commit group 2) — overlaps with QK in tensor cores.
         * O was just rescaled, so PV accumulates at the correct scale. */
        DO_PV_GEMM(prev_v_stage);

        /* Wait for QK to finish (group 1). PV (group 2) may still be running.
         * S_acc is now readable. O_lo/O_hi are NOT readable (PV writing). */
        wgmma_wait_group_1();

        /* Release K[cur] */
        if (tid_in_wg == 0)
            mbarrier_arrive(&barriers[BAR_K_EMPTY + stage]);

        /* Softmax on QK result — does NOT touch O_lo/O_hi.
         * Computes rescale factors and saves them for next iteration. */
        DO_SOFTMAX_NO_RESCALE(kv_id, false);

        /* Wait for PV to finish — now safe to touch O */
        wgmma_wait_group_0();

        /* Release V[prev] */
        if (tid_in_wg == 0)
            mbarrier_arrive(&barriers[BAR_V_EMPTY + prev_v_stage]);

        /* Pack P for next PV iteration */
        PACK_P_REGS();
        prev_v_stage = stage;
        prev_v_phase = phase;
    }

    /* === LAST HALF-BLOCK: rescale O, then PV[last] === */
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
        if (tid_in_wg == 0)
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

    int base_row0 = warp_in_wg * 16 + (lane_id / 4);
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
    }
    __syncthreads();

    const int num_kv_iter = cdiv(len_kv, BLOCK_KV);
    const int max_kv_iter = is_causal
        ? min(num_kv_iter, cdiv((q_block_id + 1) * BLOCK_Q, BLOCK_KV))
        : num_kv_iter;

    const int warp_group = tid / WG_SIZE;  /* 0=producer, 1=consumer */
    const int tid_in_wg  = tid % WG_SIZE;
    const int lane_id    = tid % WARP_SIZE;

    if (warp_group == 0) {
        producer_warp_group(
            smem, barriers,
            &tma_Q, &tma_K, &tma_V,
            q_coord_seq, batch_seq_off, coord_head,
            max_kv_iter, tid_in_wg);
    } else {
        consumer_warp_group(
            smem, barriers,
            O_base, seq_stride,
            max_kv_iter, q_block_id, is_causal,
            tid_in_wg, lane_id);
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
     * Producer issues 2 TMA loads per tile:
     *   lo half: coord0 = head_id * DIM + 0
     *   hi half: coord0 = head_id * DIM + 64
     */
    int seq_stride = H_val * D_val;
    cuuint64_t globalDim[2]    = {(cuuint64_t)seq_stride, (cuuint64_t)(B_val * S_val)};
    cuuint64_t globalStride[1] = {(cuuint64_t)(seq_stride * 2)};  /* bytes per row */
    cuuint32_t elemStride[2]   = {1, 1};

    CUtensorMap tma_Q, tma_K, tma_V;

    /* Q: box [64, BLOCK_Q] */
    {
        cuuint32_t boxDim[2] = {(cuuint32_t)HALF_DIM, (cuuint32_t)BLOCK_Q};
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
