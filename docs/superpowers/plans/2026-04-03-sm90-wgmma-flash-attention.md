# SM90 WGMMA Flash Attention Forward — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the SM90 Flash Attention forward kernel using WGMMA (warp group MMA) + warp specialization to close the 2.5x gap with CuTe DSL (current: 251–298 TFLOPS, target: 500+ TFLOPS on H100 SXM).

**Architecture:** 3-role warp-specialized kernel — prologue warp (cp.async loads), compute warp group (WGMMA QK + softmax + WGMMA PV), epilogue (output store). Pure CUDA + inline PTX, zero CUTLASS/CuTe dependency. Phased implementation: Phase 1 (WGMMA compute, unified loads), Phase 2 (warp specialization + pipeline), Phase 3 (TMA loads).

**Tech Stack:** CUDA C++17, inline PTX assembly, nvcc with `-arch=sm_90a`

---

## Baseline Performance (2026-04-03)

| Config | CuTe DSL (TFLOPS) | Generated mma.sync (TFLOPS) | Gap |
|--------|-------------------|----------------------------|-----|
| causal-b8-s4096 | 587.9 | 250.7 | 0.43x |
| causal-b4-s8192 | 662.3 | 271.0 | 0.41x |
| causal-b2-s16384 | 705.6 | 280.1 | 0.40x |
| nc-b8-s4096 | 717.0 | 277.9 | 0.39x |
| nc-b4-s8192 | 760.7 | 291.7 | 0.38x |
| nc-b2-s16384 | 773.9 | 298.4 | 0.39x |

H100 SXM peak BF16 dense = 989.5 TFLOPS.

## Root Cause (from 5 prior optimization rounds)

1. **mma.sync is synchronous** — MMA stalls warp; softmax (230 FPU + 38 MUFU instructions) cannot overlap
2. **Address computation overhead** — 21% of SASS is IMAD+LOP3 for swizzled SMEM addressing
3. **ldmatrix overhead** — 136 ldmatrix calls per KV iteration to feed mma.sync operands
4. **No pipeline** — serial load → compute → sync → load pattern

## CuTe DSL Architecture (Reference Target)

For head_dim=128 on SM90:
- **Tiles:** m=128, n=128 (2 MMA warp groups)
- **Pipeline:** 2 stages for K, 2 stages for V
- **Threads:** 384 (2 MMA WGs × 128 + 1 producer WG × 128)
- **WGMMA:** m64n128k16.f32.bf16.bf16 (SS mode QK, RS mode PV)
- **Intra-WG overlap:** QK(n) overlaps with PV(n-1) across warp groups
- **Registers:** 240/thread MMA, 24/thread producer

## Our Target Architecture

Phased approach — each phase is independently testable:

### Phase 1: WGMMA Compute (Tasks 1–5)

Replace mma.sync with WGMMA. Keep unified loads (all threads load and compute).
Same flow as current kernel but with async MMA and no ldmatrix.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Threads | 128 (1 warp group, 4 warps) | Minimum for WGMMA |
| tile_m | 64 | 1 WG = m64 WGMMA |
| tile_n | 64 | Conservative; matches QK output to m64n64k16 |
| DIM | 128 | Fixed head_dim |
| QK WGMMA | m64n64k16 (SS mode) | 32 F32 acc/thread, 8 K-steps |
| PV WGMMA | m64n128k16 (SS mode) | 64 F32 acc/thread, 4 K-steps |
| Q SMEM | 64×128×2B = 16KB | Persistent (loaded once) |
| KV SMEM | 64×128×2B = 16KB | Shared K/V slot (serial) |
| P SMEM | 64×64×2B = 8KB | Softmax output for PV |
| SMEM total | 40KB | Fits 2+ blocks/SM |

**Expected gains:** ~1.5–2x from async WGMMA + no ldmatrix overhead.

### Phase 2: Warp Specialization + Pipeline (Tasks 6–8)

Add producer/consumer warp split and double-buffered K.

| Parameter | Value |
|-----------|-------|
| Threads | 160 (1 producer warp 32 + 1 MMA WG 128) |
| K pipeline | 2-stage double-buffer (32KB) |
| V buffer | Single-buffered 16KB (reuses after K consumed) |
| Sync | Named barriers (bar.sync/bar.arrive) |
| Registers | Producer: 32/thread, Consumer: 240/thread (setmaxnreg) |
| SMEM total | 16 (Q) + 32 (K×2) + 16 (V) + 8 (P) = 72KB |

**Expected gains:** additional 1.2–1.5x from pipeline overlap.

### Phase 3: TMA Loads (Task 9, future)

Replace cp.async with TMA for K/V. Requires per-(batch,head) TMA descriptors
or device-side tensormap encoding. Deferred — cp.async is not the bottleneck
until Phase 2 is working.

---

## Key Design Decisions

### WGMMA Descriptor (64-bit)

Same format as the SM90 matmul plan. For 128B swizzle SMEM layout:

```
Bits [0,14):   start_address    = smem_byte_offset >> 4
Bits [16,30):  leading_byte_off = 1 (for SWIZZLE_128B layouts)
Bits [32,46):  stride_byte_off  = (8_rows × row_bytes) >> 4
Bits [49,52):  base_offset      = 0 (ALWAYS zero — see K-stepping note)
Bits [62,64):  layout_type      = 1 (SWIZZLE_128B)
```

**CRITICAL — K-stepping uses start_address advancement, NOT base_offset:**
The empirically validated SM90 matmul kernel (`data/generated/sm90/matmul/generated.cu`)
confirms: base_offset is ALWAYS 0. K-stepping is done by advancing the
start_address field (bits [0:14)) by `k_step * 2` (each k16 BF16 step =
32 bytes = 2 units of 16 bytes). Use `gmma_desc_advance()` to add to the
lower 32 bits of the descriptor. CUTLASS DescriptorIterator does the same.

### WGMMA Fragment Layout (m64n64k16)

For `wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16`:
each thread in the 128-thread warp group holds 32 F32 values.

**CORRECTED layout** (validated against PTX ISA diagrams and working matmul kernel):

```
Thread t (0-127):
  warp     = t / 32          (0-3)
  lane     = t % 32          (0-31)
  groupID  = lane / 4        (0-7)

Register d[i] (i = 0..31):
  half  = (i >> 1) & 1       (0 or 1 — selects top/bottom 8-row group)
  pair4 = i >> 2             (0-7 — which 8-column chunk)
  sub2  = i & 1              (0 or 1 — even/odd column within pair)

  row = warp * 16 + half * 8 + groupID       (0-63, 2 unique rows per thread)
  col = pair4 * 8 + sub2 + (lane % 4) * 2    (0-63, 16 cols per thread per row)
```

Each thread touches exactly **2 rows** and **16 columns per row** (2 per 8-col chunk × 8 chunks).
The 4 threads with the same `groupID` (lane/4) cover all 8 columns within each chunk:
lane%4=0 covers cols 0,1; lane%4=1 covers cols 2,3; lane%4=2 covers cols 4,5; lane%4=3 covers cols 6,7.

Softmax row reduction: local max/sum across 16 values, then `__shfl_xor_sync`
with masks 1 and 2 to reduce across 4 threads (lane % 4 groups → full 64-col coverage).

### WGMMA Fragment Layout (m64n128k16)

For `wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16`:
each thread holds 64 F32 values.

```
Register d[i] (i = 0..63):
  half  = (i >> 1) & 1       (0 or 1)
  pair4 = i >> 2             (0-15 — which 8-column chunk)
  sub2  = i & 1              (0 or 1)

  row = warp * 16 + half * 8 + groupID       (0-63)
  col = pair4 * 8 + sub2 + (lane % 4) * 2    (0-127, 32 cols per thread per row)
```

### SMEM Swizzle

128B swizzle (same as current kernel). Row width for different tiles:
- Q/K/V [*, 128]: 128 × 2B = 256B per row → 128B swizzle line = 2 swizzle groups/row
- P [*, 64]: 64 × 2B = 128B per row → 1 swizzle group/row

The existing `swizzle<STRIDE>()` and `global_to_shared_swizzle<>()` functions
handle this correctly. Reuse them unchanged.

### QK vs PV Transpose Flags

From CuTe DSL analysis:
- **QK GEMM:** S = Q × K^T. A=Q (K-major, tnspA=0), B=K (K-major, tnspB=0)
  - Q stored as (tile_m, DIM), K stored as (tile_n, DIM). Both have DIM as fast dim = K-major.
- **PV GEMM:** O = P × V. A=P (K-major, tnspA=0), B=V (MN-major, tnspB=1)
  - P stored as (tile_m, tile_n), tile_n is fast dim = K-major.
  - V stored as (tile_n, DIM), DIM is fast dim = MN-major (need transpose).

WGMMA instruction encoding:
- QK: `wgmma.mma_async ... tnspA=0, tnspB=0`
- PV: `wgmma.mma_async ... tnspA=0, tnspB=1`

### Stride Byte Offset for Descriptors

For WGMMA with 128B swizzle, the stride_byte_offset represents the distance
in bytes between "core matrix rows" (groups of 8 rows in the 128B swizzle pattern):

| Buffer | Rows per core | Cols | Bytes/row | stride_byte_offset | >> 4 |
|--------|--------------|------|-----------|-------------------|------|
| Q (64×128) | 8 | 128 | 256 | 8×256=2048 | 128 |
| K (64×128) | 8 | 128 | 256 | 8×256=2048 | 128 |
| P (64×64) | 8 | 64 | 128 | 8×128=1024 | 64 |
| V (64×128) | 8 | 128 | 256 | 8×256=2048 | 128 |

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `.worktrees/FA4/data/generated/sm90/fa4/generated.cu` | **Rewrite** | The WGMMA kernel |
| `.worktrees/FA4/data/generated/sm90/fa4/generated.cu.baseline` | **Keep** | Baseline for revert |
| `data/fixtures/sm90/fa4/configs.json` | Read-only | Test configs |
| `data/fixtures/sm90/fa4/cutedsl.py` | Read-only | CuTe DSL reference |
| `results/sm90/h100_sxm/fa4/` | Create new | Benchmark results |

---

## Task 1: WGMMA + SMEM Helpers

**Files:**
- Modify: `.worktrees/FA4/data/generated/sm90/fa4/generated.cu`

Replace the mma.sync kernel with WGMMA infrastructure. Keep the existing
`swizzle()`, `global_to_shared_swizzle()`, `fast_exp2f()`, `fast_rcp()`
helpers unchanged. Replace `ldmatrix_*` and `mma_m16n8k16` with WGMMA equivalents.

- [ ] **Step 1: Add WGMMA fence/commit/wait helpers**

```cuda
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
```

- [ ] **Step 2: Add WGMMA descriptor construction and advancement**

```cuda
/* Create base WGMMA descriptor. base_offset is ALWAYS 0. */
__device__ __forceinline__
uint64_t make_wgmma_desc(uint32_t smem_addr, int stride_bytes) {
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_addr >> 4) & 0x3FFF);           // [0:14)  start_address
    desc |= (uint64_t)(1) << 16;                              // [16:30) leading_byte_off = 1
    desc |= (uint64_t)(((stride_bytes >> 4) & 0x3FFF)) << 32; // [32:46) stride_byte_off
    // base_offset [49:52) = 0 always
    desc |= (uint64_t)(1) << 62;                               // [62:64) SWIZZLE_128B
    return desc;
}

/* Advance descriptor start_address by offset_16B units (16 bytes each).
 * K-stepping: each k16 BF16 step = 32 bytes = 2 units → advance by ks*2.
 * This is how CUTLASS DescriptorIterator::operator+ works. */
__device__ __forceinline__
uint64_t gmma_desc_advance(uint64_t desc, int offset_16B) {
    uint32_t lo = (uint32_t)desc + (uint32_t)offset_16B;
    uint32_t hi = (uint32_t)(desc >> 32);
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}
```

- [ ] **Step 3: Add WGMMA m64n64k16 inline asm (for QK GEMM)**

32 F32 accumulators. SS mode (both operands from SMEM descriptors).

```cuda
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
    " %32, %33, p, 1, 1, 0, 0;\n"   // scaleA=1, scaleB=1, tnspA=0, tnspB=0
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
```

- [ ] **Step 4: Add WGMMA m64n128k16 inline asm (for PV GEMM)**

64 F32 accumulators. SS mode, **tnspB=1** (V is MN-major in SMEM).

```cuda
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
    " %64, %65, p, 1, 1, 0, 1;\n"   // scaleA=1, scaleB=1, tnspA=0, tnspB=1
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
```

- [ ] **Step 5: Add fence_view_async_shared helper**

Required after writing P to SMEM before WGMMA reads it:

```cuda
__device__ __forceinline__
void fence_view_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
```

- [ ] **Step 6: Remove old ldmatrix and mma_m16n8k16 functions**

Delete: `ldmatrix_x2()`, `ldmatrix_x4()`, `ldmatrix_x2_trans()`, `mma_m16n8k16()`.
These are no longer needed — WGMMA reads directly from SMEM via descriptors.

- [ ] **Step 7: Compile to verify helpers**

```bash
cd /home/zhenc/kernel_lab/.worktrees/FA4 && \
CUDA_VISIBLE_DEVICES=4 /usr/local/cuda/bin/nvcc -arch=sm_90a -std=c++17 -O3 \
    -c data/generated/sm90/fa4/generated.cu -o /dev/null
```

Expected: clean compile with no errors.

---

## Task 2: Kernel Skeleton — WGMMA Unified Architecture

**Files:**
- Modify: `.worktrees/FA4/data/generated/sm90/fa4/generated.cu`

Rewrite the `flash_attention_kernel_unified` function to use WGMMA.
Phase 1: all 128 threads cooperate on loads (same as current), WGMMA for compute.

- [ ] **Step 1: Write kernel signature and constants**

```cuda
template<int BLOCK_Q, int BLOCK_KV, int DIM>
__launch_bounds__(128, 1)   /* 1 block/SM → 256 registers/thread budget */
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
    constexpr int TB_SIZE  = 128;  /* 1 warp group */
    constexpr int MMA_M    = 64;   /* WGMMA m-dim */

    /* SMEM layout:
     *   Q:  BLOCK_Q * DIM * 2B  = 16KB  (persistent)
     *   KV: BLOCK_KV * DIM * 2B = 16KB  (shared K/V slot)
     *   P:  BLOCK_Q * BLOCK_KV * 2B = 8KB  (softmax output for PV)
     *   Total: 40KB */
    extern __shared__ nv_bfloat16 smem[];
    const uint32_t smem_base = __cvta_generic_to_shared(smem);
    const uint32_t Q_smem  = smem_base;
    const uint32_t KV_smem = smem_base + BLOCK_Q * DIM * sizeof(nv_bfloat16);
    const uint32_t P_smem  = KV_smem + BLOCK_KV * DIM * sizeof(nv_bfloat16);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    /* Block → (batch, head, q_block) mapping */
    const int bid = blockIdx.x;
    const int num_q_blocks = cdiv(len_q, BLOCK_Q);
    const int bs_id       = bid / num_q_blocks;
    const int q_block_id  = bid % num_q_blocks;
    const int batch_id    = bs_id / H;
    const int head_id     = bs_id % H;
    const int seq_stride  = H * DIM;
    // ... (same pointer setup as current kernel)
```

- [ ] **Step 2: Load Q to SMEM (all threads, persistent)**

Reuse existing `global_to_shared_swizzle` — same as current kernel.

```cuda
    /* Load Q to shared memory (all 128 threads) */
    global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE>(
        Q_smem, Q_base, seq_stride, tid);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();
```

**Key difference from old kernel:** Q stays in SMEM (not loaded to registers).
WGMMA reads Q directly from SMEM via descriptor.

- [ ] **Step 3: Initialize accumulators**

```cuda
    /* O accumulator: 64 F32 values per thread (m64n128k16 output) */
    float O_acc[64] = {};

    /* Softmax state per row. Each thread owns 2 rows (from WGMMA fragment). */
    float rowmax[2] = {-FLT_MAX, -FLT_MAX};
    float rowsumexp[2] = {0.0f, 0.0f};

    const float softmax_scale_log2 =
        rsqrtf(static_cast<float>(DIM)) * 1.4426950408889634f;
```

- [ ] **Step 4: Compute WGMMA descriptor bases for Q**

The Q descriptor is used for all KV iterations (persistent).

```cuda
    /* Q descriptor: stride = 8 rows × 256 bytes/row = 2048 → >>4 = 128 */
    constexpr int Q_stride_bytes = 8 * DIM * sizeof(nv_bfloat16);  /* 2048 */
```

- [ ] **Step 5: Write KV loop skeleton (load K, QK, softmax, load V, PV)**

```cuda
    const int num_kv_iter = cdiv(len_kv, BLOCK_KV);
    const int max_kv_iter = is_causal
        ? min(num_kv_iter, cdiv((q_block_id + 1) * BLOCK_Q, BLOCK_KV))
        : num_kv_iter;

    const nv_bfloat16 *K_ptr = K_base;
    const nv_bfloat16 *V_ptr = V_base;

    for (int kv_id = 0; kv_id < max_kv_iter; kv_id++) {

        /* ==== Load K tile to shared memory ==== */
        global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(
            KV_smem, K_ptr, seq_stride, tid);
        K_ptr += BLOCK_KV * seq_stride;
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncthreads();

        /* ==== QK GEMM (Task 3) ==== */
        // ... wgmma QK here ...

        /* ==== Online softmax (Task 4) ==== */
        // ... softmax here ...

        /* ==== Sync before reusing KV_smem for V ==== */
        __syncthreads();

        /* ==== Load V tile to shared memory ==== */
        global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(
            KV_smem, V_ptr, seq_stride, tid);
        V_ptr += BLOCK_KV * seq_stride;
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncthreads();

        /* ==== PV GEMM (Task 3) ==== */
        // ... wgmma PV here ...

        __syncthreads();
    }

    /* ==== Epilogue (Task 5) ==== */
    // ... store O ...
```

- [ ] **Step 6: Update kernel_run to launch WGMMA kernel**

```cuda
    const int BLOCK_Q   = 64;
    const int BLOCK_KV  = 64;
    const int DIM_CONST = 128;
    const int TB_SIZE   = 128;

    int num_blocks = B * H * cdiv(S, BLOCK_Q);

    /* SMEM: Q(16KB) + KV(16KB) + P(8KB) = 40KB */
    int smem_size = BLOCK_Q * DIM_CONST * (int)sizeof(nv_bfloat16)
                  + BLOCK_KV * DIM_CONST * (int)sizeof(nv_bfloat16)
                  + BLOCK_Q * BLOCK_KV * (int)sizeof(nv_bfloat16);

    auto kernel = flash_attention_wgmma<BLOCK_Q, BLOCK_KV, DIM_CONST>;
    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    kernel<<<num_blocks, TB_SIZE, smem_size, stream>>>(
        inputs[0], inputs[1], inputs[2], outputs[0],
        B, S, H, S, S, causal ? 1 : 0);
```

- [ ] **Step 7: Compile skeleton**

Same compile command as Task 1 Step 7. Should compile but kernel body is incomplete.

---

## Task 3: QK and PV WGMMA Compute

**Files:**
- Modify: `.worktrees/FA4/data/generated/sm90/fa4/generated.cu`

Implement the WGMMA calls for QK and PV inside the KV loop.

- [ ] **Step 1: Implement QK GEMM**

QK GEMM: S[64,64] = Q[64,128] × K[64,128]^T
WGMMA m64n64k16: 8 K-steps (DIM/16 = 128/16 = 8).

```cuda
        /* ==== QK GEMM: S = Q × K^T ==== */
        float S_acc[32] = {};  /* m64n64k16 output */

        /* Build base descriptors (start_address at column 0, base_offset=0 always) */
        constexpr int Q_stride_bytes = 8 * DIM * sizeof(nv_bfloat16);  /* 2048 */
        uint64_t desc_q_base = make_wgmma_desc(Q_smem, Q_stride_bytes);
        uint64_t desc_k_base = make_wgmma_desc(KV_smem, Q_stride_bytes);

        wgmma_fence();

        #pragma unroll
        for (int ks = 0; ks < DIM / 16; ks++) {
            /* Advance start_address by ks*2 (each k16 step = 32B = 2 × 16B) */
            uint64_t desc_q = gmma_desc_advance(desc_q_base, ks * 2);
            uint64_t desc_k = gmma_desc_advance(desc_k_base, ks * 2);

            wgmma_m64n64k16_f32_bf16(S_acc, desc_q, desc_k,
                                      (ks == 0) ? 0 : 1);
        }

        wgmma_commit_group();
        wgmma_wait_group<0>();

        /* Fence before reading accumulators for softmax */
        wgmma_fence();
```

- [ ] **Step 2: Implement PV GEMM**

PV GEMM: O[64,128] += P[64,64] × V[64,128]
WGMMA m64n128k16: 4 K-steps (BLOCK_KV/16 = 64/16 = 4).

P is the A operand (K-major, tnspA=0). V is the B operand (MN-major, tnspB=1).

```cuda
        /* ==== PV GEMM: O += P × V ==== */
        /* P[64,64] × V[64,128] → O[64,128] via m64n128k16, 4 K-steps */
        constexpr int P_stride_bytes = 8 * BLOCK_KV * sizeof(nv_bfloat16); /* 1024 */
        uint64_t desc_p_base = make_wgmma_desc(P_smem, P_stride_bytes);
        uint64_t desc_v_base = make_wgmma_desc(KV_smem, Q_stride_bytes);

        wgmma_fence();

        #pragma unroll
        for (int ks = 0; ks < BLOCK_KV / 16; ks++) {
            /* Advance by ks*2 (same pattern as QK) */
            uint64_t desc_p = gmma_desc_advance(desc_p_base, ks * 2);
            uint64_t desc_v = gmma_desc_advance(desc_v_base, ks * 2);

            wgmma_m64n128k16_f32_bf16(O_acc, desc_p, desc_v,
                                       1);  /* always accumulate into O */
        }

        wgmma_commit_group();
        wgmma_wait_group<0>();
```

- [ ] **Step 3: Compile to verify**

Same compile command. Should compile cleanly. Will not produce correct
results until softmax and epilogue are implemented.

---

## Task 4: Online Softmax for WGMMA Fragments

**Files:**
- Modify: `.worktrees/FA4/data/generated/sm90/fa4/generated.cu`

Implement online softmax that operates on the WGMMA m64n64k16 output fragment.

**Fragment layout reminder:** Each thread has 32 F32 values covering 2 rows
(half=0,1) × 16 columns. Rows: `warp*16 + half*8 + lane/4`.
Columns: `pair + (lane%4)*16` where pair = i/2.

- [ ] **Step 1: Implement softmax on S_acc**

Insert between QK GEMM and V load:

```cuda
        /* ==== Online softmax ==== */

        /* WGMMA m64n64k16 fragment layout (CORRECTED):
         *   half  = (i >> 1) & 1   → row group (0=top 8, 1=bottom 8)
         *   pair4 = i >> 2         → 8-col chunk (0-7)
         *   sub2  = i & 1          → even/odd col within chunk
         *   row = warp*16 + half*8 + lane/4
         *   col = pair4*8 + sub2 + (lane%4)*2
         *
         * Each thread owns 2 rows × 16 cols = 32 values.
         * For each row (half=0,1): 8 pair4 values × 2 sub2 = 16 col values.
         * Register index for row `half`, pair4 `p4`, sub2 `s2`:
         *   i = (p4 << 2) | (half << 1) | s2
         */

        #pragma unroll
        for (int half = 0; half < 2; half++) {
            /* Gather this row's 16 column values */
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
                local_max = max(local_max, row_vals[c]);

            /* Warp reduce max across 4 threads (lane%4 groups → full 64-col coverage) */
            local_max = max(local_max,
                __shfl_xor_sync(0xFFFFFFFF, local_max, 1));
            local_max = max(local_max,
                __shfl_xor_sync(0xFFFFFFFF, local_max, 2));

            /* Update running max */
            float new_max = max(local_max, rowmax[half]);
            float rescale = fast_exp2f(rowmax[half] - new_max);
            rowmax[half] = new_max;

            /* Rescale O accumulator for this row.
             * O_acc uses m64n128k16 layout: for the same `half` value,
             * all registers with (i>>1)&1 == half belong to this row.
             * That's i = ...|half<<1|... → indices where bit 1 == half. */
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

            /* Convert to BF16 and store to P_smem.
             * P layout: (BLOCK_Q, BLOCK_KV) = (64, 64) with 128B swizzle.
             * Thread's row: warp*16 + half*8 + lane/4
             * Thread's cols: pair4*8 + sub2 + (lane%4)*2 */
            const int p_row = warp_id * 16 + half * 8 + (lane_id / 4);
            #pragma unroll
            for (int p4 = 0; p4 < 8; p4++) {
                for (int s2 = 0; s2 < 2; s2++) {
                    const int p_col = p4 * 8 + s2 + (lane_id % 4) * 2;
                    const uint32_t p_addr = swizzle<BLOCK_KV * (int)sizeof(nv_bfloat16)>(
                        P_smem + (p_row * BLOCK_KV + p_col) * sizeof(nv_bfloat16));
                    nv_bfloat16 val = __float2bfloat16(row_vals[p4 * 2 + s2]);
                    asm volatile("st.shared.b16 [%0], %1;"
                                 :: "r"(p_addr), "h"(*(uint16_t*)&val));
                }
            }
        }

        /* Ensure P writes visible to WGMMA */
        fence_view_async_shared();
        __syncthreads();
```

- [ ] **Step 2: Verify softmax row/col index mapping**

The mapping from register index `i` (0..31) to (row, col) for m64n64k16:
```
half = i % 2       → selects row within the 2 rows this thread owns
pair = i / 2       → selects column pair
row  = warp*16 + half*8 + lane/4
col  = pair + (lane%4)*16
```

Verify: for `S_acc[half + p*2]`, half selects the row (0 or 1), p selects
the pair (0..15). This matches `i = half + p*2`, so `i/2 = p`, `i%2 = half`. ✓

- [ ] **Step 3: Compile to verify**

---

## Task 5: Epilogue — Output Store

**Files:**
- Modify: `.worktrees/FA4/data/generated/sm90/fa4/generated.cu`

Write the output O from WGMMA accumulators to global memory.

- [ ] **Step 1: Implement epilogue (after KV loop)**

O_acc is in m64n128k16 layout. Each thread holds 64 F32 values covering
2 rows × 32 columns of the 64×128 output tile.

```cuda
    /* ---- Epilogue: finalize O and store ---- */

    /* Divide by sum of exponentials */
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        float inv_sum = fast_rcp(rowsumexp[half]);
        #pragma unroll
        for (int p4 = 0; p4 < 16; p4++) {
            O_acc[(p4 << 2) | (half << 1) | 0] *= inv_sum;
            O_acc[(p4 << 2) | (half << 1) | 1] *= inv_sum;
        }
    }

    /* Write O to global memory.
     * m64n128k16 CORRECTED layout:
     *   half  = (i >> 1) & 1
     *   pair4 = i >> 2         (0-15, 16 chunks of 8 cols)
     *   sub2  = i & 1
     *   row = warp*16 + half*8 + lane/4
     *   col = pair4*8 + sub2 + (lane%4)*2 */
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        const int row = warp_id * 16 + half * 8 + (lane_id / 4);
        if (row < len_q - q_block_id * BLOCK_Q) {
            #pragma unroll
            for (int p4 = 0; p4 < 16; p4++) {
                for (int s2 = 0; s2 < 2; s2++) {
                    const int col = p4 * 8 + s2 + (lane_id % 4) * 2;
                    const int idx = (p4 << 2) | (half << 1) | s2;
                    nv_bfloat16 val = __float2bfloat16(O_acc[idx]);
                    O_base[row * seq_stride + col] = val;
                }
            }
        }
    }
```

- [ ] **Step 2: Compile full kernel**

```bash
cd /home/zhenc/kernel_lab/.worktrees/FA4 && \
CUDA_VISIBLE_DEVICES=4 PTXAS_ARCH=sm_90 \
make -f /home/zhenc/kernel_lab/cuda_exec/scripts/Makefile compile \
    KERNEL=fa4 ARCH=sm_90 RUN_TAG=optim-fa4-wgmma TURN=1 \
    REFERENCE_PY=data/fixtures/sm90/fa4/cutedsl.py \
    GENERATED_CU=data/generated/sm90/fa4/generated.cu
```

Check compile output:
- Register count (expect ~200-240/thread)
- Spill bytes (must be 0)
- SMEM allocation

- [ ] **Step 3: Evaluate correctness on one config**

Run the compiled binary on `mha-noncausal-b4-s8192` and compare output
against the CuTe DSL reference. Use the eval harness.

```bash
CUDA_VISIBLE_DEVICES=4 \
CUDA_EXEC_CONFIG_JSON='{"batch_size":4,"seq_len":8192,...}' \
... (same env as Phase 1 baseline evaluation) \
/path/to/compiled/binary 2>&1 | python3 -c "..."
```

If correctness fails, debug by:
1. Testing with tiny sizes (B=1, S=64, H=1) where output is manually verifiable
2. Checking WGMMA descriptor base_offset computation
3. Checking P_smem swizzle addresses match WGMMA descriptor layout
4. Checking tnspB=1 for PV GEMM vs tnspB=0 for QK GEMM

- [ ] **Step 4: Evaluate all 6 configs and output performance table**

Run all configs, compute TFLOPS, compare against baseline and CuTe DSL.

- [ ] **Step 5: Commit Phase 1 result**

```bash
cd /home/zhenc/kernel_lab/.worktrees/FA4 && \
git add data/generated/sm90/fa4/generated.cu && \
git commit -m "perf: FA4 — replace mma.sync with WGMMA m64n64k16/m64n128k16

Phase 1 of WGMMA rewrite: async WGMMA for both QK and PV matmuls.
Eliminates ldmatrix overhead and enables softmax/MMA overlap.
<N> TFLOPS (up from 251-298 TFLOPS baseline)."
```

---

## Task 6: Warp Specialization — Producer/Consumer Split

**Files:**
- Modify: `.worktrees/FA4/data/generated/sm90/fa4/generated.cu`

Split into producer warp (cp.async K/V loads) + consumer warp group (WGMMA).

- [ ] **Step 1: Change thread count to 160 and add warp role detection**

```cuda
template<int BLOCK_Q, int BLOCK_KV, int DIM>
__launch_bounds__(160, 1)
__global__
void flash_attention_wgmma(...)
{
    constexpr int NUM_PRODUCER_THREADS = 32;  /* 1 warp */
    constexpr int NUM_CONSUMER_THREADS = 128; /* 1 warp group */
    constexpr int TB_SIZE = NUM_PRODUCER_THREADS + NUM_CONSUMER_THREADS;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const bool is_producer = (warp_id == 0);

    /* Register budget: producer gets fewer, consumer gets more */
    if (is_producer) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;\n");
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 240;\n");
    }

    /* Consumer thread index (0-127 within warp group) */
    const int consumer_tid = tid - NUM_PRODUCER_THREADS;
    const int consumer_warp = consumer_tid / 32;
    const int consumer_lane = consumer_tid % 32;
```

- [ ] **Step 2: Add named barrier helpers**

```cuda
/* Named barrier IDs */
enum BarrierId {
    BAR_K_LOADED = 1,   /* Producer → Consumer: K is in SMEM */
    BAR_K_CONSUMED = 2, /* Consumer → Producer: done with K, can load V */
    BAR_V_LOADED = 3,   /* Producer → Consumer: V is in SMEM */
    BAR_V_CONSUMED = 4, /* Consumer → Producer: done with V, can load K */
};

__device__ __forceinline__
void barrier_arrive(int bar_id, int thread_count) {
    asm volatile("bar.arrive %0, %1;" :: "r"(bar_id), "r"(thread_count));
}

__device__ __forceinline__
void barrier_sync(int bar_id, int thread_count) {
    asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(thread_count));
}
```

- [ ] **Step 3: Implement producer loop**

**IMPORTANT:** All arrive/sync calls for the same barrier ID must use the
same thread count (TB_SIZE=160). `bar.arrive` contributes threads without
blocking; `bar.sync` blocks until the total arrivals reach threadCount.

```cuda
    if (is_producer) {
        /* Producer: load Q first (32 threads — slower but only done once) */
        global_to_shared_swizzle<BLOCK_Q, DIM, NUM_PRODUCER_THREADS>(
            Q_smem, Q_base, seq_stride, tid);
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");

        /* Signal Q loaded — use barrier to sync all 160 threads */
        barrier_sync(BAR_K_LOADED, TB_SIZE);

        for (int kv_id = 0; kv_id < max_kv_iter; kv_id++) {
            /* Load K */
            global_to_shared_swizzle<BLOCK_KV, DIM, NUM_PRODUCER_THREADS>(
                KV_smem, K_ptr, seq_stride, tid);
            K_ptr += BLOCK_KV * seq_stride;
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_all;");

            /* Signal K loaded (non-blocking, 160 = producer 32 + consumer 128) */
            barrier_arrive(BAR_K_LOADED, TB_SIZE);

            /* Wait for consumer to finish with K before loading V */
            barrier_sync(BAR_K_CONSUMED, TB_SIZE);

            /* Load V (reuses KV_smem after consumer is done with K) */
            global_to_shared_swizzle<BLOCK_KV, DIM, NUM_PRODUCER_THREADS>(
                KV_smem, V_ptr, seq_stride, tid);
            V_ptr += BLOCK_KV * seq_stride;
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_all;");

            /* Signal V loaded */
            barrier_arrive(BAR_V_LOADED, TB_SIZE);

            /* Wait for consumer to finish with V */
            barrier_sync(BAR_V_CONSUMED, TB_SIZE);
        }
        return;  /* producer is done */
    }
```

- [ ] **Step 4: Implement consumer loop**

```cuda
    /* Consumer: WGMMA compute */
    {
        /* Wait for Q load (sync with producer's Q barrier) */
        barrier_sync(BAR_K_LOADED, TB_SIZE);

        for (int kv_id = 0; kv_id < max_kv_iter; kv_id++) {
            /* Wait for K (consumer side: sync contributes 128 threads) */
            barrier_sync(BAR_K_LOADED, TB_SIZE);

            /* QK GEMM (same as Task 3 Step 1) */
            // ... wgmma QK ...

            /* Signal K consumed (non-blocking, all 160 threads counted) */
            barrier_arrive(BAR_K_CONSUMED, TB_SIZE);

            /* Softmax (same as Task 4 Step 1) */
            // ... softmax ...

            /* Wait for V */
            barrier_sync(BAR_V_LOADED, TB_SIZE);

            /* PV GEMM (same as Task 3 Step 2) */
            // ... wgmma PV ...

            /* Signal V consumed */
            barrier_arrive(BAR_V_CONSUMED, TB_SIZE);
        }

        /* Epilogue (same as Task 5 Step 1) */
        // ... store O ...
    }
```

- [ ] **Step 5: Compile and test correctness**

- [ ] **Step 6: Benchmark all configs**

---

## Task 7: Double-Buffered K Pipeline

**Files:**
- Modify: `.worktrees/FA4/data/generated/sm90/fa4/generated.cu`

Add 2-stage K pipeline so producer can prefetch K[i+1] while consumer
computes QK on K[i].

- [ ] **Step 1: Expand SMEM for 2 K stages**

```cuda
    /* SMEM layout:
     *   Q:    64×128×2 = 16KB
     *   K[0]: 64×128×2 = 16KB  (stage 0)
     *   K[1]: 64×128×2 = 16KB  (stage 1)
     *   V:    64×128×2 = 16KB
     *   P:    64×64×2  = 8KB
     *   Total: 72KB */
    const uint32_t Q_smem    = smem_base;
    const uint32_t K_smem_0  = Q_smem + BLOCK_Q * DIM * sizeof(nv_bfloat16);
    const uint32_t K_smem_1  = K_smem_0 + BLOCK_KV * DIM * sizeof(nv_bfloat16);
    const uint32_t V_smem    = K_smem_1 + BLOCK_KV * DIM * sizeof(nv_bfloat16);
    const uint32_t P_smem    = V_smem + BLOCK_KV * DIM * sizeof(nv_bfloat16);
```

- [ ] **Step 2: Implement producer prelude (pre-fill pipeline)**

```cuda
    if (is_producer) {
        /* Prelude: load K[0] to stage 0 */
        global_to_shared_swizzle<BLOCK_KV, DIM, NUM_PRODUCER_THREADS>(
            K_smem_0, K_ptr, seq_stride, tid);
        K_ptr += BLOCK_KV * seq_stride;
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        barrier_arrive(BAR_K_LOADED, NUM_PRODUCER_THREADS);

        /* Main loop: load K[i+1] while consumer processes K[i] */
        for (int kv_id = 0; kv_id < max_kv_iter; kv_id++) {
            int cur_stage = kv_id % 2;
            int next_stage = (kv_id + 1) % 2;
            uint32_t next_K_smem = (next_stage == 0) ? K_smem_0 : K_smem_1;

            /* Pre-load K[kv_id+1] to next stage (overlaps with consumer QK) */
            if (kv_id + 1 < max_kv_iter) {
                global_to_shared_swizzle<BLOCK_KV, DIM, NUM_PRODUCER_THREADS>(
                    next_K_smem, K_ptr, seq_stride, tid);
                K_ptr += BLOCK_KV * seq_stride;
                asm volatile("cp.async.commit_group;");
            }

            /* Wait for consumer to finish with K[kv_id] */
            barrier_sync(BAR_K_CONSUMED, TB_SIZE);

            /* Load V[kv_id] */
            global_to_shared_swizzle<BLOCK_KV, DIM, NUM_PRODUCER_THREADS>(
                V_smem, V_ptr, seq_stride, tid);
            V_ptr += BLOCK_KV * seq_stride;
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_all;");  /* wait for both K[i+1] and V[i] */
            barrier_arrive(BAR_V_LOADED, NUM_PRODUCER_THREADS);

            /* Wait for consumer to finish with V */
            barrier_sync(BAR_V_CONSUMED, TB_SIZE);

            /* Signal next K is ready */
            if (kv_id + 1 < max_kv_iter) {
                barrier_arrive(BAR_K_LOADED, NUM_PRODUCER_THREADS);
            }
        }
        return;
    }
```

- [ ] **Step 3: Update consumer to use alternating K stages**

```cuda
    /* Consumer reads from K_smem[kv_id % 2] */
    for (int kv_id = 0; kv_id < max_kv_iter; kv_id++) {
        uint32_t cur_K_smem = (kv_id % 2 == 0) ? K_smem_0 : K_smem_1;

        barrier_sync(BAR_K_LOADED, TB_SIZE);

        /* QK GEMM using cur_K_smem */
        // ... (update KV_smem → cur_K_smem in descriptor construction) ...

        barrier_arrive(BAR_K_CONSUMED, NUM_CONSUMER_THREADS);

        /* Softmax ... */

        barrier_sync(BAR_V_LOADED, TB_SIZE);

        /* PV GEMM using V_smem */
        // ...

        barrier_arrive(BAR_V_CONSUMED, NUM_CONSUMER_THREADS);
    }
```

- [ ] **Step 4: Update SMEM size in kernel_run**

```cuda
    int smem_size = BLOCK_Q * DIM * 2          /* Q: 16KB */
                  + 2 * BLOCK_KV * DIM * 2     /* K[0]+K[1]: 32KB */
                  + BLOCK_KV * DIM * 2          /* V: 16KB */
                  + BLOCK_Q * BLOCK_KV * 2;     /* P: 8KB */
    /* Total: 72KB */
```

- [ ] **Step 5: Compile, test correctness, benchmark**

- [ ] **Step 6: Commit Phase 2 result**

```bash
git add data/generated/sm90/fa4/generated.cu && \
git commit -m "perf: FA4 — warp specialization + 2-stage K pipeline

Phase 2: producer warp (cp.async) + consumer warp group (WGMMA).
Double-buffered K overlaps load with compute.
<N> TFLOPS (up from Phase 1 <M> TFLOPS)."
```

---

## Task 8: Benchmark, Profile, and Write Results

**Files:**
- Create: `results/sm90/h100_sxm/fa4/YYYYMMDD_HHMM_<hash>_wgmma-phase2.md`

- [ ] **Step 1: Benchmark all 6 configs**

Run generated kernel, CuTe DSL reference, and (if available) cuDNN on all configs.
Output the mandatory performance comparison table.

- [ ] **Step 2: Profile with NCU (if driver compatible)**

If NCU works, profile the 2 most interesting configs:
- `mha-noncausal-b4-s8192` (largest gap in baseline)
- `mha-causal-b8-s4096` (shortest sequence)

Collect: SM%, DRAM%, occupancy, warp stalls, register usage.

- [ ] **Step 3: Write results file**

Include: hardware specs, before/after performance table, NCU comparison,
what worked and what the remaining gap is.

- [ ] **Step 4: Commit results**

---

## Task 9: Phase 3 — TMA Loads (Future)

**Deferred until Phase 2 performance is validated.**

This task replaces cp.async with TMA for K/V loads. Requires:
- `cuTensorMapEncodeTiled` via `dlopen("libcuda.so.1")` (reuse matmul plan pattern)
- Per-(batch, head) TMA descriptors (128 max for B=8, H=16)
- TMA 2D load PTX: `cp.async.bulk.tensor.2d.shared::cta.global.tile`
- mbarrier integration (TMA signals mbarrier on completion)

Key challenge: (B, S, H, D) tensor layout has stride H*D between Q/K/V rows
for the same (batch, head). Each TMA descriptor describes one (batch, head)
slice as a 2D tensor: globalDim={D, S}, stride={H*D*2 bytes}.

---

## Debugging Guide

### Common WGMMA Issues

1. **Wrong output values (all zeros or NaN)**
   - Check `scale_D` flag: 0 = zero-init accumulators, 1 = accumulate
   - First K-step of QK GEMM must use scale_D=0
   - PV GEMM always uses scale_D=1 (accumulate into O)

2. **Wrong output pattern (shifted or scrambled)**
   - WGMMA descriptor `base_offset` calculation
   - Swizzle line wrapping when `ks >= 4` (128B line holds 64 BF16 = 4 K-steps)
   - `start_address` must shift by 128B for each swizzle line

3. **Hang or deadlock**
   - Missing `wgmma.commit_group` after WGMMA calls
   - Missing `wgmma.wait_group` before reading accumulators
   - Named barrier thread count mismatch (must match exactly)

4. **Register spills (performance regression)**
   - Check `ptxas` output for spill stores/loads
   - Reduce accumulator size (try tile_n=32 for QK)
   - Use `setmaxnreg` to give consumer more registers

5. **Correctness passes on some configs but not others**
   - Causal mask row/col computation for WGMMA fragment layout
   - Boundary handling when seq_len is not divisible by BLOCK_Q or BLOCK_KV

### Testing Strategy

1. **Tiny test:** B=1, S=64, H=1, D=128 (1 Q block × 1 KV block, no loop)
2. **Non-causal first:** Easier to debug (no mask logic)
3. **Single batch/head:** B=1, H=1 to isolate indexing bugs
4. **Compare against CuTe DSL:** allclose with atol=0.05, rtol=0.05 (BF16 precision)

---

## Reference: Key PTX Instructions

### WGMMA

```ptx
wgmma.fence.sync.aligned;
wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16
    {d0..d31}, desc_a, desc_b, p, scaleA, scaleB, tnspA, tnspB;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16
    {d0..d63}, desc_a, desc_b, p, scaleA, scaleB, tnspA, tnspB;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned N;
```

### Register Budget Control

```ptx
setmaxnreg.inc.sync.aligned.u32 240;   // increase to 240 (consumer)
setmaxnreg.dec.sync.aligned.u32 32;    // decrease to 32 (producer)
```

### Named Barriers

```ptx
bar.sync barID, threadCount;    // block until all threadCount threads arrive
bar.arrive barID, threadCount;  // signal arrival without blocking
```

### Shared Memory Fence

```ptx
fence.proxy.async.shared::cta;  // make SMEM stores visible to WGMMA/TMA
```
