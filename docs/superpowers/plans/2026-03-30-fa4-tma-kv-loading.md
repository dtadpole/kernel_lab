# FA4 TMA-Based K/V Loading — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace cp.async K/V loads with hardware TMA (`cp.async.bulk.tensor.2d`) and mbarrier synchronization to close the 5-12% gap vs FA4 CuTe DSL reference.

**Architecture:** Keep 1 DMA + 4 MMA warp structure. DMA warp issues single TMA instructions per tile instead of 32-iteration cp.async loops. mbarrier replaces bar.sync/bar.arrive for DMA↔MMA synchronization. TMA descriptor created on host, base pointer adjusted per (batch,head) on device via `tensormap.replace`.

**Tech Stack:** CUDA 12.x, PTX inline asm for TMA/mbarrier, `cuTensorMapEncodeTiled` (CUDA driver API), SM120 (RTX 5090)

**Spec:** `docs/superpowers/specs/2026-03-30-fa4-tma-kv-loading-design.md`

**File:** All changes in `conf/fixtures/fa4/generated.cu`

---

### Task 1: Add TMA and mbarrier PTX inline helper functions

**Files:**
- Modify: `conf/fixtures/fa4/generated.cu:56-170` (helper section)

- [ ] **Step 1: Add CUDA driver API include and TMA descriptor include**

Add after the existing `#include` block at line 52-57:

```c
#include <cuda.h>          /* CUtensorMap, cuTensorMapEncodeTiled */
```

- [ ] **Step 2: Add TMA 128B swizzle helper**

Add after the existing `swizzle<STRIDE>` template (line 78), before the cp.async section:

```c
/* --- TMA 128B swizzle (matches CU_TENSOR_MAP_SWIZZLE_128B) ---------- */

__device__ inline
uint32_t tma_swizzle_128b(uint32_t addr) {
    /* Swizzle<3,4,3>: XOR bits [6:4] with bits [9:7] */
    return addr ^ (((addr >> 7) & 0x7) << 4);
}
```

- [ ] **Step 3: Add mbarrier PTX inline helpers**

Add after the existing `bar_sync`/`bar_arrive` helpers (lines 157-162):

```c
/* ======================================================================
 *  mbarrier helpers for TMA synchronization
 * ====================================================================== */

/* Initialize an mbarrier with expected arrive count */
__device__ inline void mbar_init(uint32_t smem_addr, uint32_t arrive_count) {
    asm volatile(
        "mbarrier.init.shared.b64 [%0], %1;"
        :: "r"(smem_addr), "r"(arrive_count));
}

/* Invalidate an mbarrier (cleanup) */
__device__ inline void mbar_invalidate(uint32_t smem_addr) {
    asm volatile(
        "mbarrier.inval.shared.b64 [%0];"
        :: "r"(smem_addr));
}

/* Producer: arrive with tx_count bytes expected from TMA */
__device__ inline void mbar_arrive_expect_tx(uint32_t smem_addr, uint32_t tx_bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
        :: "r"(smem_addr), "r"(tx_bytes));
}

/* Consumer: blocking wait on mbarrier phase */
__device__ inline void mbar_wait(uint32_t smem_addr, uint32_t phase) {
    /* Spin on try_wait until the barrier flips to the expected phase */
    uint32_t done = 0;
    while (!done) {
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  mbarrier.try_wait.parity.acquire.cta.shared.b64 p, [%1], %2;\n"
            "  selp.u32 %0, 1, 0, p;\n"
            "}\n"
            : "=r"(done)
            : "r"(smem_addr), "r"(phase));
    }
}

/* TMA: issue cp.async.bulk.tensor.2d with mbarrier */
__device__ inline void tma_load_2d(
    uint32_t smem_dst,      /* SMEM destination (shared address) */
    uint32_t smem_desc,     /* SMEM address of CUtensorMap descriptor */
    int32_t coord_x,        /* tile X coordinate (inner dim) */
    int32_t coord_y,        /* tile Y coordinate (outer dim) */
    uint32_t smem_mbar)     /* SMEM address of mbarrier */
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst), "r"(smem_desc),
           "r"(coord_x), "r"(coord_y),
           "r"(smem_mbar));
}

/* Copy TMA descriptor from global to shared memory (128 bytes) */
__device__ inline void tma_desc_to_smem(uint32_t smem_dst, const CUtensorMap* gmem_src) {
    /* Use 128-byte copy via uint4 (4x 16-byte loads) */
    const uint4* src = reinterpret_cast<const uint4*>(gmem_src);
    uint4* dst = reinterpret_cast<uint4*>(
        __cvta_shared_to_generic(reinterpret_cast<void*>((uintptr_t)smem_dst)));
    for (int i = 0; i < 8; i++) {  /* 128 bytes / 16 bytes per uint4 = 8 */
        dst[i] = src[i];
    }
}

/* Replace TMA descriptor's global address (per-block adjustment) */
__device__ inline void tma_desc_replace_addr(uint32_t smem_desc, const void* new_addr) {
    uint64_t addr64 = reinterpret_cast<uint64_t>(new_addr);
    uint32_t lo = static_cast<uint32_t>(addr64);
    uint32_t hi = static_cast<uint32_t>(addr64 >> 32);
    asm volatile(
        "tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;"
        :: "r"(smem_desc), "l"(addr64));
}

/* Fence after tensormap.replace to ensure visibility */
__device__ inline void tma_desc_fence() {
    asm volatile("fence.proxy.tensormap::generic.release.cta;");
}
```

- [ ] **Step 4: Compile to verify helpers don't introduce syntax errors**

Run:
```bash
make -f cuda_exec/scripts/Makefile compile KERNEL=fa4 RUN_TAG=opt-loop-20260330 TURN=3 2>&1 | tail -10
```
Expected: compile succeeds (helpers are defined but not yet called).

- [ ] **Step 5: Commit helpers**

```bash
git add conf/fixtures/fa4/generated.cu
git commit -m "feat(fa4): add TMA and mbarrier PTX inline helpers"
```

---

### Task 2: Add TMA descriptor creation to kernel_run

**Files:**
- Modify: `conf/fixtures/fa4/generated.cu:652-722` (kernel_run function)

- [ ] **Step 1: Add TMA descriptor creation before kernel launch**

Replace the kernel launch section (inside the `{ ... }` block starting at line 692) with TMA descriptor setup + launch. The full replacement for lines 692-719:

```c
    /* ---- Launch warp-specialized Flash Attention on [B,S,H,D] layout ---- */
    {
        const int BLOCK_Q   = 128;
        const int BLOCK_KV  = 64;
        const int DIM_CONST = 128;
        const int TB_SIZE   = 160;  /* 5 warps: 1 DMA + 4 MMA */

        int effective_bs = B * H;
        int num_blocks = effective_bs * cdiv(S, BLOCK_Q);

        /* SMEM budget:
         *   mbarrier: 64 bytes (4 barriers × 16 bytes, 128-byte aligned → 128 bytes)
         *   Q:  128*128*2 = 32768 (128-byte aligned)
         *   K:  2*64*128*2 = 32768 (128-byte aligned)
         *   V:  2*64*128*2 = 32768 (128-byte aligned)
         *   TMA descriptors: 2*128 = 256 bytes (128-byte aligned)
         *   Total: 128 + 32768 + 32768 + 32768 + 256 = 98688 (~96.4KB, within 99KB)
         */
        const int MBAR_BYTES = 128;  /* 64 bytes padded to 128-byte alignment */
        int smem_q  = BLOCK_Q * DIM_CONST * (int)sizeof(nv_bfloat16);
        int smem_kv = 4 * BLOCK_KV * DIM_CONST * (int)sizeof(nv_bfloat16);
        int smem_desc = 256;  /* 2 × CUtensorMap (128 bytes each) */
        int smem_size = MBAR_BYTES + smem_q + smem_kv + smem_desc;

        /* Create TMA descriptors for K and V.
         *
         * Model K/V as 2D: inner dim = D (128 bf16, contiguous),
         * outer dim = S (seq_len, stride = H*D elements between rows).
         * Each (batch, head) pair is a uniformly-strided 2D slice.
         * Base pointer adjusted per-block via tensormap.replace.
         */
        CUtensorMap K_desc, V_desc;

        uint64_t globalDim[2]    = {(uint64_t)DIM_CONST, (uint64_t)S};
        uint64_t globalStrides[1] = {(uint64_t)(H * DIM_CONST * sizeof(nv_bfloat16))};
        uint32_t boxDim[2]       = {(uint32_t)DIM_CONST, (uint32_t)BLOCK_KV};
        uint32_t elemStrides[2]  = {1, 1};

        CUresult rc_k = cuTensorMapEncodeTiled(
            &K_desc,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,                                    /* tensorRank */
            (void*)(inputs[1]),                   /* globalAddress (K) */
            globalDim,
            globalStrides,
            boxDim,
            elemStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

        CUresult rc_v = cuTensorMapEncodeTiled(
            &V_desc,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            (void*)(inputs[2]),                   /* globalAddress (V) */
            globalDim,
            globalStrides,
            boxDim,
            elemStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

        if (rc_k != CUDA_SUCCESS || rc_v != CUDA_SUCCESS) return -3;

        auto kernel = flash_attention_kernel_ws<BLOCK_Q, BLOCK_KV, DIM_CONST>;
        if (smem_size > 48000)
            cudaFuncSetAttribute(kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        kernel<<<num_blocks, TB_SIZE, smem_size, stream>>>(
            inputs[0], inputs[1], inputs[2], outputs[0],
            &K_desc, &V_desc,
            B, S, H, S, S, causal ? 1 : 0);
    }
```

- [ ] **Step 2: Update kernel signature to accept TMA descriptors**

Change the kernel template function signature (around line 185):

```c
template<int BLOCK_Q, int BLOCK_KV, int DIM>
__launch_bounds__(160, 1)
__global__
void flash_attention_kernel_ws(
    const nv_bfloat16 *Q,   /* [B, S, H, D] */
    const nv_bfloat16 *K,   /* [B, S, H, D] */
    const nv_bfloat16 *V,   /* [B, S, H, D] */
    nv_bfloat16 *O,         /* [B, S, H, D] */
    const CUtensorMap *K_desc_gmem,  /* TMA descriptor for K (in gmem) */
    const CUtensorMap *V_desc_gmem,  /* TMA descriptor for V (in gmem) */
    int B, int S, int H,
    int len_q,
    int len_kv,
    int is_causal)
```

- [ ] **Step 3: Compile to verify descriptor creation and new signature**

Run:
```bash
make -f cuda_exec/scripts/Makefile clean compile KERNEL=fa4 RUN_TAG=opt-loop-20260330 TURN=3 2>&1 | tail -10
```
Expected: compile succeeds. May need to add `-lcuda` linker flag if `cuTensorMapEncodeTiled` is not found. If so, add `-lcuda` to the nvcc binary command in `compile.sh` (or pass via NVCC flags).

- [ ] **Step 4: Commit descriptor creation**

```bash
git add conf/fixtures/fa4/generated.cu
git commit -m "feat(fa4): add TMA descriptor creation in kernel_run"
```

---

### Task 3: Restructure SMEM layout for mbarrier + TMA descriptors

**Files:**
- Modify: `conf/fixtures/fa4/generated.cu` (kernel body, SMEM layout section)

- [ ] **Step 1: Replace SMEM layout computation**

Replace the SMEM layout section (around lines 231-236) with the new layout that includes mbarrier storage and TMA descriptor storage at the front, with 128-byte aligned regions:

```c
    /* Shared memory layout (128-byte aligned for TMA):
     *   mbarrier:  4 × 8 bytes = 32 bytes → padded to 128 bytes
     *   TMA desc:  2 × 128 bytes = 256 bytes (128-byte aligned)
     *   Q region:  [BLOCK_Q, DIM] = 32,768 bytes (128-byte aligned)
     *   K region:  [2, BLOCK_KV, DIM] = 32,768 bytes (128-byte aligned)
     *   V region:  [2, BLOCK_KV, DIM] = 32,768 bytes (128-byte aligned)
     *   Total: 128 + 256 + 32768 + 32768 + 32768 = 98688 bytes (~96.4KB)
     */
    extern __shared__ char smem_raw[];
    const uint32_t smem_raw_base = __cvta_generic_to_shared(smem_raw);

    /* mbarrier region: 4 barriers (K[0], K[1], V[0], V[1]) */
    const uint32_t mbar_base = smem_raw_base;
    const uint32_t K_mbar_0 = mbar_base + 0 * 8;
    const uint32_t K_mbar_1 = mbar_base + 1 * 8;
    const uint32_t V_mbar_0 = mbar_base + 2 * 8;
    const uint32_t V_mbar_1 = mbar_base + 3 * 8;

    /* TMA descriptor region: 2 × 128 bytes */
    const uint32_t desc_base = smem_raw_base + 128;  /* 128-byte aligned */
    const uint32_t K_desc_smem = desc_base;
    const uint32_t V_desc_smem = desc_base + 128;

    /* Data regions (128-byte aligned) */
    const uint32_t data_base = desc_base + 256;
    const uint32_t Q_smem  = data_base;
    const uint32_t KV_base = data_base + BLOCK_Q * DIM * sizeof(nv_bfloat16);
    const uint32_t K_smem  = KV_base;
    const uint32_t V_smem  = KV_base + 2 * BLOCK_KV * DIM * sizeof(nv_bfloat16);
```

Also update `smem` pointer references: the existing code uses `extern __shared__ nv_bfloat16 smem[]` — change to `extern __shared__ char smem_raw[]` and use `smem_raw_base` for all offset calculations. Remove the old `smem_base` variable.

- [ ] **Step 2: Add mbarrier initialization and TMA descriptor copy**

Add right after the SMEM layout section, before Q loading:

```c
    /* ---- Initialize mbarriers (thread 0 only) ---- */
    if (tid == 0) {
        /* K and V tile size in bytes (for TMA tx_count) */
        constexpr uint32_t TILE_BYTES = BLOCK_KV * DIM * sizeof(nv_bfloat16);

        /* Each mbarrier expects: 1 TMA tx completion (TILE_BYTES) */
        mbar_init(K_mbar_0, 1);  /* arrive_count=1 (only DMA thread arrives) */
        mbar_init(K_mbar_1, 1);
        mbar_init(V_mbar_0, 1);
        mbar_init(V_mbar_1, 1);

        /* Copy TMA descriptors from gmem to smem */
        tma_desc_to_smem(K_desc_smem, K_desc_gmem);
        tma_desc_to_smem(V_desc_smem, V_desc_gmem);
    }
    __syncthreads();

    /* ---- Per-block: adjust TMA descriptor base pointers ---- */
    if (tid == 0) {
        const nv_bfloat16 *K_block_base = K + batch_id * S * seq_stride + head_id * DIM;
        const nv_bfloat16 *V_block_base = V + batch_id * S * seq_stride + head_id * DIM;
        tma_desc_replace_addr(K_desc_smem, K_block_base);
        tma_desc_replace_addr(V_desc_smem, V_block_base);
        tma_desc_fence();
    }
    __syncthreads();
```

- [ ] **Step 3: Compile to verify SMEM layout**

Run:
```bash
make -f cuda_exec/scripts/Makefile clean compile KERNEL=fa4 RUN_TAG=opt-loop-20260330 TURN=3 2>&1 | tail -10
```
Expected: compiles successfully. Check ptxas output for SMEM usage.

- [ ] **Step 4: Commit SMEM layout changes**

```bash
git add conf/fixtures/fa4/generated.cu
git commit -m "feat(fa4): restructure SMEM layout for mbarrier + TMA descriptors"
```

---

### Task 4: Rewrite DMA warp to use TMA loads

**Files:**
- Modify: `conf/fixtures/fa4/generated.cu` (DMA warp body inside `if (warp_id == 0)`)

- [ ] **Step 1: Replace the DMA warp body**

Replace the entire DMA warp body (the `if (warp_id == 0) { ... }` block, approximately lines 256-299) with TMA-based loading:

```c
    if (warp_id == 0) {
        /* ================================================================
         *  DMA warp — issues TMA loads for K/V tiles
         *
         *  Only thread 0 issues TMA instructions. Other threads in warp 0
         *  are idle (future optimization: join MMA work).
         *
         *  K and V are double-buffered. TMA auto-arrives at the mbarrier
         *  on completion with tx_bytes = TILE_BYTES.
         * ================================================================ */

        constexpr uint32_t TILE_BYTES = BLOCK_KV * DIM * sizeof(nv_bfloat16);
        constexpr uint32_t K_SLOT_BYTES = BLOCK_KV * DIM * (int)sizeof(nv_bfloat16);
        constexpr uint32_t V_SLOT_BYTES = BLOCK_KV * DIM * (int)sizeof(nv_bfloat16);

        if (tid == 0) {
            uint32_t phase = 0;

            for (int kv_id = 0; kv_id < max_kv_iter; kv_id++) {
                const int slot = kv_id % 2;
                const uint32_t K_mbar = (slot == 0) ? K_mbar_0 : K_mbar_1;
                const uint32_t V_mbar = (slot == 0) ? V_mbar_0 : V_mbar_1;

                /* Wait for consumers to release this K slot (skip first iteration) */
                if (kv_id >= 2) {
                    mbar_wait(K_mbar, phase);
                }

                /* Set expected tx bytes for this barrier and issue TMA load for K */
                mbar_arrive_expect_tx(K_mbar, TILE_BYTES);
                tma_load_2d(
                    K_smem + slot * K_SLOT_BYTES,
                    K_desc_smem,
                    0,                          /* coord_x = 0 (full D dim) */
                    kv_id * BLOCK_KV,           /* coord_y = seq offset */
                    K_mbar);

                /* Set expected tx bytes for V barrier and issue TMA load for V */
                mbar_arrive_expect_tx(V_mbar, TILE_BYTES);
                tma_load_2d(
                    V_smem + slot * V_SLOT_BYTES,
                    V_desc_smem,
                    0,
                    kv_id * BLOCK_KV,
                    V_mbar);

                /* Advance phase every 2 iterations (one full double-buffer cycle) */
                if (slot == 1) phase ^= 1;
            }
        }
        /* Other threads in warp 0 idle — no barrier needed, they skip to end */

    } else {
```

- [ ] **Step 2: Remove the old cp.async-based DMA code**

Ensure the old `global_to_shared_swizzle` calls for K and V, the `cp.async.commit_group`, `cp.async.wait_all`, and `bar_arrive(BAR_K_FULL)` / `bar_arrive(BAR_V_FULL)` in the DMA warp are fully removed. The code in step 1 replaces the entire DMA warp body.

- [ ] **Step 3: Compile to verify DMA warp changes**

Run:
```bash
make -f cuda_exec/scripts/Makefile clean compile KERNEL=fa4 RUN_TAG=opt-loop-20260330 TURN=3 2>&1 | tail -10
```
Expected: compiles. Check for warnings about unused `global_to_shared_swizzle` (still used for Q loading — keep it).

- [ ] **Step 4: Commit DMA warp rewrite**

```bash
git add conf/fixtures/fa4/generated.cu
git commit -m "feat(fa4): rewrite DMA warp to use TMA cp.async.bulk.tensor.2d"
```

---

### Task 5: Update MMA warp synchronization to mbarrier

**Files:**
- Modify: `conf/fixtures/fa4/generated.cu` (MMA warp body inside `else { ... }`)

- [ ] **Step 1: Replace barrier synchronization in MMA warps**

In the MMA warp KV loop body (the `for (int kv_id = 0; ...)` loop inside the `else` block):

**Replace** the K wait + K consumed signal:
```c
// OLD:
bar_sync(BAR_K_FULL, BAR_THREADS);
// ... QK MMA ...
bar_arrive(BAR_K_EMPTY, BAR_THREADS);
```

**With** mbarrier-based synchronization:
```c
// NEW — at start of kv_id iteration:
const int slot = kv_id % 2;
const uint32_t K_mbar_cur = (slot == 0) ? K_mbar_0 : K_mbar_1;
const uint32_t V_mbar_cur = (slot == 0) ? V_mbar_0 : V_mbar_1;
const uint32_t kv_phase = (kv_id / 2) & 1;

/* Wait for K tile ready (TMA auto-arrived at K_mbar) */
mbar_wait(K_mbar_cur, kv_phase);
```

**Replace** the V wait:
```c
// OLD:
bar_sync(BAR_V_FULL, BAR_THREADS);
```

**With:**
```c
// NEW:
mbar_wait(V_mbar_cur, kv_phase);
```

**Remove** the old `bar_arrive(BAR_K_EMPTY, BAR_THREADS)` since mbarrier parity handles buffer reuse automatically. The mbarrier flips phase after all expected arrivals, signaling the producer that the slot is free.

Wait — mbarrier parity-based double-buffering works differently. The producer waits on the barrier's PREVIOUS phase (meaning consumers from the last-last iteration have all arrived). Let me think about this carefully.

Actually, the correct mbarrier pattern for double-buffered TMA pipeline:

1. **Producer (DMA thread):** Before issuing TMA load to slot `s`:
   - Call `mbar_arrive_expect_tx(mbar[s], TILE_BYTES)` to set expected bytes
   - Issue TMA load → TMA hardware auto-arrives with `TILE_BYTES` when done
   - The barrier completes when `arrive_count` arrivals + `tx_bytes` are satisfied

2. **Consumer (MMA warps):** Before reading from slot `s`:
   - Call `mbar_wait(mbar[s], phase)` to wait for data ready
   - Read data
   - No explicit "release" needed — the barrier automatically resets for the next phase

The phase flips each time the barrier completes. So on iteration 0 (slot 0), the barrier starts at phase 0. Producer sets expect_tx + TMA arrives → barrier completes → phase flips to 1. Consumer waits on phase 0 (which is already complete) → proceeds.

On iteration 2 (slot 0 again), the barrier is at phase 1. Producer sets expect_tx again + TMA arrives → barrier completes → phase flips to 0. Consumer waits on phase 1 → proceeds.

So the consumer just needs to track the phase per slot. The phase alternates: slot 0 sees phases 0, 1, 0, 1, ... and slot 1 sees phases 0, 1, 0, 1, ...

For consumers: `phase = (kv_id / 2) % 2` — but this needs careful verification. Actually since each mbarrier (per slot) is used every OTHER iteration:
- K_mbar_0: used at kv_id=0,2,4,... → phase at kv_id=0 is 0, at kv_id=2 is 1, at kv_id=4 is 0, ...
- K_mbar_1: used at kv_id=1,3,5,... → phase at kv_id=1 is 0, at kv_id=3 is 1, ...

So consumer phase = `(kv_id / 2) % 2` for both slots. Let me verify:
- kv_id=0, slot=0: consumer phase = 0/2 % 2 = 0 ✓ (first use of slot 0)
- kv_id=1, slot=1: consumer phase = 1/2 % 2 = 0 ✓ (first use of slot 1)
- kv_id=2, slot=0: consumer phase = 2/2 % 2 = 1 ✓ (second use of slot 0)
- kv_id=3, slot=1: consumer phase = 3/2 % 2 = 1 ✓ (second use of slot 1)
✓ Correct.

The full MMA warp KV loop becomes:

```c
        for (int kv_id = 0; kv_id < max_kv_iter; kv_id++) {
            const int slot = kv_id % 2;
            const uint32_t K_mbar_cur = (slot == 0) ? K_mbar_0 : K_mbar_1;
            const uint32_t V_mbar_cur = (slot == 0) ? V_mbar_0 : V_mbar_1;
            const uint32_t kv_phase = (kv_id / 2) & 1;

            /* Wait for DMA to signal K is ready (via TMA auto-arrive at mbarrier) */
            mbar_wait(K_mbar_cur, kv_phase);

            /* K_cur uses TMA 128B swizzle — update address calculation */
            const uint32_t K_cur = K_smem_thread_base +
                slot * (BLOCK_KV * DIM * (int)sizeof(nv_bfloat16));

            /* V_smem_thread for this iteration's double-buffer slot */
            const uint32_t V_smem_thread = V_smem_thread_base +
                slot * V_SLOT_BYTES;

            /* ... QK MMA (unchanged inner loops) ... */

            /* No explicit K_EMPTY signal — mbarrier phase tracks buffer reuse */

            /* Wait for DMA to signal V is ready */
            mbar_wait(V_mbar_cur, kv_phase);

            /* ... PV MMA (unchanged inner loops) ... */

        } /* end kv_id loop */
```

- [ ] **Step 2: Update K and V ldmatrix swizzle to match TMA 128B pattern**

The K and V fragments are now loaded by TMA with `CU_TENSOR_MAP_SWIZZLE_128B` swizzle. The ldmatrix addresses for K and V must use `tma_swizzle_128b()` instead of the old `swizzle<>()`.

Replace K_smem_thread computation (around line 328-331):
```c
        /* OLD K base: */
        // K_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(
        //     K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));

        /* NEW K base — using TMA 128B swizzle: */
        uint32_t K_smem_thread_base;
        {
            const int row_off = lane_id % 8;
            const int col_off = lane_id / 8 * 8;
            K_smem_thread_base = tma_swizzle_128b(
                K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
        }
```

Replace V_smem_thread_base computation (around line 334-341):
```c
        /* OLD V base: */
        // V_smem_thread_base = swizzle<DIM * sizeof(nv_bfloat16)>(
        //     V_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));

        /* NEW V base — using TMA 128B swizzle: */
        uint32_t V_smem_thread_base;
        {
            const int row_off = lane_id % 16;
            const int col_off = lane_id / 16 * 8;
            V_smem_thread_base = tma_swizzle_128b(
                V_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
        }
```

Also update the K and V fragment load addresses inside the QK and PV inner loops where `kaddr ^= ...` and `addr ^= ...` are used for d-step XOR offsets. These XOR operations must also be compatible with TMA 128B swizzle. Since `tma_swizzle_128b` operates on byte addresses and the d-step XOR (`mma_id_d * MMA_K * sizeof(bf16)`) operates on lower bits that don't conflict with the swizzle bits [9:7] → [6:4], the XOR offsets should still work correctly. Verify by examining bit positions:
- `mma_id_d * MMA_K * sizeof(bf16)` = `mma_id_d * 16 * 2` = `mma_id_d * 32` → affects bits [4:0] to [9:5] depending on mma_id_d
- For mma_id_d=0..7 (DIM/MMA_K=8): offsets are 0, 32, 64, 96, 128, 160, 192, 224 bytes
- These affect bits [7:5], which overlap with swizzle source bits [9:7]

This means the XOR trick for d-step pipelining may NOT be compatible with TMA 128B swizzle. The safest approach: compute the swizzled address from scratch for each (row, col) instead of using XOR shortcuts.

Replace the K fragment load inside the QK d-loop:
```c
                    #pragma unroll
                    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                        uint32_t K_frag[2];
                        {
                            const int k_row = lane_id % 8;
                            const int k_col = lane_id / 8 * 8 + mma_id_d * MMA_K;
                            uint32_t kaddr = K_cur +
                                tma_swizzle_128b(K_smem  /* relative to K tile base */
                                    + slot * (BLOCK_KV * DIM * (int)sizeof(nv_bfloat16))
                                    + (mma_id_kv * MMA_N + k_row) * DIM * sizeof(nv_bfloat16)
                                    + k_col * sizeof(nv_bfloat16))
                                - K_smem  /* subtract base to get swizzled offset */
                                ;
                            /* Simpler: compute full swizzled addr */
                            uint32_t k_tile_base = K_smem + slot * (BLOCK_KV * DIM * (int)sizeof(nv_bfloat16));
                            uint32_t k_byte_off = ((mma_id_kv * MMA_N + k_row) * DIM + k_col) * sizeof(nv_bfloat16);
                            kaddr = tma_swizzle_128b(k_tile_base + k_byte_off);
                            ldmatrix_x2(K_frag, kaddr);
                        }
                        mma_m16n8k16(Q_cur, K_frag, S_local[mma_id_kv]);
                    }
```

Wait, this is getting complex. Let me simplify. Instead of recomputing from scratch every time, note that `tma_swizzle_128b` XORs bits [6:4] with bits [9:7]. The original approach of precomputing a base address and XOR-ing with d-step offsets works IF the XOR doesn't corrupt the swizzle.

Let me just use a helper that computes the fully swizzled address each time:

```c
/* Compute swizzled SMEM address for reading TMA-loaded K/V data */
__device__ inline
uint32_t kv_smem_addr(uint32_t tile_base, int row, int col) {
    uint32_t byte_off = (row * DIM + col) * sizeof(nv_bfloat16);
    return tma_swizzle_128b(tile_base + byte_off);
}
```

Then in the inner loops, call `kv_smem_addr(K_tile_base, row, col)` instead of the XOR-based approach. This is slightly more instructions per ldmatrix but correct.

- [ ] **Step 3: Remove old barrier constants**

Remove or comment out the old barrier constants that are no longer used:
```c
// Remove these lines:
// static constexpr int BAR_K_FULL  = 1;
// static constexpr int BAR_K_EMPTY = 2;
// static constexpr int BAR_V_FULL  = 3;
// static constexpr int BAR_THREADS = 160;
```

- [ ] **Step 4: Compile to verify MMA warp changes**

Run:
```bash
make -f cuda_exec/scripts/Makefile clean compile KERNEL=fa4 RUN_TAG=opt-loop-20260330 TURN=3 2>&1 | tail -15
```
Expected: compiles. Check register count (target: ≤255), spill, barrier count.

- [ ] **Step 5: Commit MMA warp changes**

```bash
git add conf/fixtures/fa4/generated.cu
git commit -m "feat(fa4): update MMA warps to use mbarrier + TMA 128B swizzle"
```

---

### Task 6: Compile final kernel and verify resource usage

**Files:**
- No changes — verification only

- [ ] **Step 1: Clean compile**

```bash
make -f cuda_exec/scripts/Makefile clean compile KERNEL=fa4 RUN_TAG=opt-loop-20260330 TURN=4 2>&1 | tail -15
```

Expected output should show:
- `ptxas info: Used ≤255 registers`
- Spill ≤ 24 bytes (ideally less)
- Binary created successfully

- [ ] **Step 2: Check for linker errors with cuTensorMapEncodeTiled**

If `cuTensorMapEncodeTiled` is not found at link time, add `-lcuda` to the nvcc binary command. Edit `cuda_exec/scripts/compile.sh` line 116-118 to include `-lcuda`:

```bash
# In compile.sh, binary command section:
if [[ -n "$HARNESS" ]]; then
  BINARY_CMD=("$NVCC" "${COMMON_NVCC_ARGS[@]}" "${HARNESS_INCLUDE_ARGS[@]}" "$HARNESS" "$SOURCE" -lcuda -o "$OUTPUT")
else
  BINARY_CMD=("$NVCC" "${COMMON_NVCC_ARGS[@]}" "$SOURCE" -lcuda -o "$OUTPUT")
fi
```

- [ ] **Step 3: Verify binary runs without crash**

Quick smoke test with small config:
```bash
cd /home/centos/.cuda_exec/opt-loop-20260330/v1/99_fa4/turn_4
CUDA_EXEC_PARAM_BATCH_SIZE=2 CUDA_EXEC_PARAM_SEQ_LEN=256 \
CUDA_EXEC_PARAM_NUM_HEADS=16 CUDA_EXEC_PARAM_HEAD_DIM=128 \
CUDA_EXEC_PARAM_CAUSAL=false CUDA_EXEC_PARAM_INPUT_SIZE=1048576 \
CUDA_EXEC_PARAM_NUM_INPUTS=3 CUDA_EXEC_PARAM_NUM_OUTPUTS=1 \
CUDA_EXEC_NUM_WARMUPS=1 CUDA_EXEC_NUM_TRIALS=1 \
./artifacts/compile.attempt_001.generated.bin 2>&1 | \
python3 -c "import json,sys; d=json.load(sys.stdin); print('latency:', d['performance']['latency_ms']['median'])"
```

Expected: non-zero latency (kernel actually ran). If latency is 0ms or binary crashes, debug TMA descriptor setup.

---

### Task 7: Benchmark all 8 configs and compare to baseline

**Files:**
- No changes — benchmark only

- [ ] **Step 1: Run evaluation across all 8 configs**

```bash
rm -f /home/centos/.cuda_exec/.lock_cuda_0
for config in mha-causal-b8-s4096 mha-causal-b4-s8192 mha-causal-b2-s16384 mha-causal-b1-s32768 \
              mha-noncausal-b8-s4096 mha-noncausal-b4-s8192 mha-noncausal-b2-s16384 mha-noncausal-b1-s32768; do
    echo "=== $config ==="
    make -f cuda_exec/scripts/Makefile evaluate KERNEL=fa4 RUN_TAG=opt-loop-20260330 TURN=4 CONFIG=$config 2>&1 | \
        grep -E '(reference:|generated:|speedup:)'
    rm -f /home/centos/.cuda_exec/.lock_cuda_0
    echo ""
done
```

- [ ] **Step 2: Compare against baseline**

Compare Gen/FA4 ratios against baseline (from spec):

| Config | Baseline Gen/FA4 | TMA Gen/FA4 | Delta |
|--------|-----------------|-------------|-------|
| causal b8-s4096 | 0.94x | ? | ? |
| causal b4-s8192 | 0.94x | ? | ? |
| causal b2-s16384 | 0.95x | ? | ? |
| causal b1-s32768 | 0.95x | ? | ? |
| noncausal b8-s4096 | 0.88x | ? | ? |
| noncausal b4-s8192 | 0.88x | ? | ? |
| noncausal b2-s16384 | 0.89x | ? | ? |
| noncausal b1-s32768 | 0.89x | ? | ? |

Target: noncausal ≥ 0.93x, causal ≥ 0.97x.

- [ ] **Step 3: If regression, debug**

If performance is worse than baseline:
1. Check NCU for unexpected spills or low tensor pipe
2. Verify TMA loads are actually completing (check GMEM sector counts)
3. Check if mbarrier wait is stalling too long (compare timing breakdown)
4. Try `CU_TENSOR_MAP_SWIZZLE_NONE` to isolate swizzle issues

---

### Task 8: NCU profile and commit

**Files:**
- No changes — profiling + commit

- [ ] **Step 1: NCU profile the noncausal b2-s16384 config**

```bash
NCU_PREFIX="/home/centos/.cuda_exec/opt-loop-20260330/v1/99_fa4/turn_4/artifacts/ncu.tma-noncausal-b2"
CUDA_EXEC_CONFIG_JSON='{"config_id":"mha-noncausal-b2-s16384","params":{"batch_size":2,"seq_len":16384,"num_heads":16,"head_dim":128,"causal":false}}' \
CUDA_EXEC_PARAM_BATCH_SIZE=2 CUDA_EXEC_PARAM_SEQ_LEN=16384 \
CUDA_EXEC_PARAM_NUM_HEADS=16 CUDA_EXEC_PARAM_HEAD_DIM=128 \
CUDA_EXEC_PARAM_CAUSAL=false CUDA_EXEC_PARAM_INPUT_SIZE=67108864 \
CUDA_EXEC_PARAM_NUM_INPUTS=3 CUDA_EXEC_PARAM_NUM_OUTPUTS=1 \
CUDA_EXEC_NUM_WARMUPS=2 CUDA_EXEC_NUM_TRIALS=1 \
sudo --preserve-env /usr/local/cuda/bin/ncu --set full --target-processes all -f \
    --export "$NCU_PREFIX" \
    /home/centos/.cuda_exec/opt-loop-20260330/v1/99_fa4/turn_4/artifacts/compile.attempt_001.generated.bin 2>&1 | tail -5
```

- [ ] **Step 2: Extract and compare key metrics**

```bash
/usr/local/cuda/bin/ncu --import "$NCU_PREFIX.ncu-rep" --page raw 2>&1 | \
    grep -iE '(gpu__time_duration\.sum|sm__pipe_tensor_subpipe_hmma|local_spilling_requests[^_]|l1tex__data_pipe_lsu_wavefronts_mem_shared\.sum[^.]|l1tex__t_sectors_pipe_lsu_mem_global_op_ld\.sum[^.]|gpu__dram_throughput\.avg)' | \
    grep -v '(!) nan'
```

Compare against baseline NCU:
| Metric | Baseline | TMA | Better? |
|--------|----------|-----|---------|
| gpu__time_duration.sum | 23.86ms | ? | |
| tensor_pipe_hmma % | 43.24% | ? | |
| local_spilling_requests | 25,165,824 | ? | |
| smem_wavefronts | 1,959,956,005 | ? | |
| gmem_load_sectors | 1,077,936,128 | ? | |

- [ ] **Step 3: Commit the final optimized kernel**

```bash
git add conf/fixtures/fa4/generated.cu cuda_exec/scripts/compile.sh cuda_exec/scripts/eval_harness.cu conf/fixtures/fa4/configs.json conf/fixtures/fa4/reference.py
git commit -m "perf(fa4): TMA-based K/V loading — replace cp.async with cp.async.bulk.tensor.2d

Replace software cp.async DMA warp with hardware TMA for K/V tile loads.
mbarrier replaces bar.sync/bar.arrive for producer-consumer synchronization.
TMA descriptor created on host, base pointer adjusted per-block on device.

Changes:
- TMA + mbarrier PTX inline helpers
- cuTensorMapEncodeTiled descriptor creation in kernel_run
- DMA warp: single TMA instruction per tile (was 32-iteration cp.async loop)
- MMA warps: mbarrier wait/arrive (was bar.sync/bar.arrive)
- SMEM layout: mbarrier storage + 128-byte alignment for TMA
- K/V ldmatrix: TMA 128B swizzle (was manual swizzle)

Baseline → TMA Gen/FA4:
  Causal:    0.94-0.95x → [fill after benchmark]
  Noncausal: 0.88-0.89x → [fill after benchmark]"
```

Update the commit message with actual benchmark numbers before committing.
