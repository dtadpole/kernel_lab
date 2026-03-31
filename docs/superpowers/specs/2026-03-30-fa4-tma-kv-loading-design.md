# FA4 TMA-Based K/V Loading Optimization

## Problem

The FA4 kernel achieves 88-89% of FA4 CuTe DSL reference on non-causal configs and 94-95% on causal. The remaining gap is primarily from the DMA warp using software `cp.async` with manual address arithmetic for strided [B,S,H,D] global memory loads, while FA4 CuTe DSL uses hardware TMA (`cp.async.bulk.tensor`).

### Baseline Performance (commit 822840d)

| Config | Reference (ms) | Generated (ms) | Gen/FA4 |
|--------|---------------|----------------|---------|
| causal b8-s4096 | 2.65 | 2.83 | 0.94x |
| causal b4-s8192 | 5.09 | 5.42 | 0.94x |
| causal b2-s16384 | 10.02 | 10.59 | 0.95x |
| causal b1-s32768 | 19.94 | 21.02 | 0.95x |
| noncausal b8-s4096 | 4.55 | 5.19 | 0.88x |
| noncausal b4-s8192 | 9.12 | 10.32 | 0.88x |
| noncausal b2-s16384 | 18.30 | 20.52 | 0.89x |
| noncausal b1-s32768 | 36.64 | 41.03 | 0.89x |

### NCU Profile (noncausal b2-s16384)

- 1.08B GMEM load sectors, 0% L1 hit (all go to L2)
- 25M local memory spill requests (24 bytes/thread)
- Tensor pipe (HMMA): 43% of peak (PM sampling)
- SMEM loads: 19.84% of time (1.88B wavefronts)
- DRAM throughput: 1.27% — not bandwidth-bound

## Root Cause

The DMA warp (32 threads) runs a loop of `cp.async.cg.shared.global` instructions with manual address computation for each 16-byte chunk:
1. Each tile (64×128 bf16 = 16KB) requires `64*128 / (32*8) = 32` iterations per DMA thread
2. Per iteration: index math + swizzle + cp.async = 3-4 ALU instructions + 1 memory instruction
3. The strided layout means consecutive rows are 4096 bytes apart — every cp.async misses L1

TMA replaces this entire loop with a single `cp.async.bulk.tensor.2d` instruction per tile. The hardware handles address generation, optimal L2 sector requests, SMEM swizzle, and automatic mbarrier arrival on completion.

## Design

### Architecture

Keep the existing 1 DMA + 4 MMA warp structure. Replace cp.async data movement with TMA for K and V loads.

### TMA Descriptor Strategy

The [B,S,H,D] layout means head `h` in batch `b` has K/V data at positions with stride `H*D` between consecutive sequence positions. This is a valid strided 2D tensor for TMA.

**One template descriptor per tensor (K, V)**, created on the host with:
- `globalDim = {D, S}` — 128 columns, S rows per (batch, head)
- `globalStrides = {H*D*sizeof(bf16), sizeof(bf16)}` — row stride = 4096 bytes
- `boxDim = {D, BLOCK_KV}` — 128×64 tile
- `swizzle = CU_TENSOR_MAP_SWIZZLE_128B` — 128-byte swizzle for bank conflict avoidance

On the device, each block adjusts the base pointer via `tensormap.replace.tile.global_address` to point at its (batch, head) slice:
```
K_base_for_block = K + batch_id * S * H * D + head_id * D
```

The DMA thread then loads tiles with coordinates:
- `coord_x = 0` (always full D=128 columns)
- `coord_y = kv_id * BLOCK_KV`

### Host Side (kernel_run)

```c
// Create template TMA descriptors for K and V
CUtensorMap K_desc, V_desc;

uint64_t globalDim[2]    = {(uint64_t)DIM, (uint64_t)S};
uint64_t globalStrides[1] = {(uint64_t)(H * DIM * sizeof(__nv_bfloat16))}; // only outer stride needed
uint32_t boxDim[2]       = {(uint32_t)DIM, (uint32_t)BLOCK_KV};
uint32_t elemStrides[2]  = {1, 1};

cuTensorMapEncodeTiled(&K_desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    2, (void*)K_ptr, globalDim, globalStrides, boxDim, elemStrides,
    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

// Same for V_desc with V_ptr

// Pass descriptors to kernel (copied to SMEM or via grid constant)
kernel<<<grid, block, smem, stream>>>(Q, O, &K_desc, &V_desc, B, S, H, ...);
```

### Barrier Changes

Replace `bar.sync/bar.arrive` with mbarrier:

**SMEM allocation (add to existing layout):**
```
mbarrier K_bar[2]   — 2 barriers for K double-buffer (16 bytes each)
mbarrier V_bar[2]   — 2 barriers for V double-buffer (16 bytes each)
Total: 64 bytes (negligible)
```

**Initialization (thread 0, before KV loop):**
```ptx
mbarrier.init.shared.b64 [K_bar + slot*8], expected_count;
mbarrier.init.shared.b64 [V_bar + slot*8], expected_count;
```

Where `expected_count`:
- For K_bar: DMA arrival (1) + TMA tx_count arrival (auto). TMA automatically arrives with tx_count = tile_bytes = 16KB.
- For V_bar: same pattern.

**Producer arrive pattern** (replaces `bar_arrive`):
TMA auto-arrives at the specified mbarrier on completion. No explicit producer arrive needed.

**Consumer wait pattern** (replaces `bar_sync`):
```ptx
mbarrier.try_wait.parity.acquire.cta.shared.b64 %pred, [K_bar + slot*8], %parity;
```

**Consumer release** (replaces `bar_arrive` for K_EMPTY):
```ptx
mbarrier.arrive.release.cta.shared.b64 [K_bar + slot*8], 1;
```

### DMA Warp (warp 0)

```
// Copy TMA descriptors to SMEM (thread 0)
__shared__ CUtensorMap smem_K_desc, smem_V_desc;
copy K_desc → smem_K_desc, V_desc → smem_V_desc

// Prefetch descriptors
tensormap.cp_fenceproxy.global.shared::cta [smem_K_desc]
tensormap.cp_fenceproxy.global.shared::cta [smem_V_desc]

// Replace base pointer for this block's (batch, head)
tensormap.replace.tile.global_address [smem_K_desc], K_base_for_block
tensormap.replace.tile.global_address [smem_V_desc], V_base_for_block

for kv_id = 0 to max_kv_iter:
    // Wait for previous K consumed (skip first iteration)
    if kv_id > 0:
        mbarrier.try_wait.parity K_bar[kv_id % 2], phase

    // Issue TMA load for K
    cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
        [K_smem + (kv_id%2)*tile_bytes], [smem_K_desc], {0, kv_id*BLOCK_KV}, [K_bar[kv_id%2]]

    // Issue TMA load for V (no wait — V is double-buffered)
    cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
        [V_smem + (kv_id%2)*tile_bytes], [smem_V_desc], {0, kv_id*BLOCK_KV}, [V_bar[kv_id%2]]
```

### MMA Warps (warps 1-4)

```
for kv_id = 0 to max_kv_iter:
    // Wait for K ready
    mbarrier.try_wait.parity K_bar[kv_id % 2], phase

    // QK MMA + softmax (unchanged — same ldmatrix + mma.sync + exp2f)
    ...

    // Signal K consumed
    mbarrier.arrive K_bar[kv_id % 2]

    // Wait for V ready
    mbarrier.try_wait.parity V_bar[kv_id % 2], phase

    // PV MMA (unchanged)
    ...
```

The ldmatrix + mma.sync inner loops remain identical. Only the synchronization mechanism changes.

### SMEM Layout

```
Offset 0:      mbarrier K_bar[0..1]  (16 bytes)
Offset 16:     mbarrier V_bar[0..1]  (16 bytes)
Offset 128:    Q  [128 × 128 × 2]   (32,768 bytes, 128-byte aligned)
Offset 32,896: K  [2 × 64 × 128 × 2] (32,768 bytes, 128-byte aligned)
Offset 65,664: V  [2 × 64 × 128 × 2] (32,768 bytes, 128-byte aligned)
Total: 98,432 bytes (~96KB, fits SM120's 99KB = 101,376 bytes)
```

TMA requires 128-byte aligned SMEM addresses for tile destinations. The K and V offsets must be 128-byte aligned.

### SMEM Swizzle Compatibility

Current kernel uses manual `swizzle<STRIDE>()` for cp.async and ldmatrix. TMA applies hardware swizzle (128B mode) to SMEM writes. The ldmatrix reads in MMA warps must use the SAME swizzle pattern.

With `CU_TENSOR_MAP_SWIZZLE_128B`, TMA XORs address bits to avoid bank conflicts. The existing `swizzle<WIDTH*sizeof(bf16)>()` helper computes `row_idx / max(64/STRIDE, 1)` and XORs into bits [4:6]. For K/V tiles with WIDTH=128 (bf16 elements = 256 bytes), the manual swizzle should match TMA's 128B swizzle. If not, adjust ldmatrix address computation to match TMA's swizzle pattern.

**Verification step:** After TMA loads K tile to SMEM, compare the SMEM layout with the manual cp.async+swizzle version. If they differ, update the ldmatrix address computation for K and V fragments.

### Q Loading

Q loading (128×128 tile, done once per block) currently uses cp.async with 128 threads. Options:
1. **Keep cp.async for Q** — it's done once, minimal impact
2. **Use TMA for Q** — single instruction, cleaner code

Recommendation: keep cp.async for Q in this iteration. TMA for Q can be a future optimization.

## Expected Impact

- Eliminates ~100 cp.async instructions per tile × 2 tiles (K+V) × 256 iterations (noncausal S=16384) = ~51K instructions removed from DMA warp critical path
- Hardware address generation replaces per-thread index arithmetic
- Better L2 utilization — TMA generates minimal sector requests
- DMA warp becomes nearly idle (one instruction per tile vs 32 iterations)

**Estimated improvement:** 5-10%, bringing noncausal to ~93-97% and causal to ~97-99%.

## Risk Mitigation

1. **Descriptor API complexity:** The `cuTensorMapEncodeTiled` API has many parameters. Use the exact pattern from CUTLASS SM120 GEMM as reference. Test descriptor creation with a standalone copy kernel before integrating into FA4.
2. **mbarrier semantics:** mbarrier has parity-based reuse (not simple arrive/wait). Follow CUTLASS `PipelineTmaAsync` pattern with explicit phase tracking.
3. **Swizzle mismatch:** TMA's hardware swizzle may differ from our manual swizzle. Must verify and potentially update ldmatrix address computation.
4. **SMEM alignment:** TMA requires 128-byte aligned SMEM destinations. Pad SMEM layout as needed.
5. **`tensormap.replace` PTX:** Device-side descriptor modification uses PTX inline asm. Must match the correct encoding for SM120.

## Files Modified

- `conf/fixtures/fa4/generated.cu` — kernel rewrite (TMA loads + mbarrier sync)
- No changes to reference.py, configs.json, or eval harness

## Verification

1. **Compile:** 0 errors, check register count stays at 255, spill ≤ 24 bytes
2. **Correctness:** smoke test one config (compare output sample vs cp.async version)
3. **Benchmark:** all 8 configs, compare Gen/FA4 ratio vs baseline
4. **NCU profile:** check tensor pipe %, GMEM sectors, local memory spill, SMEM wavefronts
