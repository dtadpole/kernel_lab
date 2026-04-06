# Plan: SM90 Matmul — Beat cuBLAS (715 → 760+ TFLOPS)

**Current**: 715 TFLOPS (1.53ms), 97% of cuBLAS
**cuBLAS**: 740 TFLOPS (1.45ms), kernel `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_NNT`
**Target**: 760+ TFLOPS — **surpass cuBLAS**

## NCU Gap Analysis (fact-based)

```
                         cuBLAS    Ours      Gap     Fix
Duration                 1.45ms   1.53ms   +0.08ms
dispatch_stall           0.3%     4.4%     +4.1%   → Persistent kernel
lg_throttle              0.0%     4.0%     +4.0%   → TMA store
SM throughput            93.5%    88.4%    -5.1%   → Both above
SMEM LSU                 94.6%    88.2%    -6.4%   → More WGMMA per overhead
```

Closing dispatch_stall (0.07ms) + lg_throttle (0.06ms) = 0.13ms > 0.08ms gap.

## Phase 1: Persistent Kernel (target: dispatch_stall 4.4% → <1%)

### What
Launch exactly `num_SMs` (132) blocks. Each block loops through multiple output tiles
via atomic counter. cuBLAS uses grid=132 — this is the #1 architectural difference.

### Why
With grid=2048, the GPU scheduler launches/retires 2048 blocks. Each launch has overhead
(register file init, SMEM alloc, scheduler dispatch). With 132 persistent blocks, each
block is launched once and processes ~16 tiles internally. dispatch_stall drops from
4.4% to ~0.3%.

### Design
The previous persistent attempt (commit reverted) failed because of per-tile mbar
re-init overhead. New approach: **don't re-init barriers**. Track phase across tiles.

Key insight: after processing one tile with `numK` K-steps, the mbar phases are
deterministic. If `numK` is the same for all tiles (true for square matmul), the
phase state after each tile is identical. We just need to track the cumulative phase.

```
mbar_full[s] phase after 1 tile:
  Stage s was used ceil(numK/STAGES) or floor(numK/STAGES) times.
  Phase parity = usage_count & 1.
  For numK=128, STAGES=4: stages 0-3 used 32 times each. Phase parity = 0 (even).
  → All mbar_full return to phase 0 after each tile! No re-init needed!

mbar_empty[s] phase after 1 tile:
  Consumer signals once per K-step (for previous stage) + once at drain.
  Same analysis: each stage signaled the same number of times.
  → All mbar_empty also return to phase 0 after each tile!
```

For square matrices (M=N=K), numK is always a multiple of STAGES when STAGES divides K/TILE_K.
For 8192: numK=128, STAGES=4 → 128/4=32 (exact). Phases cycle back to 0. **No re-init needed.**

For non-multiple cases: add re-init only when numK % STAGES != 0.

### Inter-tile sync (lightweight)
- Producer: after K-loop, waits for consumer to signal last mbar_empty. Then gets next
  tile via atomicAdd, writes tile_info to SMEM, prefills TMA, arrives mbar_tready.
- Consumer: after epilogue, waits mbar_tready, reads tile_info, enters K-loop.
- mbar_tready: count=1 (producer only), auto-cycles. No re-init.
- Total sync per tile: 1 mbar arrive + 1 mbar wait. No CTA-wide barriers.

### SMEM additions
```
mbar_tready[1]:  16 bytes (count=1, producer signals tile ready)
tile_info[3]:    12 bytes (ctaRow, ctaCol, tile_id)
tile_counter:    device memory, 4 bytes (atomic counter for tile scheduling)
```

### Implementation
1. Add `__device__ unsigned* g_tile_counter` parameter
2. Add mbar_tready + tile_info to SMEM layout
3. Wrap producer K-loop in outer tile loop
4. Wrap consumer K-loop in outer tile loop
5. Producer: atomicAdd → tile_info → prefill → arrive mbar_tready
6. Consumer: wait mbar_tready → read tile_info → K-loop → epilogue → loop back
7. Host: cudaMemsetAsync(tile_counter, 0), launch 132 blocks

### Critical detail: producer must wait for consumers to finish SMEM
After the producer's K-loop ends, the consumers are still processing the last few
K-steps. The producer must wait for the last mbar_empty signal before prefilling
TMA for the next tile. This is already handled by the K-loop structure:
- Producer's last K-loop iteration waits on mbar_empty for the stage it wants to reuse
- After the loop, the producer needs to wait for the REMAINING mbar_empty signals
  (from consumer's final K-steps + drain)

Solution: after producer K-loop, explicitly wait for mbar_empty on the last
stage used by the consumer. Since phases cycle back to 0, the parity for this
wait is computable.

### Phase parity for inter-tile mbar_tready
- Tile 0: consumer waits parity 0 (init phase)
- Tile 1: consumer waits parity 1
- Pattern: wait parity = tile_counter & 1

### Bounds check
If tile_id >= total_tiles, both producer and consumer exit immediately.
The extra CTA (if total_tiles < 132) does nothing.

### Verify
`ik:bench matmul` — all 6 configs must pass ✓.

---

## Phase 2: TMA Store Epilogue (target: lg_throttle 4.0% → 0%)

### What
Replace scalar `st.global.b32` epilogue with register → SMEM → TMA bulk store.
cuBLAS has lg_throttle = 0% because it uses TMA store.

### Why
Current epilogue: 1M warp-level store instructions. Each goes through the LSU
pipeline, causing lg_throttle. TMA store uses the TMA engine (separate hardware
unit) for async bulk GMEM writes, bypassing LSU entirely.

### Design
After WGMMA drain:
1. Convert FP32 accumulators to BF16 and write to SMEM (reuse stage 0's buffer)
2. Issue TMA store: `cp.async.bulk.tensor.2d.shared::cta.global.tile [gmem_desc, {coords}], [smem_addr]`
3. Wait for TMA store completion

### SMEM layout for epilogue
Output tile per consumer: 64×256 = 32KB (BF16). Fits in one stage's A buffer (16KB)
+ part of B buffer (16KB) = 32KB. Or use the full 48KB stage buffer.

But the SMEM layout for TMA store must match the TMA store descriptor's swizzle.
The output C is row-major (N contiguous). TMA store with 128B swizzle on a
64-wide or 256-wide tile.

### Register → SMEM packing
WGMMA m64n64k16 output register mapping:
- Each thread holds elements at specific (row, col) positions
- Need to pack these into SMEM in the row-major layout expected by TMA store
- This requires a transpose/scatter from register layout to SMEM layout

The `stmatrix` instruction (`STSM` in SASS) does exactly this:
- `stmatrix.sync.aligned.m8n8.x4.shared.b16 [smem_addr], {r0, r1, r2, r3}`
- Scatters 4 registers worth of data into SMEM in the matrix layout
- This is what CuTe DSL uses: `StMatrix8x8x16bOp`

### TMA store descriptor
```cpp
// C is M×N row-major, BF16
cuuint64_t dims[2] = {N, M};
cuuint64_t str[1]  = {N * 2};
cuuint32_t box[2]  = {TILE_NQ, 64};  // store 64 rows × 64 cols per TMA op
cuuint32_t el[2]   = {1, 1};
s_encodeTiled(&tma_C, BF16, 2, C, dims, str, box, el,
    NONE, SWIZZLE_128B, L2_NONE, OOB_FILL_NONE);
```

### Implementation
1. Add TMA store descriptor creation in kernel_run
2. Pass tma_C descriptor to kernel
3. After WGMMA drain: `stmatrix` to write accumulators to SMEM
4. Issue TMA store from SMEM to GMEM
5. Wait for TMA store completion (mbarrier or cp.async.bulk.wait)

### Risk
High complexity. The register → SMEM packing must exactly match the TMA store
descriptor's expected layout. Getting the stmatrix addressing wrong produces
silent corruption.

### Verify
`ik:bench matmul` — all 6 configs must pass ✓.

---

## Phase 3: Cluster + TMA Multicast (target: long_scoreboard 19% → ~10%)

### What
Launch with cluster size 1×2. Two SMs share B tiles via TMA multicast.
cuBLAS uses `1x2` cluster with `coopB` (cooperative B loading).

### Why
B accounts for 67% of TMA traffic. Multicast halves B DRAM reads.
Reduces long_scoreboard (DRAM latency stalls) by ~33%.

### Design
- `__cluster_dims__(1, 2, 1)` — 2 CTAs in N-dimension share B
- Both CTAs in cluster use SAME N-column but DIFFERENT M-rows (already the case with CTA swizzle)

Wait — cuBLAS uses `1x2` which means cluster in N-dimension. But both CTAs in
our cluster should share the SAME B tile (same N-column). If cluster is 1×2,
the two CTAs have different N but need to load B for their respective N-columns.
That doesn't help for multicast.

Actually, re-reading: `coopB` means B is cooperatively loaded. With cluster 1×2,
two CTAs at different N-positions can share parts of B if they have overlapping
K-tile loads. But for our setup, each CTA loads a different B N-column.

Let me reconsider: cluster 2×1 means 2 CTAs with different M-rows but same N-column.
They share B data (same N-column) → B multicast works.

cuBLAS name: `320x128_64x3_1x2`. The `1x2` might mean cluster_M=1, cluster_N=2.
With tile 320×128 and cluster 1×2: each cluster covers 320×256 of output.
Two CTAs at adjacent N-columns → they DON'T share B directly.

But `coopB` means the two CTAs cooperatively load B. Each CTA loads HALF of
the B tile and multicasts to the other CTA. This way, each CTA does half
the B DRAM reads.

### Implementation (deferred)
Requires `cudaLaunchKernelEx` for cluster configuration.
Complex: cluster barrier init, TMA multicast addressing, CTA swizzle adjustment.
Only attempt after Phase 1+2 are working.

---

## Execution Order

Phase 1 (Persistent) → Phase 2 (TMA Store) → Phase 3 (Cluster)

Each phase independently reduces a different stall type.
Phase 1+2 together should close the 0.08ms gap.
Phase 3 provides headroom to surpass cuBLAS.

## Critical Rules

1. One phase at a time. Verify before moving on.
2. `ik:bench matmul` all ✓ before committing.
3. NCU profile after each phase to confirm stall reduction.
4. Debug forward — if persistent kernel produces wrong results, debug the phase tracking.
5. For square matrices (M=N=K), numK is always K/TILE_K. Verify phases cycle back to 0.
