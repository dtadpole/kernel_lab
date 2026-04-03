# SM90 Matmul — SMEM-Buffered Coalesced Epilogue

## Summary

Replaced direct `__nv_bfloat162` global stores in the epilogue with an
SMEM-buffered approach: write F32→BF16 fragments to padded SMEM, then issue
128-bit vectorized coalesced stores from SMEM to GMEM. This fixes the poor
store coalescing in the original epilogue where threads within a warp wrote to
8 different rows (8 cache sectors per store vs ideal 1).

## Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA H100 SXM |
| Architecture | SM 9.0 (Hopper) |
| SMs | 132 |
| Peak BF16 Tensor Core | ~990 TFLOPS (dense) |
| Host | devvm8490 |
| GPU Index | CUDA_VISIBLE_DEVICES=4 |
| CUDA Toolkit | 12.8 |

## Change

### Problem: Uncoalesced epilogue stores

The WGMMA m64n256k16 fragment layout maps threads within a warp to 8 different
output rows (`groupID = lane/4`). Direct global stores from this layout produce
8 cache line transactions per warp per store instead of 1 — only 50% sector
efficiency.

### Solution: SMEM-buffered coalesced stores

1. After `wgmma_wait_group_0()`, `__syncthreads()` to ensure both warpgroups
   are done with mainloop SMEM
2. Write F32→BF16 fragments to padded SMEM (reuses mainloop buffer, zero extra
   SMEM allocation). Padding (stride = TILE_N + 8) avoids bank conflicts:
   consecutive groupIDs hit banks 4 apart (264*2/4 mod 32 = 4)
3. `__syncthreads()` to make all writes visible
4. 8 warps × 16 rows each, 32 threads write 256 columns per row as 128-bit
   (`uint4`) stores — perfectly coalesced, 1 sector per warp per store

### Compile stats (unchanged)

- `wgmma_matmul`: 154 registers, 0 spills, 0 warnings
- `wgmma_matmul_small`: 90 registers, 0 spills, 0 warnings

## Performance (TFLOPS, median of 20 trials)

| Config | cuBLAS | CuTe DSL | Gen (before) | Gen (after) | Δ Gen | Gen/cuBLAS |
|--------|--------|----------|-------------|------------|-------|------------|
| 256×256 | 1.8 | — | 1.8 | 1.8 | ~0% | 98% |
| 512×512 | 13.6 | — | 13.3 | 13.3 | ~0% | 98% |
| 1024×1024 | 68.8 | 91.6 | 77.9 | **86.4** | **+10.9%** | **126%** |
| 2048×2048 | 352.7 | 482.4 | 422.7 | **470.1** | **+11.2%** | **133%** |
| 4096×4096 | 606.4 | 667.5 | 643.4 | **688.4** | **+7.0%** | **114%** |
| 8192×8192 | 742.5 | 735.8 | 687.7 | **710.9** | **+3.4%** | **96%** |

## Key Findings

1. **3-11% improvement across all compute-heavy configs** — largest gains at
   medium sizes where epilogue is a bigger fraction of total time
2. **4096 now beats CuTe DSL** (688.4 vs 667.5 = +3.1%) — coalesced stores
   are more efficient than CuTe DSL's stmatrix + TMA S2G at this size
3. **8192 at 96% of cuBLAS** (up from 92%) — remaining gap is instruction
   scheduling quality (ptxas vs cuBLAS's hand-optimized SASS)
4. **Zero register pressure change** — 154 registers, 0 spills (identical to
   baseline). The SMEM epilogue adds no register overhead
5. **Two `__syncthreads()` added** but the coalescing benefit far outweighs
   the barrier cost (64KB of stores going from 8 sectors → 1 sector per warp)

## Why this works

WGMMA accumulator registers are organized by `groupID` (8 row groups per warp).
Direct stores scatter across 8 cache lines. Going through SMEM lets us decouple
the register layout from the store layout, enabling arbitrary coalescing patterns.

The padding trick (stride = TILE_N + 8) ensures:
- R2S writes: adjacent groupIDs hit banks 4 apart (no bank conflicts)
- S2G reads: consecutive lanes read consecutive 16-byte chunks (no conflicts)
- Alignment: 264 × 2 = 528 bytes per row, divisible by 16 (uint4 aligned)
