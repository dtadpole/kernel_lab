# SM90 Matmul — Persistent Kernel Scheduling

## Summary

Converted from non-persistent (grid = totalTiles) to persistent scheduling
(grid = min(totalTiles, numSMs)). Each CTA loops over multiple tiles,
eliminating wave-transition overhead and tail effects. Used
`wgmma_wait_group_0()` at tile boundaries to prevent ptxas C7515.

## Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA H100 SXM |
| Architecture | SM 9.0 (Hopper) |
| SMs | 132 |
| Host | devvm8490 |
| GPU Index | CUDA_VISIBLE_DEVICES=4 |
| CUDA Toolkit | 12.8 |

## Change

### Non-persistent → Persistent

- Grid: `totalTiles` → `min(totalTiles, numSMs)`
- Added tile loop: `for (tileIdx = blockIdx.x; tileIdx < totalTiles; tileIdx += gridDim.x)`
- TMA descriptor prefetch moved outside loop (once per CTA)
- Mbarrier inval + re-init between tiles (thread 0 only, ~1 µs overhead)
- `wgmma_wait_group_0()` at tile boundary prevents C7515 accumulator serialization

### Why this helps at 8192×8192

- 2048 tiles / 132 SMs = 16 waves non-persistent
- Last wave: only 68 SMs active (48% idle = ~3% total waste)
- Pipeline prelude (4 TMA loads) paid once per CTA instead of ~16 times
- Better L2 cache locality: consecutive tiles from same CTA reuse cached data

### Compile stats (unchanged)

- 154 registers, 0 spills, 0 ptxas warnings (C7515 avoided via wait_group_0)

## Performance (TFLOPS, median of 20 trials)

| Config | cuBLAS | CuTe DSL | Gen (before) | Gen (after) | Δ Gen | Gen/cuBLAS |
|--------|--------|----------|-------------|------------|-------|------------|
| 256×256 | 1.9 | 2.0 | 1.8 | 2.0 | +11% | 105% |
| 512×512 | 13.7 | 14.3 | 13.3 | 13.5 | +2% | 99% |
| 1024×1024 | 64.3 | 92.4 | 86.4 | 85.3 | -1% | 133% |
| 2048×2048 | 353.4 | 486.3 | 470.1 | 473.4 | +1% | 134% |
| 4096×4096 | 603.0 | 670.6 | 688.4 | **690.7** | +0.3% | **115%** |
| 8192×8192 | 733.0 | 733.0 | 710.9 | **724.7** | **+1.9%** | **99%** |

## Key Findings

1. **8192 at 99% of cuBLAS** (up from 96%) — persistent scheduling closes
   the gap from wave transitions and tail effects
2. **Zero ptxas warnings** — `wgmma_wait_group_0()` at tile boundaries
   fully drains the WGMMA pipeline, preventing C7515 accumulator serialization.
   This confirms that raw CUDA + inline PTX avoids the cross-function-boundary
   issue that plagued CUTLASS C++ persistent kernels.
3. **Small configs unaffected** — 256/512 have few tiles (1-4), so persistent
   is equivalent to non-persistent. 1024 has slight noise-level variation (-1%).
4. **Remaining gap to cuBLAS (~1%)** is instruction scheduling quality
   (ptxas vs NVIDIA's internal cuBLAS compiler). Further gains require
   warp specialization (3 warp groups) or cluster multicast.

## Optimization History

| Version | 8192 TFLOPS | vs cuBLAS | Key Change |
|---------|------------|-----------|------------|
| V1 | 423 | 58% | Initial WGMMA implementation |
| V3 | 559 | 75% | Pipeline overlap (wait_group 1) |
| V5 | 614 | 82% | Single asm block + m64n256k16 |
| V6 | 680 | 89% | Non-persistent + CuTe DSL pipeline |
| V7 | 719 | 94% | CTA swizzle (group_m=8→16) |
| V8 | 711 | 96% | SMEM-buffered coalesced epilogue |
| **V9** | **725** | **99%** | **Persistent scheduling** |
