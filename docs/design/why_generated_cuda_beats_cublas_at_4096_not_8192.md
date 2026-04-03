# Why Generated CUDA Beats cuBLAS by 15% at 4096 but Only 1% at 8192

## TL;DR

Our hand-written WGMMA kernel wins on **per-tile overhead** (persistent scheduling, warp specialization, persistent barriers). cuBLAS wins on **mainloop WGMMA scheduling** (hand-tuned SASS). At 4096, overhead is 14% of total time — our advantage dominates. At 8192, overhead shrinks to 8% — cuBLAS's better mainloop scheduling catches up.

## Benchmark Numbers (H100 SXM, BF16)

| Size | cuBLAS (TFLOPS) | Generated (TFLOPS) | Speedup |
|------|----------------|--------------------|---------| 
| 4096 | 642.5 | 741.8 | **1.15x** |
| 8192 | 768.7 | 779.4 | **1.01x** |

## The Two Competing Forces

### Force 1: Our Kernel Wins on Overhead

Three architectural advantages that reduce per-tile overhead:

1. **Persistent scheduling** — 132 CTAs (1/SM) loop over tiles. No CTA launch/teardown between tiles. cuBLAS likely uses wave scheduling with inter-wave gaps.

2. **Warp specialization (3 WGs)** — Dedicated producer WG runs TMA continuously while 2 consumer WGs run WGMMA. The producer-consumer separation eliminates pipeline bubbles at tile boundaries. In cooperative mode (cuBLAS-style), the same threads do both TMA and WGMMA, serializing at transitions.

3. **Persistent barriers** — mbarrier auto-reset is exploited to skip teardown/rebuild between tiles. This saves 8x `mbarrier_inval` + 8x `mbarrier_init` + 1x `__syncthreads` + 4x `mbarrier_arrive` per tile boundary.

### Force 2: cuBLAS Wins on Mainloop Scheduling

cuBLAS uses **hand-tuned SASS** (not compiled through ptxas). The WGMMA instruction scheduling is tighter:

- Better interleaving of WGMMA fence/commit/wait with descriptor computation
- `setmaxnreg` register redistribution works properly (our kernel's ptxas ignores it, allocating 154 regs uniformly instead of 40 producer / 216 consumer)
- Potentially uses `wgmma.wait_group<1>` with proper pipeline depth compensation

## Why the Gap Changes with Size

Each output tile is 128x256 (fixed). As the matrix grows, only K grows:

```
                         4096×4096         8192×8192
                        ──────────        ──────────
K-tiles per tile:           64                128
Mainloop compute:        ~42 µs/tile       ~84 µs/tile    ← scales with K
Per-tile overhead:        ~7 µs/tile        ~7 µs/tile    ← FIXED
Overhead fraction:         14.2%             7.7%
Tiles per SM:              3.9               15.5
```

### At 4096: Overhead Dominates → Our Advantage Shines

- Mainloop is short (64 K-steps = ~42 µs per tile)
- Per-tile overhead (~7 µs) is **14.2%** of tile time
- Our persistent scheduling + warp specialization eliminates most of this 14%
- cuBLAS's better WGMMA scheduling saves maybe 2-3% in the mainloop
- Net: **+15% for us** (overhead wins >> mainloop loss)

### At 8192: Mainloop Dominates → Advantage Disappears

- Mainloop is long (128 K-steps = ~84 µs per tile)
- Per-tile overhead (~7 µs) is only **7.7%** of tile time
- Even eliminating ALL overhead would only gain 7.7%
- cuBLAS's hand-tuned SASS recovers ~6% in the mainloop
- Net: **+1% for us** (overhead wins ≈ mainloop loss)

## Convergence at Peak

Both kernels converge toward the same tensor core throughput ceiling (~78% of H100's 989.5 TFLOPS peak). The remaining 22% gap to theoretical peak comes from:

| Source | Estimated | Addressable? |
|--------|-----------|-------------|
| SM WGMMA scheduling overhead | ~15% | Needs hand-tuned SASS |
| Tile load imbalance (2048 mod 132) | ~6% | Needs stream-K |
| Epilogue + sync | ~1% | Already minimized |

## Visual Model

```
4096×4096 tile timeline:
├── mainloop (42 µs) ──────────────────┤ overhead (7 µs) ┤
                                        ^^^^^^^^^^^^^^^^
                                        14% — we eliminate most of this

8192×8192 tile timeline:
├── mainloop (84 µs) ──────────────────────────────────────────────┤ overhead (7 µs) ┤
                                                                    ^^^^^^^^^^^^^^^^
                                                                    8% — small target
```

## Implication

To push further at 8192, we need to improve the **mainloop**, not the overhead:
- `setmaxnreg` fix (separate compilation units) for proper register redistribution
- Hand-tuned SASS for optimal WGMMA scheduling
- Stream-K for better tile load balancing across 132 SMs
