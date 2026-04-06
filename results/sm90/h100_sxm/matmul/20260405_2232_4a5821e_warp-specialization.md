# SM90 Matmul — 3-WG Warp Specialization

**Commit**: 4a5821e  
**Date**: 2026-04-05 22:32  
**Host**: devvm8490 (h8_3), GPU 4, NVIDIA H100 SXM5 96GB  
**CUDA**: 13.2, Driver 595.45.04  
**Harness**: eval_harness (cold-L2, fresh pointers, 10 warmup + 20 trials)

## Architecture

- 3 warpgroups (384 threads): 1 producer + 2 consumers
- WG0 = producer: TMA loads only, setmaxnreg.dec 40
- WG1 = consumer0: WGMMA rows 0-63, setmaxnreg.inc 232
- WG2 = consumer1: WGMMA rows 64-127, setmaxnreg.inc 232
- CTA tile: 128×128, TILE_K=64, 5-stage pipeline
- B loaded MN-major (no transpose) via split TMA: two 64-col loads per stage
- WGMMA m64n64k16 with trans-b=1
- Dual mbarrier: mbar_full (producer→consumers) + mbar_empty (consumers→producer)
- 168 registers, 0 spills, 0 stack frame

## Results

| Config       | Median (ms) | TFLOPS | Correctness |
|-------------|------------|--------|-------------|
| 256×256     | 0.0120     | 2.8    | OK          |
| 512×512     | 0.0133     | 20.2   | OK          |
| 1024×1024   | 0.0171     | 125.4  | OK          |
| 2048×2048   | 0.0429     | 400.1  | OK          |
| 4096×4096   | 0.2640     | 520.5  | OK          |
| 8192×8192   | 2.2698     | 484.4  | OK          |

Peak: **520 TFLOPS** at 4096×4096 (65% of H100 peak 800 TFLOPS)

## Comparison vs Previous (1-WG, no transpose)

| Config       | 1-WG TFLOPS | 3-WG TFLOPS | Change |
|-------------|------------|------------|--------|
| 256×256     | 2.6        | 2.8        | +8%    |
| 512×512     | 18.3       | 20.2       | +10%   |
| 1024×1024   | 105.4      | 125.4      | +19%   |
| 2048×2048   | 353.7      | 400.1      | +13%   |
| 4096×4096   | 494.9      | 520.5      | +5%    |
| 8192×8192   | 544.0      | 484.4      | -11%   |

3-WG wins at small-medium sizes (up to +19%). 1-WG wins at 8192 (-11%).

## Tuning Explored

| Parameter | Values Tested | Best |
|-----------|--------------|------|
| STAGES | 3, 4, 5, 6, 7, 8 | 5 |
| wait_group | 1, 2 | 1 |
| setmaxnreg | dec 24/inc 240, dec 40/inc 232 | 40/232 |

- More stages improve performance monotonically up to 5-6 (then plateau)
- 8 stages exceeds H100's 228KB SMEM limit → corruption
- wait_group 2 slightly worse than 1 (more latency)
- setmaxnreg values don't significantly affect performance

## Key Bug Found

mbar_empty (consumers→producer) was initialized with count=2 (expecting 1 arrive
per consumer warpgroup), but ALL 128 threads per consumer called `mbarrier.arrive`.
This caused 256 arrives on a count-2 barrier, corrupting the phase tracking.
Fix: only `local_tid == 0` per consumer warpgroup calls arrive.

## Analysis

The 3-WG warp-specialized kernel trades large-matrix performance for small/medium
gains. The barrier synchronization overhead (mbar_full wait + mbar_empty arrive)
adds per-K-step latency that exceeds the benefit of TMA/WGMMA decoupling.

To reach 650+ TFLOPS would require:
- Larger tiles (128×256) to amortize barrier overhead
- Persistent kernels with cooperative launch
- TMA multicast for B matrix
