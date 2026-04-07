# SM90 Matmul — v26: Vectorized 128-bit Epilogue

**Date**: 2026-04-07 05:30
**Host**: devvm8491 (h8_4), GPU 4,5, NVIDIA H100 SXM R&R SKU 96GB
**CUDA**: 13.2, Driver 595.45.04

## Changes (v26 vs v25)

1. **128-bit vectorized SMEM→GMEM**: Use uint4 (16 bytes = 8 BF16) loads and
   stores in the epilogue read phase. 8 threads cover one row (8×8=64 cols),
   4 warps × 4 rows/warp = 16 rows per iteration, 4 iterations for 64 rows.
   Reduces loop iterations from 16 to 4, cutting instruction count by ~75%.

2. **Remove trailing barrier on last N-tile**: The 4th N-tile (acc3) no longer
   issues a trailing barrier since there's no subsequent SMEM reuse. Saves
   1 barrier (~20 cycles) per consumer per tile.

## Results (formal bench)

| Config      | v25 TF | v26 TF | cuBLAS TF | v26/cuBLAS |
|-------------|--------|--------|-----------|------------|
| 256×256     | 3.0    | 3.2    | 3.0       | 1.07×      |
| 512×512     | 19.8   | 20.6   | 22.2      | 0.93×      |
| 1024×1024   | 109.5  | 111.3  | 96.1      | 1.16×      |
| 2048×2048   | 567.5  | 579.8  | 428.1     | 1.35×      |
| 4096×4096   | 721.5  | 724.2  | 694.5     | 1.04×      |
| 8192×8192   | 762.9  | 770.3  | 760.3     | 1.01×      |

**Peak: 770.3 TFLOPS (96.3% of 800 TF)** — up from 762.9 (95.4%)

## 512×512 Gap Analysis

512×512 remains at 0.93× cuBLAS. Root cause: structural tile mismatch.
- Our 128×256 tile → grid 4×2 = 8 tiles for 132 SMs (94% idle)
- cuBLAS dynamically selects smaller tiles for small problems
- Kernel launch overhead (~3-5 μs) dominates the 13 μs total

Closing this gap requires a separate small-matrix kernel or dynamic tile
selection, not epilogue tuning.
