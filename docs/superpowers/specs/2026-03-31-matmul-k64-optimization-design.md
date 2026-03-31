# Matmul Kernel Optimization: K-tile 64 + Pipeline Tuning

**Date:** 2026-03-31
**Status:** Implementation-ready

## Problem

The generated matmul kernel reaches only 58-65% of peak BF16 tensor throughput (503.8 TFLOPS) on large matrices (4096+). The CuTe DSL reference achieves ~72% on the same hardware. Key bottleneck: the K-tile is 32 BF16 elements, causing excessive pipeline iterations and synchronization overhead.

## Root Cause Analysis

For 8192x8192 with K-tile=32:
- 256 pipeline iterations, each with `__syncthreads()` + `cp.async.wait_group`
- 32 HMMA instructions per iteration (low compute-to-sync ratio)
- 4 cp.async per thread per stage (low memory-to-sync ratio)

The CuTe reference uses K-tile=64 with only 128 iterations, doubling the compute per sync barrier.

## Design

### Change 1: Double K-tile from 32 to 64

**SMEM layout change:**
- Old: `As[N_STAGES * 64][8]` — 64 SMEM rows per matrix per stage (128 A-rows × 32 K-elements)
- New: `As[N_STAGES * 128][8]` — 128 SMEM rows per matrix per stage (128 A-rows × 64 K-elements)

**SMEM row mapping (per stage, 128 rows for A):**
- Rows 0-31: A rows 0-63, K elements [0..31] (first K-half)
- Rows 32-63: A rows 64-127, K elements [0..31] (first K-half)
- Rows 64-95: A rows 0-63, K elements [32..63] (second K-half)
- Rows 96-127: A rows 64-127, K elements [32..63] (second K-half)

**cp.async per thread per stage:** 4 for A + 4 for B = 8 total (was 2+2=4)

**MMA inner loop:** Split into two K-halves, each reusing the same register arrays:
1. Load first K-half (k=0..31) from SMEM rows 0-63
2. Issue cp.async for next pipeline stage (overlaps with MMA)
3. Compute first K-half (32 HMMA)
4. Load second K-half (k=32..63) from SMEM rows 64-127
5. Compute second K-half (32 HMMA)

**Register budget:** Unchanged at ~128/thread (reuse aReg/bReg between K-halves).

### Change 2: Keep 3-stage pipeline

With K=64, 3-stage pipeline gives `wait_group(1)` — 1 group can be in-flight while computing, providing good overlap.

**SMEM cost:** 3 stages × 2 matrices × 128 rows × 8 cols × 16 bytes = 98,304 bytes (96 KB). Fits in Blackwell's 163 KB static SMEM limit.

**Occupancy:** 1 block per SM (96 KB > 64 KB half of 128 KB). With 188 SMs, large matrices (4096+ blocks) still have abundant parallelism.

### What stays the same

- Transpose kernel (B pre-transpose)
- Tile size: 128×128 output
- Thread block: 256 threads (16×16)
- MMA atom: m16n8k16
- All helper functions (cp_async, ldmatrix, mma)
- Epilogue (FP32→BF16 store)
- kernel_run entry point

## Expected Impact

- **Pipeline iterations halved** (e.g., 256→128 for 8192x8192)
- **Compute-per-sync doubled** (64 HMMA per iteration vs 32)
- **Estimated improvement:** ~10% for large matrices, potentially matching the reference (~360 TFLOPS)
- **Small matrix regression risk:** Low (minimal iteration count anyway)

## Constraints

- K must be multiple of 64 (all current configs satisfy this)
- M, N must be multiple of 128 (unchanged)
