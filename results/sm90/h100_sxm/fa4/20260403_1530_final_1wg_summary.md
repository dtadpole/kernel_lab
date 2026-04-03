# FA4 SM90 Optimization Summary — Session 2026-04-03

## Final Performance (1 Consumer WG)

| Config | mma.sync baseline | TMA+wgmma+overlap | Speedup | cuDNN | vs cuDNN |
|--------|-------------------|-------------------|---------|-------|----------|
| nc-b8-s4096 | 222 | 436 | 1.96x | ~646 | 67% |
| nc-b4-s8192 | 218 | 457 | 2.10x | ~607 | 75% |
| nc-b2-s16384 | 226 | 471 | 2.08x | ~593 | 79% |
| c-b8-s4096 | 191 | 357 | 1.87x | ~579 | 62% |
| c-b4-s8192 | 199 | 388 | 1.95x | ~615 | 63% |
| c-b2-s16384 | 206 | 412 | 2.00x | ~615 | 67% |
| **Average** | **210** | **420** | **2.00x** | **~609** | **69%** |

## Optimization History

1. **TMA + wgmma + QK/PV overlap** (220 -> 326 TFLOPS, 1.55x)
   - TMA replaces cp.async, zero SMEM rearrangement overhead
   - wgmma.mma_async RS variant (async tensor core)
   - QK and PV overlap via wgmma.wait_group(1) + deferred O rescaling

2. **BLOCK_KV 64->128** (326 -> 399 TFLOPS, 1.23x)
   - Halved KV loop iterations
   - Single softmax pass over 128 positions

3. **wgmma.m64n128k16 for QK** (399 -> 420 TFLOPS, 1.05x)
   - Halved QK wgmma call count (16 -> 8)
   - Unified S_acc[64] register array

4. **2 Consumer WGs (FAILED)** — attempted tile_m=128 with 2 WGs
   - Achieved 504-530 TFLOPS on first launch
   - Hangs with causal masking and on repeated launches
   - Root cause: bar.sync scheduler barriers deadlock
   - Reverted to 1-WG kernel

## Architecture (Final)

- 256 threads: 1 producer WG + 1 consumer WG
- BLOCK_Q=64, BLOCK_KV=128, DIM=128
- TMA with SWIZZLE_128B, split-DIM half-tiles
- wgmma: m64n128k16 for QK, m64n64k16 for PV
- 2-stage K/V pipeline with mbarrier
- 0 spills, 255 registers, no ptxas warnings

## Remaining Gap (31% to cuDNN)

The 1-WG architecture is at its ceiling. To reach cuDNN (~600 TFLOPS):
- **2 Consumer WGs** — requires correct inter-WG barrier protocol (CuTe DSL
  uses NamedBarrierFwd with canonical_warp_group_idx, not simple bar.sync)
- **PV m64n128k16** — blocked by SWIZZLE_128B row width constraint (V half-tiles
  have 64-column rows, m64n128k16 needs 128 columns)
