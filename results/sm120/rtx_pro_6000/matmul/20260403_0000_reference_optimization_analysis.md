# Matmul Reference (CuTe DSL) Optimization Analysis — 2026-04-03

## Objective

Analyze and optimize the CuTe DSL reference matmul implementation, comparing
against the hand-written generated kernel and cuBLAS.

## Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA RTX PRO 6000 Blackwell |
| Architecture | SM 12.0 (Blackwell) |
| SMs | 188 |
| Peak BF16 Tensor Core | 503.7 TFLOPS |
| DRAM Bandwidth | 1,536 GB/s |
| L2 Cache | 96 MB |
| SMEM per SM | 100 KB (configurable) |

## Baseline Performance

| Config | FLOPs | Reference (CuTe) | Generated (PTX) | cuBLAS | ref/cuBLAS | gen/cuBLAS |
|--------|-------|---|---|---|---|---|
| | | ms / TFLOPS | ms / TFLOPS | ms / TFLOPS | | |
| 256x256 | 33.6M | 0.0152 / 2.2 | 0.0071 / 4.7 | 0.0163 / 2.1 | 1.07x | 2.30x |
| 512x512 | 268M | 0.0190 / 14.1 | 0.0091 / 29.5 | 0.0169 / 15.9 | 0.89x | 1.86x |
| 1024x1024 | 2.15G | 0.0249 / 86.3 | 0.0136 / 158.1 | 0.0212 / 101.4 | 0.85x | 1.56x |
| 2048x2048 | 17.18G | 0.0673 / 255.3 | 0.0578 / 297.2 | 0.0641 / 268.0 | 0.95x | 1.11x |
| 4096x4096 | 137.4G | 0.3407 / 403.3 | 0.3257 / 421.9 | 0.3282 / 418.7 | 0.96x | 1.01x |
| 8192x8192 | 1099.5G | 2.6663 / 412.4 | 2.4388 / 450.9 | 2.4490 / 449.0 | 0.92x | 1.00x |

## NCU Architecture Comparison (4096x4096)

| Metric | Reference (CuTe) | Generated (PTX) | cuBLAS |
|--------|---|---|---|
| Tile shape | 128x128x64 | 256x128x32 | 256x128x32 |
| Threads | 160 (5 warps) | 256 (8 warps) | 256 (8 warps) |
| Warp specialization | Yes (1 DMA + 4 MMA) | No (all MMA) | No |
| Registers/thread | 207 | 191 | 218 |
| Dynamic SMEM | 99.3 KB | 73.9 KB | 73.7 KB |
| Achieved occupancy | 10.41% | 16.63% | 16.65% |
| Active warps/SM | 4.99 | 7.98 | 7.99 |
| Compute throughput | 84.2% | 85.1% | 86.7% |
| Memory throughput | 71.3% | 54.0% | 55.4% |
| Arithmetic intensity | 62.5 FLOPs/B | 83.3 FLOPs/B | 83.3 FLOPs/B |

## Configuration Sweep Results (4096x4096 / 8192x8192)

| Configuration | 4096 TFLOPS (% cuBLAS) | 8192 TFLOPS (% cuBLAS) | Notes |
|---|---|---|---|
| **128x128x64, 4w, 3s (baseline)** | **403 (95%)** | **412 (92%)** | **Optimal** |
| 128x128x32, 4w, 3s | 379 (89%) | 394 (88%) | More K-iters, no gain |
| 128x128x32, 4w, 5s | 384 (91%) | 393 (88%) | More stages, still worse |
| 128x128x32, 4w, 4s | 383 (91%) | 395 (88%) | Similar |
| 128x128x64, 4w, 2s | 396 (94%) | 404 (90%) | Less pipeline depth |
| 128x128x64, 8w, 3s | 390 (92%) | — | 9-warp register pressure |
| 256x128x32, 8w, 3s | 17 (4%) | — | CuTe DSL codegen failure |
| 256x128x64, 8w, 2s | 12 (3%) | — | CuTe DSL codegen failure |
| 128x128x32, 4w, 3s, 376cta | 381 (90%) | 397 (88%) | 2 blocks/SM attempt |

## Root Cause Analysis

### Why the reference is slower than cuBLAS/generated

**1. Tile shape: 128x128 vs 256x128 (primary factor)**

The 128x128 tile has 23% lower arithmetic intensity (62.5 vs 83.3 FLOPs/byte)
because the data footprint scales with `(M+N)*K` while compute scales with `M*N*K`.
A larger M (256 vs 128) amortizes data loading over more compute.

At 8192x8192: reference processes 4096 tiles vs cuBLAS's 1024. More tiles = more
epilogue and pipeline-prelude overhead per total compute.

**2. Occupancy: 10.4% vs 16.6% (secondary factor)**

Warp specialization dedicates 1/5 warps (20%) to DMA. cuBLAS and the generated
kernel use all warps for MMA, with thread 0 issuing TMA inline.

Lower occupancy means less latency hiding: 4.99 vs 7.99 active warps per SM.

**3. CuTe DSL 256x128 tile failure (blocking factor)**

The CuTe DSL generates catastrophically bad code for 256x128 tiles on SM120
(~100x slower). With 8 MMA + 1 DMA = 9 warps at 232 regs/thread:
`9 warps * 32 threads * 232 regs = 66,816 > 65,536 register file`. This exceeds
the SM register file, causing massive spilling despite `setmaxregister`.

This is confirmed by the 128x128 8-warp test also degrading (390 vs 403 TFLOPS).

### Why 2 blocks/SM with bK=32 doesn't help

To enable 2 blocks/SM, SMEM must be < 50 KB, requiring bK=32 (48 KB) instead of
bK=64 (99 KB). But bK=32 doubles K-loop iterations (256 vs 128 for 8192x8192),
adding pipeline synchronization overhead that outweighs the occupancy improvement
(20.8% vs 10.4%).

## Conclusions

1. **The baseline config (128x128x64, 4w, 3s) is optimal** for CuTe DSL on SM120.
   No parameter combination improves it.

2. **The 5-8% gap to cuBLAS is structural**: tile shape determines arithmetic
   intensity, and the CuTe DSL cannot efficiently generate 256x128 kernels.

3. **The generated kernel matches cuBLAS** at large sizes (1.00-1.01x) because it
   uses the same 256x128 tile, 8-warp, all-MMA architecture.

4. **Closing the gap requires either**:
   - Fixing CuTe DSL codegen for 256x128 tiles on SM120 (CUTLASS framework work)
   - Implementing non-warp-specialized pipeline in CuTe DSL (cooperative TMA)
   - Both are upstream CUTLASS improvements, not kernel-level tuning

## Commit

- **Branch**: `worktree-matmul`
