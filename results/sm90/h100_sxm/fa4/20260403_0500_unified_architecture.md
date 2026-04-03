# FA4 Flash Attention Optimization — 2026-04-03 (Unified Architecture)

## Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA H100 SXM5 |
| Architecture | SM 9.0a (Hopper) |
| SMs | 132 |
| Peak BF16 Tensor Core | 989.5 TFLOPS (FP32 accum, dense) |
| DRAM Bandwidth | 3,352 GB/s (HBM3, 5120-bit) |
| L2 Cache | 50 MB |
| Driver | 550.90.07 |
| CUDA Toolkit | 12.8 |

## Benchmark Configuration

- **Host**: devvm8490, GPU index 4
- **Measurement**: CUDA event timing with L2 cache flush + fresh random inputs per trial, 10 warmup + 20 timed trials, median reported
- **Correctness**: Bit-exact match vs baseline across 67M output elements (both causal and non-causal)
- **All configs**: H=16, D=128, total_tokens=B×S=32768

## What Changed (vs `20260403_0400_optimized`)

### Architecture: Warp-specialized → Unified

Replaced the 5-warp architecture (1 DMA + 4 MMA) with a 4-warp unified architecture where all warps cooperatively load data and compute.

| Aspect | Before (warp-specialized) | After (unified) |
|--------|--------------------------|-----------------|
| Threads | 160 (5 warps) | 128 (4 warps) |
| DMA | Dedicated warp 0 | All warps cooperate |
| K/V buffering | Double-buffered (separate SMEM) | Single-buffered (shared SMEM slot) |
| Q fragments | Loaded from SMEM each KV iter | Cached in registers |
| Synchronization | 5 named barriers | 1 barrier (__syncthreads) |
| SMEM usage | 80 KB | 32 KB |
| Registers/thread | 168 | 224 |
| Spills | 0 | 0 |
| Occupancy | 2 blocks/SM | 2 blocks/SM |

### Key design decisions

1. **Q register caching**: Q fragments are loaded to registers once before the KV loop. This saves 8 LDSM per KV iteration (64 LDSM total for s4096). The larger register budget (256 vs 204 per thread) makes this possible.

2. **K/V SMEM reuse**: K and V tiles are loaded to the same SMEM slot, one at a time. K is loaded, QK is computed, then V overwrites K for PV computation. This halves SMEM usage (32KB vs 80KB).

3. **No DMA warp**: Eliminates 4 named barrier synchronizations per KV iteration. All 128 threads cooperatively load K/V via cp.async, which completes in ~80 cycles (128 threads × 16B each) — negligible vs compute time (~3000 cycles).

4. **Simpler synchronization**: Uses `__syncthreads()` instead of named barriers. Fewer barriers = less synchronization overhead.

## Performance Comparison

FLOPs formula:
- **Non-causal**: `4 × B × H × S² × D`
- **Causal**: `2 × B × H × S² × D` (triangular ≈ half)

| Config | Unified (ms) | Unified TF | Baseline (ms) | Baseline TF | cuDNN (ms) | cuDNN TF | Speedup |
|--------|-------------|------------|---------------|-------------|------------|----------|---------|
| causal b8-s4096 | 4.626 | 118.8 | 5.118 | 107.4 | 1.032 | 532.6 | 1.11x |
| causal b4-s8192 | 6.503 | 169.1 | 9.942 | 110.6 | 1.956 | 562.1 | 1.53x |
| causal b2-s16384 | 15.201 | 144.7 | 19.711 | 111.6 | 6.281 | 350.1 | 1.30x |
| noncausal b8-s4096 | 6.387 | 172.1 | 9.827 | 111.9 | 1.764 | 623.2 | 1.54x |
| noncausal b4-s8192 | 14.860 | 148.0 | 19.438 | 113.1 | 5.975 | 368.1 | 1.31x |
| noncausal b2-s16384 | 31.962 | 137.6 | 41.217 | 106.7 | 14.381 | 305.8 | 1.29x |

## Analysis

- **1.11x–1.54x speedup** across all configs (avg ~1.35x)
- **TFLOPS range**: 119–172 (up from 107–113)
- **Best improvement**: configs with longer sequences (b4-s8192, b8-s4096) benefit most
- **Correctness**: Bit-exact match vs baseline (max_abs_error = 0.0 across 67M elements)

### Why it works

The unified architecture eliminates two sources of overhead:
1. **Named barrier stalls**: Each named barrier (bar.sync/bar.arrive) has multi-cycle latency. The warp-specialized kernel uses 4 barriers per KV iteration × ~8 cycles each = ~32 wasted cycles per iteration.
2. **Q LDSM elimination**: Caching Q in registers saves 8 ldmatrix_x4 calls per KV iteration. Each LDSM takes ~4 cycles + address computation overhead.

The reduced SMEM (32KB vs 80KB) and thread count (128 vs 160) maintain the same 2 blocks/SM occupancy while giving each thread 33% more register budget.

### Remaining gap

Still 3–5x behind cuDNN (305–623 TFLOPS). The fundamental bottleneck remains: synchronous `mma.sync.m16n8k16` prevents overlap between tensor core compute and scalar softmax work. Closing this gap requires SM90-native WGMMA (async warp group MMA).
