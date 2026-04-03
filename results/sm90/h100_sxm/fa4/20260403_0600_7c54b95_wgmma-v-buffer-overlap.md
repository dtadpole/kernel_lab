# FA4 WGMMA Optimization: V-Buffer Overlap + tile_n=128 Attempt

## Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA H100 SXM5 |
| Architecture | SM 9.0a (Hopper) |
| SMs | 132 |
| Peak BF16 Tensor Core | 989.5 TFLOPS (FP32 accum, dense) |
| DRAM Bandwidth | 3,352 GB/s (HBM3) |
| Driver | 550.90.07 |
| CUDA Toolkit | 12.8 |
| Host | devvm8490, GPU index 4 |

## Benchmark Configuration

- Measurement: CUDA event timing, L2 flush + fresh random inputs per trial, 10 warmup + 20 timed trials, median reported
- All configs: H=16, D=128, total_tokens=B*S=32768

## Optimization A: tile_n=128 (BLOCK_KV 64->128) -- REVERTED

Changed QK GEMM from m64n64k16 to m64n128k16, doubling the KV tile processed per iteration.

### Compile stats

| Metric | Baseline | Tile128 |
|--------|----------|---------|
| Registers/thread | 168 | 242 |
| Spill stores | 0 | 0 |
| Spill loads | 0 | 0 |
| SMEM/block | 56 KB | 96 KB |
| Occupancy | 1 block/SM | 1 block/SM |

### Performance

| Config | Baseline (TFLOPS) | Tile128 (TFLOPS) | Speedup |
|--------|-------------------|------------------|---------|
| causal b8-s4096 | 357.1 | 351.1 | 0.98x |
| causal b4-s8192 | 381.8 | 379.6 | 0.99x |
| causal b2-s16384 | 393.3 | 393.9 | 1.00x |
| noncausal b8-s4096 | 420.0 | 418.2 | 1.00x |
| noncausal b4-s8192 | 435.3 | 433.9 | 1.00x |
| noncausal b2-s16384 | 440.8 | 424.1 | 0.96x |

### Why it failed

No improvement because halving KV iterations also doubles per-iteration cost (2x K load, 2x P store, 2x PV k-steps, 2x softmax values). The pipeline overhead that would have been amortized was already small. The 96KB SMEM and 242 registers may have worsened scheduling. **Reverted.**

## Optimization B: Dedicated V Buffer -- KEPT

Added a dedicated 16KB V SMEM buffer so V loads overlap with QK+softmax compute, instead of loading V serially after softmax completes.

### Pipeline change

Before (serial V):
```
K[i+1] load (async) | QK GEMM | softmax | wait K[i+1] | V[i] load (BLOCKING) | wait V | PV GEMM
```

After (overlapped V):
```
V[i] load (async) + K[i+1] load (async) | QK GEMM | softmax | wait V[i] | PV GEMM | wait K[i+1]
```

### Compile stats

| Metric | Baseline | V-Buffer |
|--------|----------|----------|
| Registers/thread | 168 | 168 |
| Spill stores | 0 | 0 |
| Spill loads | 0 | 0 |
| SMEM/block | 56 KB | 72 KB |
| Occupancy | 1 block/SM | 1 block/SM |

### Performance

| Config | Baseline (TFLOPS) | V-Buffer (TFLOPS) | Speedup | cuDNN | CuTe DSL | vs cuDNN | vs DSL |
|--------|-------------------|-------------------|---------|-------|----------|----------|--------|
| causal b8-s4096 | 357.1 | 373.6 | 1.05x | 497.1 | 571.9 | 0.75x | 0.65x |
| causal b4-s8192 | 381.8 | 399.3 | 1.05x | 544.2 | 659.8 | 0.73x | 0.61x |
| causal b2-s16384 | 393.3 | 408.7 | 1.04x | 548.0 | 710.3 | 0.75x | 0.58x |
| noncausal b8-s4096 | 420.0 | 436.2 | 1.04x | 601.6 | 711.6 | 0.72x | 0.61x |
| noncausal b4-s8192 | 435.3 | 449.1 | 1.03x | 583.8 | 762.9 | 0.77x | 0.59x |
| noncausal b2-s16384 | 440.8 | 429.6 | 0.97x | 564.2 | 739.1 | 0.76x | 0.58x |
| **Average** | | | **1.03x** | | | **0.75x** | **0.60x** |

### Analysis

- Consistent 3-5% improvement on 5/6 configs (the noncausal b2-s16384 config regressed 3%, likely due to the 16KB extra SMEM reducing L1 data cache availability)
- Range: 374-449 TFLOPS (up from 357-441)
- Same register count (168) as baseline, no spills
- Gap to CuTe DSL is still ~1.6x

### Remaining gap to CuTe DSL

The ~1.6x gap is now dominated by:
1. **No intra-WG overlap**: QK[n] and PV[n-1] cannot overlap because we use 1 warp group; CuTe DSL uses 2 WGs
2. **P SMEM store**: softmax results must be written to SMEM before PV GEMM (SS mode); CuTe DSL uses RS mode (P stays in registers)
3. **cp.async vs TMA**: cp.async requires 128 threads to issue loads; TMA uses 1 thread with hardware-accelerated tensor loads
4. **Tile size**: 64x64 vs 128x128 (CuTe DSL processes 4x more work per iteration, better amortizing overhead)
