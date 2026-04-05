# CuTe DSL matmul optimization ceiling analysis

- **Date:** 2026-04-05
- **Host:** devvm8490 (h8_3)
- **GPU:** NVIDIA H100 SXM R&R SKU 96GB, 650W TDP
- **Peak BF16:** 800 TFLOPS
- **Kernel:** matmul BF16, gen-cutedsl (CuTe DSL WGMMA GEMM)
- **Baseline config:** tile 128x256, 2 WGs cooperative, cluster (1,1), 4 pipeline stages

## Summary

CuTe DSL's reference WGMMA GEMM is consistently 3.3–4.7% slower than cuBLAS
across all 8 GPUs on this machine. This gap is stable and intrinsic to the
generated kernel code quality — not tunable via tile shape, cluster, or
pipeline parameters.

## All-GPU benchmark (mat-8192x8192 BF16)

| GPU | cuBLAS (ms) | CuTe DSL (ms) | DSL/cuBLAS | gap |
|-----|------------|---------------|------------|-----|
| 0 | 1.426 | 1.485 | 1.041x | 4.1% |
| 1 | 1.452 | 1.505 | 1.036x | 3.6% |
| 2 | 1.432 | 1.487 | 1.039x | 3.9% |
| 3 | 1.475 | 1.523 | 1.033x | 3.3% |
| 4 | 1.489 | 1.547 | 1.039x | 3.9% |
| 5 | 1.483 | 1.552 | 1.047x | 4.7% |
| 6 | 1.434 | 1.481 | 1.033x | 3.3% |
| 7 | 1.466 | 1.521 | 1.037x | 3.7% |
| **avg** | **1.457** | **1.513** | **1.038x** | **3.8%** |

GPU individual variation: ~4% (GPU 0/2/6 fastest, GPU 4/5 slowest).
The fast/slow ordering is consistent across cuBLAS and CuTe DSL.

## Full config benchmark (GPU 4)

| Config | ref-cublas (ms) | gen-cutedsl (ms) | DSL/cuBLAS | TFLOPS (DSL) |
|--------|----------------|-----------------|------------|-------------|
| 256x256 | 0.033 | 0.070 | 2.12x | 0.5 |
| 512x512 | 0.033 | 0.069 | 2.09x | 3.9 |
| 1024x1024 | 0.033 | 0.072 | 2.18x | 29.7 |
| 2048x2048 | 0.047 | 0.081 | 1.72x | 213.5 |
| 4096x4096 | 0.210 | 0.248 | 1.18x | 554.3 |
| 8192x8192 | 1.49 | 1.55 | 1.04x | 710.3 |

Small sizes (256–1024): CuTe DSL has ~0.070ms floor vs cuBLAS ~0.033ms.
This is kernel launch overhead, not compute.

## NCU profile comparison (GPU 4, mat-8192x8192)

| Metric | gen-cutedsl | ref-cublas | Interpretation |
|--------|-------------|------------|----------------|
| SM Busy | 92.84% | 93.65% | cuBLAS slightly better |
| Tensor (FP) active | 96.19% | 97.10% | cuBLAS tensor pipe tighter |
| Compute Throughput | 92.84% | 93.65% | |
| Memory Throughput | 58.51% | 94.83% | cuBLAS overlaps mem+compute better |
| L1/TEX Throughput | 60.62% | 98.32% | |
| Achieved Occupancy | 12.44% | 14.78% | cuBLAS fits more warps |
| IPC Active | 0.53 | 0.65 | cuBLAS better instruction packing |
| Stall: barrier | 35,620 | 17,317 | **CuTe DSL 2.06x more barrier stalls** |
| Stall: math_throttle | 222 | 2,706 | cuBLAS saturates tensor pipe more |
| Stall: mio_throttle | 14,755 | 15,295 | similar |
| Stall: wait | 17,391 | 16,710 | similar |

## Root cause

The performance gap is driven by **barrier synchronization overhead**:

1. **Barrier stalls 2x higher**: CuTe DSL's producer-consumer pipeline
   (TMA load warp vs WGMMA compute warp groups) spends 2x more time
   waiting on barriers compared to cuBLAS.

2. **Math pipe underutilized**: cuBLAS shows 12x more `math_pipe_throttle`
   stalls, meaning its WGMMA instructions are packed back-to-back and
   actually saturating the tensor core pipeline. CuTe DSL has gaps between
   WGMMAs due to barrier waits.

3. **Lower occupancy**: cuBLAS achieves 14.78% vs 12.44% occupancy (9.46
   vs 7.96 active warps/SM), allowing better latency hiding.

4. **Memory throughput difference**: cuBLAS at 95% memory throughput vs
   CuTe DSL at 59% — not because CuTe DSL is memory-inefficient, but
   because cuBLAS overlaps TMA loads with compute more effectively, keeping
   both pipelines busy simultaneously.

## Optimization attempts (all failed)

| # | Change | Result | Why |
|---|--------|--------|-----|
| 1 | Cluster (2,1) TMA multicast | No change (88.8%) | Kernel is compute-bound, multicast doesn't help |
| 2 | Tile 128x128 + cluster (2,1) | Regression (79.8%) | 1 WG instead of 2, lost compute throughput |
| 3 | CTA swizzle group_size_m=4 | Regression (85.8%) | Smaller swizzle groups hurt L2 reuse at 8192 |
| 4 | mma_inst_tile_k=2 (K=32, 9 stages) | Regression (87.9%) | Lower arithmetic intensity per stage, more loop overhead |

## Conclusion

The 3.8% average gap between CuTe DSL and cuBLAS at large sizes is an
intrinsic limitation of the CuTe DSL compiler's SASS code generation
for pipeline synchronization. The barrier overhead in the producer-consumer
warp group pattern cannot be reduced through parameter tuning (tile shape,
cluster shape, pipeline depth, CTA swizzle). Closing this gap would require
changes to the CuTe DSL compiler itself.

At small sizes (256–1024), the 2x gap is dominated by kernel launch overhead
(~0.070ms floor for CuTe DSL vs ~0.033ms for cuBLAS), which is a fixed cost
of the CuTe DSL runtime.
