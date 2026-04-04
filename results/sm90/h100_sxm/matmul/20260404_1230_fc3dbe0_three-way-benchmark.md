# SM90 H100 BF16 Matmul — 3-Way Benchmark

**Date:** 2026-04-04 12:30 PDT
**Host:** devvm8490 (h8_3)
**GPU:** NVIDIA H100 SXM5 (SM90, 132 SMs), GPU 4
**Driver:** 570.86.15 / CUDA 12.8
**Commit:** fc3dbe0

## Kernels Under Test

| Kernel | Description |
|--------|-------------|
| **cuBLAS** | `torch.mm()` → `cublasGemmEx` BF16 (vendor baseline) |
| **CuTe DSL** | `HopperWgmmaGemmKernel` 128×256 tile, 2 WG cooperative, TMA G2S + S2G |
| **Generated CUDA** | Hand-written raw PTX WGMMA m64n256k16, TMA, warp-specialized (1 producer + 2 consumers), persistent tile scheduling, 4-stage pipeline |

## Benchmark Methodology

- 10 warmup + 20 timed trials, median latency
- CUDA event timing per trial
- A re-randomized each trial; B held constant (transpose amortized)
- Generated kernel runs in subprocess per config (static shape cache requires fresh process)

## Results

```
┌────────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────┬──────────┐
│ H100 SXM (h8_3)    │      cuBLAS      │     CuTe DSL     │  Generated CUDA  │ CuTe DSL │ Gen CUDA │
│ GPU4, CUDA 12.8    │  TFLOPS   (ms)   │  TFLOPS   (ms)   │  TFLOPS   (ms)   │ vs cuBLAS│ vs cuBLAS│
├────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ mat-256x256        │    4.1  (0.008)  │    3.3  (0.010)  │    3.3  (0.010)  │   0.80×  │   0.80×  │
├────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ mat-512x512        │   29.0  (0.009)  │   21.3  (0.013)  │   21.3  (0.013)  │   0.73×  │   0.73×  │
├────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ mat-1024x1024      │  145.6  (0.015)  │  125.9  (0.017)  │  123.6  (0.017)  │   0.86×  │   0.85×  │
├────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ mat-2048x2048      │  541.2  (0.032)  │  605.3  (0.028)  │  602.5  (0.029)  │   1.12×  │   1.11×  │
├────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ mat-4096x4096      │  618.8  (0.222)  │  660.7  (0.208)  │  724.8  (0.190)  │   1.07×  │   1.17×  │
├────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ mat-8192x8192      │  744.5  (1.477)  │  734.6  (1.497)  │  751.6  (1.463)  │   0.99×  │   1.01×  │
├────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ % of peak          │   75.2%          │   74.2%          │   75.9%          │          │  989.5TF │
└────────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────┴──────────┘
```

## Analysis

- **Small sizes (256-1024):** cuBLAS dominates — hand-written kernels pay fixed overhead costs (TMA descriptor setup, warp specialization) that matter when the matrix is small. Generated and CuTe DSL are nearly identical at 73-86% of cuBLAS.

- **Medium sizes (2048-4096):** Both WGMMA-based kernels (CuTe DSL and Generated) overtake cuBLAS by 7-17%. Persistent tile scheduling with L2 super-tiling pays off at these sizes where there are enough tiles to keep all 132 SMs busy.

- **Large sizes (8192):** All three converge to ~735-752 TFLOPS (74-76% of H100's 989.5 TF peak). The generated kernel leads slightly at 751.6 TFLOPS (1.01× cuBLAS). At this size the workload is large enough that all implementations are memory/compute-saturated.

- **Peak utilization:** 75.9% (Generated at 8192×8192). The remaining ~24% gap to theoretical peak is expected — it includes epilogue store overhead, L2 flush between tiles, and the fact that BF16 WGMMA m64n256k16 doesn't achieve 100% tensor core utilization due to instruction scheduling constraints.

## Correctness

All 6 configs pass correctness (max abs error < 1.0 vs cuBLAS reference).
