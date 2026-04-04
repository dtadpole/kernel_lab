# SM90 H100 BF16 Matmul — Unified Eval Harness Benchmark

**Date:** 2026-04-04 15:30 PDT
**Host:** devvm8490 (h8_3)
**GPU:** NVIDIA H100 SXM5 (SM90, 132 SMs), GPU 4
**Driver:** 570.86.15 / CUDA 12.8
**Power Limit:** 650W (800 TFLOPS rated peak at this TDP)
**Commit:** 3b17c43

## Environment

| Component | Version |
|-----------|---------|
| NVIDIA Driver | 570.86.15 (kernel module built from source with LLVM) |
| CUDA Toolkit | 12.8 |
| cuBLAS | 12.8.3 (pip `nvidia-cublas-cu12==12.8.3.14`) |
| CUTLASS / CuTe DSL | 4.4.2 (pip `nvidia-cutlass-dsl==4.4.2`) |
| torch | 2.11.0+cu128 |
| cuda-bindings | 12.8.0 |
| Eval methodology | Unified harness: 5 warmup + 10 trials, L2 flush, CUDA event timing |

## Methodology

All three implementations timed through the **unified eval harness**:
- **Generated CUDA**: `eval_harness.cu` (C harness) — L2 flush + fresh `cudaMalloc` per trial
- **CuTe DSL**: `measure_reference()` (Python harness) — L2 flush + in-place `normal_()` per trial
- **cuBLAS**: `measure_reference()` (Python harness) — same as CuTe DSL

Note: Python harness has ~35-95 us dispatch overhead per call that dominates
small-size timing. C harness has ~10 us dispatch overhead. This means Generated
CUDA numbers at small sizes reflect lower dispatch overhead, not just faster
kernel execution. At 4096+ the kernel execution time dominates and dispatch
overhead is negligible.

## Results

```
┌────────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────┬──────────┐
│ H100 SXM (h8_3)    │  cuBLAS 12.8.3   │     CuTe DSL     │  Generated CUDA  │ CuTe DSL │ Gen CUDA │
│ GPU4, CUDA 12.8    │  TFLOPS   (ms)   │  CUTLASS 4.4.2   │  raw PTX WGMMA   │ vs cuBLAS│ vs cuBLAS│
│ eval harness 5w+10t│                  │  TFLOPS   (ms)   │  TFLOPS   (ms)   │          │          │
├────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ mat-256x256        │    0.9  (0.037)  │    0.4  (0.094)  │    3.2  (0.010)  │   0.39×  │   3.53×  │
├────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ mat-512x512        │    6.8  (0.039)  │    2.9  (0.094)  │   20.6  (0.013)  │   0.42×  │   3.04×  │
├────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ mat-1024x1024      │   58.5  (0.037)  │   22.4  (0.096)  │  116.1  (0.019)  │   0.38×  │   1.98×  │
├────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ mat-2048x2048      │  318.6  (0.054)  │  155.6  (0.110)  │  563.3  (0.031)  │   0.49×  │   1.77×  │
├────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ mat-4096x4096      │  633.1  (0.217)  │  424.0  (0.324)  │  718.0  (0.191)  │   0.67×  │   1.13×  │
├────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ mat-8192x8192      │  729.6  (1.506)  │  573.6  (1.918)  │  753.8  (1.459)  │   0.79×  │   1.03×  │
├────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ % of peak          │   91.2%          │   71.7%          │   94.2%          │          │  800.0TF │
└────────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────┴──────────┘
```

## Key Findings

1. **Generated CUDA 753.8 TFLOPS = 94.2% of 800 TF rated peak** at 8192×8192,
   beating cuBLAS 12.8.3 by 3%.

2. **CuTe DSL has ~95 us Python dispatch overhead** that dominates small sizes
   (256-1024 all show ~0.095ms regardless of matrix size). At 8192 the kernel
   execution dominates and CuTe DSL reaches 573.6 TFLOPS (71.7% peak).

3. **cuBLAS 12.8.3 reaches 729.6 TFLOPS = 91.2% peak** at 8192. Previously
   cuBLAS was loading fbcode's bundled 12.0.2 version; fixed by adding pip
   package path to LD_LIBRARY_PATH in the Makefile.

4. **800 TFLOPS peak** is the correct roofline ceiling for this machine (H100
   SXM with 650W TDP cap), not the NVIDIA spec-sheet 989.5 TFLOPS which
   assumes 700W TDP.

## Changes Made This Session

- Unified eval harness: `measure_reference()` now does L2 flush + fresh data per trial
- Removed self-execution timing code from `cudnn.py`
- Fixed `evaluate.py` JSON overflow: binary output instead of inline 67M-float JSON
- Fixed `compile.sh` arch: pass `--arch` from Makefile, default PTXAS_ARCH to match
- Fixed cuBLAS version: LD_LIBRARY_PATH includes pip package path before fbcode
- AGENTS.md: documented unified harness rule
