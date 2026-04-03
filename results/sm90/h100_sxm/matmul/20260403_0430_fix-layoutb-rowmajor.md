# SM90 Matmul — Fix LayoutB ColumnMajor→RowMajor

## Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA H100 SXM |
| Architecture | SM 9.0 (Hopper) |
| SMs | 132 |
| Peak BF16 Tensor Core | ~990 TFLOPS (dense) |
| Host | devvm8491.cco0.facebook.com |
| GPU Index | CUDA_VISIBLE_DEVICES=4 |
| CUDA Toolkit | 12.9 (nvcc) |
| CUTLASS Headers | 4.x (cutlass-4-headers fbpkg) |

## Change

**LayoutB: ColumnMajor → RowMajor** in `generated.cu`

### Root Cause

The kernel declared `LayoutB = cutlass::layout::ColumnMajor`, which expects B
in N×K storage order (column-major K×N). But the benchmark harness passes B
as K×N row-major (standard PyTorch convention), causing the kernel to compute
`C = A * B^T` instead of `C = A * B`.

### Diagnosis

With identity matrix test: `I * B` produced `B^T` instead of `B`.
With random matrices: `vs A*B: max_err=96.5`, `vs A*B.t(): max_err=0.0`.
Confirmed by passing `B.t().contiguous()`: correct result.

### Fix

Changed line 54: `LayoutB = cutlass::layout::ColumnMajor` → `LayoutB = cutlass::layout::RowMajor`

This makes the TMA load path handle the row-major→column-major transform
when loading B from global memory to shared memory (required by WGMMA).

## Correctness

| Config | Before | After |
|--------|--------|-------|
| 256×256 | FAIL (max_err=220.0) | PASS (max_err=0.0) |
| 512×512 | FAIL | PASS (max_err=0.0) |
| 1024×1024 | FAIL | PASS (max_err=0.0) |
| 2048×2048 | FAIL | PASS (max_err=0.0) |
| 4096×4096 | FAIL | PASS (max_err=0.0) |
| 8192×8192 | FAIL | PASS (max_err=0.0) |

## Performance (TFLOPS, median of 20 trials)

| Config | cuBLAS | CuTe DSL | Gen (before) | Gen (after) | Gen/cuBLAS | Gen/DSL |
|--------|--------|----------|-------------|------------|------------|---------|
| 256×256 | 1.8 | 2.0 | 1.7 | 1.7 | 94% | 85% |
| 512×512 | 14.1 | 14.3 | 12.7 | 12.5 | 88% | 87% |
| 1024×1024 | 72.2 | 93.2 | 87.6 | 88.4 | 123% | 95% |
| 2048×2048 | 371.3 | 485.9 | 416.2 | 419.8 | 113% | 86% |
| 4096×4096 | 618.9 | 662.0 | 657.8 | **665.4** | **108%** | **100%** |
| 8192×8192 | 779.3 | 754.5 | 620.5 | 636.9 | 82% | 84% |

## Key Findings

1. **Correctness fixed**: all configs now produce exact match vs cuBLAS
2. **4096 performance**: Generated now beats both cuBLAS (+8%) and CuTe DSL (+0.5%)
3. **RowMajor B didn't hurt performance**: TMA handles the layout transform efficiently
4. **8192 gap remains**: 636.9 vs 754.5 TFLOPS (84% of CuTe DSL) — still limited by
   ptxas C7510 wgmma pipeline serialization warning

## Remaining Issues

- ptxas C7510: `wgmma.mma_async instructions are serialized due to wgmma pipeline
  crossing function boundary` — affects both Small and Big kernels
- This is a fundamental nvcc/ptxas limitation vs CuTe DSL's MLIR JIT compiler
- Possible mitigations: cluster shape, device LTO, different CUTLASS schedule
