# Matmul Cold L2 Fair Comparison — 2026-04-03

## Methodology

All three implementations measured under identical cold-L2 conditions:
- **Generated CUDA**: eval_harness with L2 flush (`cudaMemset` 96MB) + fresh `cudaMalloc` + `fill_random_bf16` per trial
- **cuDNN**: `torch.mm()` with L2 flush + `torch.randn` fresh inputs per trial
- **CuTe DSL**: NVIDIA `Sm120GemmKernel` (TMA store epilogue) with L2 flush per trial

Previous measurements used warm-L2 for cuDNN (no flush, same `torch.arange` data
across trials), inflating cuDNN numbers by up to 15% at large sizes.

## Hardware

- GPU: NVIDIA RTX PRO 6000 Blackwell (SM 12.0, 188 SMs)
- Peak BF16 Tensor Core: 503.7 TFLOPS
- DRAM Bandwidth: 1,536 GB/s
- L2 Cache: 96 MB

## Results

| Config | cuDNN (TFLOPS) | CuTe DSL (TFLOPS) | Generated CUDA (TFLOPS) | CuTe DSL vs cuDNN | Generated CUDA vs cuDNN |
|--------|---------------|-------------------|--------------------------|-------------------|------------------------|
| 256×256 | 2.7 | 2.3 | **4.7** | 0.85× | **1.74×** |
| 512×512 | 22.0 | 15.0 | **29.3** | 0.68× | **1.33×** |
| 1024×1024 | 119.0 | 84.2 | **135.6** | 0.71× | **1.14×** |
| 2048×2048 | 263.8 | 243.3 | **276.5** | 0.92× | **1.05×** |
| 4096×4096 | **402.5** | 395.3 | 400.2 | 0.98× | 0.99× |
| 8192×8192 | 445.9 | **447.4** | 446.1 | 1.00× | 1.00× |

## Key Findings

1. **At large sizes (4096+), all three implementations perform within 2% of each other**
   (~400-447 TFLOPS, 79-89% of peak).

2. **Generated CUDA dominates at small sizes** (256-2048) due to the small-kernel
   variant (128×64 tile, 128 threads) which has lower launch overhead than cuDNN's
   cuBLAS dispatch or CuTe DSL's JIT compilation.

3. **CuTe DSL lags at small sizes** (0.68-0.85× cuDNN) due to Python/MLIR JIT overhead
   that dominates when kernel execution time is < 100µs.

4. **Previous "cuBLAS 518 TFLOPS" was an artifact** of warm-L2 benchmarking in cudnn.py
   (no L2 flush, same `torch.arange` data reused across trials). Under cold-L2 conditions,
   cuDNN achieves 446 TFLOPS at 8192 — matching Generated CUDA and CuTe DSL.

## Architecture Comparison

| Property | cuDNN (cuBLAS) | CuTe DSL | Generated CUDA |
|----------|---------------|----------|----------------|
| Tile shape | 256×128×32 | 128×128×64 | 256×128×32 |
| Threads | 256 (8 warps) | 160 (5 warps) | 256 (8 warps) |
| Warp specialization | No | Yes (1 DMA + 4 MMA) | No |
| Registers/thread | 218 | 207 | 191 |
| Occupancy | 16.67% | 10.41% | 16.63% |
| Epilogue | Unknown | TMA S2G (stmatrix) | Direct R2G (st.global) |
| Pipeline stages | 3 | 2 | 3 |
