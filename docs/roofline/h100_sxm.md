# NVIDIA H100 SXM — Roofline Parameters

Source: nvidia.com/en-us/data-center/h100 (official product page)

## Architecture
| Parameter | Value |
|---|---|
| GPU | GH100 (Hopper) |
| SMs | 132 |
| CUDA Cores | 16,896 |
| Tensor Cores | 528 (4th Gen) |
| Boost Clock | 1980 MHz |
| TDP | 700 W (configurable) |
| L2 Cache | 50 MB |

## Memory
| Parameter | Value |
|---|---|
| Memory | 80 GB HBM3 |
| Interface | 5120-bit |
| Bandwidth | 3,350 GB/s |

## Peak Compute (dense / sparse)
| Precision | TFLOPS (dense) | TFLOPS (sparse) |
|---|---|---|
| FP8 Tensor | 1,979 | 3,958 |
| FP16 Tensor | 989.5 | 1,979 |
| **BF16 Tensor** | **989.5** | **1,979** |
| TF32 Tensor | 494.7 | 989.5 |
| INT8 Tensor | 1,979 | 3,958 |
| FP64 Tensor | 66.9 | — |
| FP32 (non-Tensor) | 66.9 | — |
| FP64 (non-Tensor) | 33.5 | — |

## Roofline Ridge Points (dense)
| Precision | Ridge (FLOP/byte) |
|---|---|
| BF16 Tensor | 295 |
| FP16 Tensor | 295 |
| FP8 Tensor | 591 |
| TF32 Tensor | 148 |
| FP32 (non-Tensor) | 20 |
