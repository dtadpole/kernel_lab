# RTX PRO 6000 Blackwell Workstation Edition — Roofline Parameters

Source: `docs/nvidia-rtx-blackwell-pro-gpu-architecture.pdf` (Table 4, pages 45-46)

## Architecture
| Parameter | Value |
|---|---|
| GPU | GB202 (Blackwell) |
| SMs | 188 |
| CUDA Cores | 24,064 |
| Tensor Cores | 752 (5th Gen) |
| Boost Clock | 2617 MHz |
| TDP | 600 W |
| L2 Cache | 128 MB |

## Memory
| Parameter | Value |
|---|---|
| Memory | 96 GB GDDR7 (ECC) |
| Interface | 512-bit |
| Bandwidth | 1,792 GB/s |

## Peak Compute (dense / sparse)
| Precision | TFLOPS (dense) | TFLOPS (sparse) |
|---|---|---|
| FP4 Tensor (FP32 accum) | 2,015.2 | 4,030.4 |
| FP8 Tensor (FP32 accum) | 1,007.6 | 2,015.2 |
| FP8 Tensor (FP16 accum) | 1,007.6 | 2,015.2 |
| FP16 Tensor (FP16 accum) | 503.8 | 1,007.6 |
| **FP16 Tensor (FP32 accum)** | **503.8** | **1,007.6** |
| **BF16 Tensor (FP32 accum)** | **503.8** | **1,007.6** |
| TF32 Tensor | 251.9 | 503.8 |
| INT8 Tensor | 1,007.6 | 2,015.2 |
| FP32 (non-Tensor) | 126.0 | — |
| FP16 (non-Tensor) | 126.0 | — |
| BF16 (non-Tensor) | 126.0 | — |

## Roofline Ridge Points (dense)
| Precision | Ridge (FLOP/byte) |
|---|---|
| BF16 Tensor (FP32 accum) | 281 |
| FP16 Tensor (FP32 accum) | 281 |
| FP8 Tensor (FP32 accum) | 562 |
| TF32 Tensor | 141 |
| FP32 (non-Tensor) | 70 |
