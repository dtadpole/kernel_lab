# NVIDIA A100 80GB SXM — Roofline Parameters

Source: `docs/roofline/nvidia-a100-datasheet.pdf`

## Architecture
| Parameter | Value |
|---|---|
| GPU | GA100 (Ampere) |
| SMs | 108 |
| CUDA Cores | 6,912 |
| Tensor Cores | 432 (3rd Gen) |
| Boost Clock | 1410 MHz |
| TDP | 400 W |
| L2 Cache | 40 MB |

## Memory
| Parameter | Value |
|---|---|
| Memory | 80 GB HBM2e |
| Interface | 5120-bit |
| Bandwidth | 2,039 GB/s |

## Peak Compute (dense / sparse)
| Precision | TFLOPS (dense) | TFLOPS (sparse) |
|---|---|---|
| FP16 Tensor | 312 | 624 |
| **BF16 Tensor** | **312** | **624** |
| TF32 Tensor | 156 | 312 |
| INT8 Tensor | 624 | 1,248 |
| FP64 Tensor | 19.5 | — |
| FP32 (non-Tensor) | 19.5 | — |
| FP64 (non-Tensor) | 9.7 | — |

## Roofline Ridge Points (dense)
| Precision | Ridge (FLOP/byte) |
|---|---|
| BF16 Tensor | 153 |
| FP16 Tensor | 153 |
| TF32 Tensor | 77 |
| FP32 (non-Tensor) | 10 |
