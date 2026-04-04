# Hardware Roofline Reference

Peak compute and memory bandwidth specs for roofline analysis.

## Quick Comparison — BF16 Tensor (dense, FP32 accumulate)

| GPU | BF16 TFLOPS | Mem BW (GB/s) | Ridge (FLOP/byte) |
|---|---|---|---|
| A100 80GB SXM | 312 | 2,039 | 153 |
| H100 SXM (NVIDIA spec) | 989.5 | 3,350 | 295 |
| H100 SXM (Meta 650W cap) | ~712 burst | 2,447 | ~291 |
| B200 (HGX) | 2,250 | 7,700 | 292 |
| RTX 5090 | 209.5 | 1,792 | 117 |
| **RTX PRO 6000 BW** | **503.8** | **1,792** | **281** |

## Per-GPU Spec Sheets

- [A100 80GB SXM](a100_sxm.md)
- [H100 SXM](h100_sxm.md)
- [B200 (HGX / NVL72)](b200.md)
- [RTX 5090](rtx_5090.md)
- [RTX PRO 6000 Blackwell WS](rtx_pro_6000.md)

## Source Datasheets (PDFs)

- `nvidia-a100-datasheet.pdf`
- `nvidia-b200-datasheet.pdf`
- `nvidia-rtx-blackwell-geforce-gpu-architecture.pdf` (RTX 5090)
- `../nvidia-rtx-blackwell-pro-gpu-architecture.pdf` (RTX PRO 6000)
- `../rtx-pro-6000-blackwell-workstation-edition-datasheet.pdf`
