# NVIDIA H100 SXM — Roofline Parameters

## NVIDIA Reference Spec (official product page)

Source: nvidia.com/en-us/data-center/h100

### Architecture
| Parameter | Value |
|---|---|
| GPU | GH100 (Hopper) |
| SMs | 132 |
| CUDA Cores | 16,896 |
| Tensor Cores | 528 (4th Gen) |
| Boost Clock | 1980 MHz |
| TDP | 700 W (configurable) |
| L2 Cache | 50 MB |

### Memory
| Parameter | Value |
|---|---|
| Memory | 80 GB HBM3 |
| Interface | 5120-bit |
| Bandwidth | 3,350 GB/s |

### Peak Compute (dense / sparse)
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

### Roofline Ridge Points (dense)
| Precision | Ridge (FLOP/byte) |
|---|---|
| BF16 Tensor | 295 |
| FP16 Tensor | 295 |
| FP8 Tensor | 591 |
| TF32 Tensor | 148 |
| FP32 (non-Tensor) | 20 |

---

## Meta Fleet Variants

Source: internal docs — "Why 800 TFLOPS for H100?", "AMD GPU Ads Ranking
Inference", "[Milestone] GTi High TDP"

H100 SXM is deployed in multiple configurations across Meta with different
TDP caps, memory types, and effective peak TFLOPS. The NVIDIA reference spec
(989.5 TFLOPS, 700W) applies to the GenAI / LLaMA fleet only.

### Variant Comparison

| Variant | Memory | HBM BW | TDP | BF16 Peak | Platform | Fleet |
|---------|--------|--------|-----|-----------|----------|-------|
| **Standard SXM5** | 80 GB HBM3 | 3,350 GB/s | 700W | 989.5 TF | Grand Teton | GenAI / LLaMA |
| **R&R SKU** | 96 GB HBM3 | 2,400 GB/s | 500W → 650W | ~800 TF | Grand Teton Inference (GTI) | Ads / GEM / Monetization |
| **Arches CG1** | 96 GB HBM3 | 2,447 GB/s | 700W | 989 TF | Arches | PM Stage serving |

### R&R SKU Details

The R&R SKU is the dominant H100 variant in Meta's Ads inference fleet. Key
characteristics:

- **TDP was originally 500W**, later increased to **650W** across all Ads
  dedicated reservations. The increase yielded ~10% QPS improvement.
- **BF16 peak ~800 TFLOPS** is the power-capped value at 500W TDP. At 650W
  the effective peak is higher but still below the 989.5 TF NVIDIA spec.
- Memory bandwidth of **2,400 GB/s** corresponds to **HBM3 with 6144-bit bus
  @ ~1562 MHz** (slightly downclocked from standard 1593 MHz). Note: some
  internal docs label this as "HBM2e" but 2,400 GB/s is not achievable with
  HBM2e (5120-bit bus, max ~2,039 GB/s at 1600 MHz). The bandwidth math
  confirms HBM3.
- Max hardware power limit is **700W**, meaning higher TDP is physically
  possible but capped by datacenter power policy.

### Why Different Teams Use Different Peak Numbers

| Team | Peak TFLOPS | Reason |
|------|-------------|--------|
| Ads / GEM / Monetization | 800 | R&R SKU, 500W TDP cap (hardcoded in MFU profiler) |
| GenAI / LLaMA | 989 | Standard SXM5, 700W TDP |
| Some Ads MFU docs | 1,979 | Mistakenly using 2:4 sparsity number |

This is not a convention choice — it reflects different physical GPU variants
and power configurations deployed in different clusters.

---

## devvm8490 (h8_3) Measured Specs

Measured 2026-04-04. This is a development VM, not a production fleet machine,
but shares the same H100 SXM hardware.

### Hardware Identification

| Parameter | Value | Matches |
|---|---|---|
| GPU | NVIDIA H100 SXM | — |
| Memory | 96 GB | CG1 / R&R (not standard 80GB) |
| Memory Clock | 1593 MHz | Standard (not downclocked) |
| Memory Bus | 6144-bit (HBM3) | Calculated from BW |
| Memory BW | 2,447 GB/s | **CG1 spec** |
| Current Power Limit | 650W | R&R TDP ceiling |
| Max Power Limit | 700W | Hardware supports full spec |
| SMs | 132 | Standard |
| Max Boost Clock | 1980 MHz | Standard |

### Sustained Clock Under Load (BF16 matmul 8192×8192)

| Metric | Value |
|---|---|
| SM Clock (sustained) | 1125–1425 MHz (average ~1300 MHz) |
| Power Draw | 641–655 W (**power-limited**) |
| Temperature | 50–55°C (not thermal-limited) |
| Memory Clock | 1593 MHz (constant) |

### Effective Peak at Current Power Limit

The GPU is **power-limited, not thermal-limited**. Under sustained BF16
matmul load, the 650W power limit forces SM clock down from 1980 MHz to
~1300 MHz.

```
Theoretical peak @ 1980 MHz:  989.5 TFLOPS (NVIDIA spec)
Clock-corrected @ 1300 MHz:   989.5 × (1300/1980) = 650 TFLOPS (sustained)
Clock-corrected @ 1425 MHz:   989.5 × (1425/1980) = 712 TFLOPS (burst)
```

Measured cuBLAS sustained performance: **~700 TFLOPS** (short bursts to ~750).
This is consistent with the clock-corrected estimate.

### What It Would Take to Reach 989.5 TFLOPS

Raise `Current Power Limit` from 650W to 700W (the hardware maximum):
```bash
sudo nvidia-smi -i 4 -pl 700  # requires admin
```
This would allow the GPU to sustain higher SM clocks under load, closing
the gap toward the NVIDIA reference peak. The ~10% QPS improvement observed
in Ads production when TDP was raised from 500W to 650W suggests a similar
gain from 650W to 700W is plausible.
