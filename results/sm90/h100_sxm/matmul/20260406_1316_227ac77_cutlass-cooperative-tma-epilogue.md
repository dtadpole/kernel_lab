# SM90 Matmul — CUTLASS 3.x Cooperative + TMA Epilogue

**Commit**: 227ac77
**Date**: 2026-04-06 13:16
**Host**: devvm8490 (h8_3), GPU 4, NVIDIA H100 SXM R&R SKU 96GB
**CUDA**: 13.2, Driver 595.45.04
**CUTLASS**: 4.3.5
**Harness**: eval_harness (cold-L2, fresh pointers, 5 warmup + 10 trials)

## Architecture

- CUTLASS 3.x CollectiveBuilder API
- Schedule: KernelTmaWarpSpecializedCooperative
- Tile: 128×256×64 (M×N×K)
- Cluster: 2×1×1 (TMA multicast for A across 2 CTAs along M)
- Epilogue: TmaWarpSpecializedCooperative (TMA store)
- Pipeline: 4 stages, persistent scheduler
- WGMMA: m64n256k16 descriptor-based (128B swizzle)
- 168 registers, 0 spills, 2 barriers

## Results (bench v003)

| Config      | gen-cuda (ms) | gen TF | ref-cublas TF | gen/ref |
|-------------|--------------|--------|---------------|---------|
| 256×256     | 0.017        | 1.9    | 1.2           | 1.7×    |
| 512×512     | 0.020        | 13.5   | 9.6           | 1.4×    |
| 1024×1024   | 0.027        | 78.8   | 76.0          | 1.0×    |
| 2048×2048   | 0.040        | 434.0  | 388.3         | 1.1×    |
| 4096×4096   | 0.212        | 649.5  | 671.5         | 1.0×    |
| 8192×8192   | 1.610        | 680.9  | 738.2         | 0.9×    |

**Peak: 680.9 TFLOPS (85.1% of 800 TF, 92% of cuBLAS)**

## Optimization History

| Config | v001 (128×128) | v002 (128×256) | v003 (TMA epi) | Change |
|--------|---------------|----------------|----------------|--------|
| 8192   | 482.7 TF      | 609.8 TF       | **680.9 TF**   | +41%   |
| 4096   | 474.2 TF      | 545.5 TF       | **649.5 TF**   | +37%   |
| 2048   | 326.2 TF      | 349.5 TF       | **434.0 TF**   | +33%   |

## Failed Attempts

| # | Change | Result | Why |
|---|--------|--------|-----|
| 1 | 128×256×128 (larger K) | 377.8 TF | Only 2 stages, insufficient pipelining |
| 2 | Pingpong schedule | 43 TF | 4858B register spills, catastrophic |
| 3 | Non-cooperative schedule | massive spills | 255 regs, 4540B spills |
| 4 | StreamK scheduler | 634 TF | Overhead for perfectly square sizes |
| 5 | Cluster 1×2 | 615.8 TF | Slightly worse than 2×1 |
| 6 | No cluster (1×1) + TMA epi | 671.7 TF | -1.3% vs cluster 2×1 |

## Key Optimizations

1. **128×256 tile** (+26%): Larger N tile amortizes pipeline overhead, uses m64n256k16 WGMMA
2. **Cluster 2×1** (+1.6%): TMA multicast shares A tiles between 2 CTAs along M
3. **TMA epilogue** (+10%): TMA store for output replaces default per-thread stores

## Root Cause of Remaining Gap

**C7510: WGMMA serialization across function boundaries.** CUTLASS template
code generates WGMMA calls inside function bodies that ptxas treats as
separate functions (due to template instantiation). ptxas cannot pipeline
WGMMA instructions across these boundaries, serializing what should be
overlapped compute.

cuBLAS avoids this because it's compiled with NVIDIA-internal tools that
don't have this limitation. CuTe DSL also partially avoids it via its
JIT compilation path (reaching ~96% of cuBLAS in previous analysis).

## Analysis

The ~8% gap to cuBLAS at large sizes (8192) is consistent with the CuTe DSL
ceiling analysis (3.8% gap at 8192). Our CUTLASS kernel shows a larger gap
(8% vs 3.8%) because CUTLASS template instantiation creates more function
boundaries than CuTe DSL's JIT-generated code.

Closing this gap requires either:
1. Hand-written PTX that avoids all function boundaries (extremely complex)
2. CuTe DSL JIT compilation (already analyzed, reaches ~96%)
3. Future ptxas improvement to handle C7510 correctly
