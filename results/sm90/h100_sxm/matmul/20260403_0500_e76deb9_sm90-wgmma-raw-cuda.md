# SM90 WGMMA Raw CUDA Matmul — Benchmark Results

**Date:** 2026-04-03 05:00  
**Commit:** e76deb9  
**Kernel:** `data/generated/sm90/matmul/generated.cu` (V7)

## Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA H100 SXM |
| Architecture | SM 9.0 (Hopper) |
| SMs | 132 |
| Peak BF16 Tensor Core | ~990 TFLOPS |
| Driver | 550.90.07 (CUDA 12.4) + cuda-compat-12-8 |
| CUDA Toolkit | 12.8 |

## Performance (TFLOPS)

| Config | Raw CUDA (WGMMA) | cuBLAS | CuTe DSL | CUTLASS C++ | Raw/cuBLAS |
|--------|------------------|--------|----------|-------------|------------|
| 2048×2048 | 489 | 518 | 514 | 496 | 94% |
| 4096×4096 | 647 | 658 | 708 | 674 | 98% |
| **8192×8192** | **719** | **761** | **755** | **668** | **94%** |

## Key Achievements

1. **94% of cuBLAS** at 8192×8192 — hand-written raw CUDA + inline PTX
2. **95% of CuTe DSL** — nearly matching NVIDIA's MLIR-compiled kernel
3. **Beats CUTLASS C++ by 7.6%** (719 vs 668 TFLOPS)
4. **Bit-exact correctness** — zero error vs cuBLAS at all sizes
5. **Zero CUTLASS dependency** — pure `cuda_bf16.h` + `cuda.h` + `dlfcn.h`
6. **Zero ptxas warnings**, 154 registers, zero spill

## Architecture

- **Tile:** 128×256×64, 2 warpgroups (256 threads), 4-stage pipeline
- **WGMMA:** m64n256k16, SS mode (both operands from SMEM via descriptors)
- **TMA:** G2S with 128B swizzle, B pre-transposed to K-major
- **Pipeline:** Non-persistent, fence + 4× WGMMA + commit in single asm block
- **Scheduling:** Non-persistent (grid = totalTiles), CuTe DSL CTA swizzle (group_m=8)
- **Epilogue:** Direct bfloat162 global stores

## NCU Profile (8192×8192)

| Metric | Value |
|--------|-------|
| Duration | 1.87ms |
| SM Compute Throughput | 89.0% |
| Tensor Pipe (shared) | 92.2% (of active cycles) |
| Memory Throughput | 61.4% |
| Occupancy | 12.5% (1 block/SM) |
| Achieved Warps/SM | 7.94 |
| Dominant Stall | Barrier (37.5% of CPI) |

## Optimization History

| Version | 8192 TFLOPS | vs cuBLAS | Key Change |
|---------|------------|-----------|------------|
| V1 | 423 | 58% | Initial WGMMA implementation |
| — | 423→correct | — | K-stepping: start_address (not base_offset) |
| V3 | 559 | 75% | Pipeline overlap (wait_group 1) |
| V5 | 614 | 82% | Single asm block + m64n256k16 |
| V6 | 680 | 89% | Non-persistent + CuTe DSL pipeline |
| **V7** | **719** | **94%** | **CTA swizzle (group_m=8)** |

## Key Technical Discoveries

1. **WGMMA K-stepping must use start_address advancement**, not base_offset.
   CUTLASS confirms: `DescriptorIterator::operator+` adds to `reg32_[0]` only.
   `base_offset` stays 0 always — it's for swizzle alignment, not K-stepping.

2. **WGMMA m64n256k16 fragment layout** (empirically verified):
   ```
   row = warp_in_wg * 16 + ((i>>1)&1) * 8 + groupID
   col = (i>>2) * 8 + (i&1) + thread_in_group * 2
   ```

3. **Non-persistent scheduling eliminates ptxas C7515** (accumulator serialization).
   The epilogue's register reads across persistent loop iterations triggered the warning.

4. **CTA swizzle pattern matters**: group_m=8 gives +5.7% over 4×4 super-tiling.

5. **TILE_K must match swizzle line width**: TILE_K=64 (128B rows) → 128B swizzle.
   TILE_K=32 (64B) → 64B swizzle. TILE_K=128 (256B) → broken with 128B swizzle.

## Remaining Gap Analysis (6% to cuBLAS)

The kernel achieves 92.2% tensor pipe utilization. The remaining gap is from:
- **Instruction scheduling quality** — nvcc/ptxas vs NVIDIA's internal cuBLAS compiler
- **Pipeline management overhead** — mbarrier wait, descriptor construction, fence/commit
- **Epilogue overhead** — direct global stores (vs potential TMA S2G)
- **B transpose** — cached after first call but adds memory footprint
