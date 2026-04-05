# SM90 Matmul — Optimization Ceiling on CUDA 13.0

## Hardware
- NVIDIA H100 SXM (96GB HBM3, 650W TDP)
- GPU0, devvm8491 (h8_4)
- CUDA 13.0 (nvcc V13.0.88), Driver 580.65.06
- Peak BF16: 800 TFLOPS (650W SKU)

## Baseline
- 128×128 tile, 1 warpgroup (128 threads), 2-stage TMA pipeline
- **460 TFLOPS at 8192×8192 (57.5% of peak)**

## Attempts

### 1. Warp specialization (3 WG: 1 producer + 2 consumers)
- **Result**: ~451 TF, no improvement
- **Root cause**: CUDA 13.0 ptxas C7507 — `setmaxnreg` **ignored**
- Without register redistribution, all 384 threads get ~170 regs uniformly
- Consumers need 128 acc + 30 context = 158 regs (fits in 170)
- But producer wastes 170 regs doing nothing → no register savings
- The warp specialization overhead (producer-consumer protocol, barrier sync,
  per-tile re-init) cancels the benefit of load-compute overlap

### 2. 3-stage pipeline (128×128 tile)
- **Result**: 458 TF, no improvement
- Extra stage doesn't help because TMA and WGMMA execute sequentially
  in a single warpgroup — more buffers just waste SMEM (96KB vs 64KB)
- Would only help with warp specialization (producer fills ahead)

### 3. SMEM-buffered coalesced epilogue
- **Result**: 439 TF, **regression**
- Extra sync + SMEM write/read cycle costs more than it saves
- With 128×128 output and 128 threads, the per-element stores aren't
  the bottleneck — K-loop compute dominates at large sizes

## Blocking Issue

**CUDA 13.0's ptxas ignores `setmaxnreg` (C7507).** This blocks the primary
optimization path (warp specialization with register redistribution). Previous
CUDA 12.x versions honored `setmaxnreg`, enabling the same kernel to reach
775 TFLOPS.

## Options to Unblock

1. **Compile with CUDA 12.8/12.9** where `setmaxnreg` works — test if the
   compiled binary runs correctly on CUDA 13.0 driver
2. **Use CUTLASS 3.x C++ API** (CollectiveBuilder) which generates warp-
   specialized code through templates, avoiding inline `setmaxnreg` PTX
3. **Wait for CUDA 13.x patch** that fixes the C7507 regression
