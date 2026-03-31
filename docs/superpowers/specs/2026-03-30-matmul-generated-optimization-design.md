# Generated Matmul Kernel Optimization Design

**Date:** 2026-03-30
**Goal:** Make the generated CUDA matmul kernel faster than the CuTe DSL reference (0.666ms at 4096×4096)

## Current State

**Reference (CuTe DSL):** 0.666ms / 205 TFLOPS at 4096×4096
- TMA G2S bulk copy for both A and B (zero-copy, no transpose)
- Warp specialization: 1 DMA warp + 4 MMA warps
- Persistent tile scheduling (170 SMs)
- 2-stage pipeline with PipelineTmaAsync barriers
- BF16 in-kernel epilogue

**Generated (CUDA):** 0.764ms / 180 TFLOPS at 4096×4096
- `cp.async` for A (efficient, contiguous 16-byte copies)
- Scalar gather for B: 8× `__ldg` per thread per tile — **the main bottleneck**
- No warp specialization — all 256 threads do both load and compute
- No persistent scheduling — one CTA per output tile
- 3-stage pipeline, 128 registers, 49KB SMEM

**Diagnosis:** The B-loading gather path generates 8 individual 2-byte global loads per thread instead of one 16-byte bulk transfer. This underutilizes memory bandwidth and stalls the pipeline.

## Approach: cp.async for Both A and B

Replace the scatter-gather B loading with contiguous `cp.async` 16-byte copies by pre-transposing B in `kernel_run()` before launching the GEMM kernel. This is the minimal-change, highest-impact optimization.

### Changes

1. **Add a device-side transpose kernel** in `kernel_run()`:
   - Allocate a temporary buffer `B_t` of size N×K (same total elements)
   - Launch a tiled transpose kernel: B(K,N) row-major → B_t(N,K) row-major
   - Use 32×32 tiles with SMEM to coalesce both reads and writes
   - This is a one-time cost amortized across the GEMM

2. **Replace `load_b_tile_to_smem` with `cp.async`** identical to the A-loading path:
   - B_t is now contiguous along K, so `cp.async` works directly
   - Remove `gather_b_k8` and `load_b_tile_to_smem` functions
   - The B loading code mirrors the A loading code exactly

3. **Increase thread count to 256 (8 warps)** and use warp specialization:
   - 1 DMA warp handles all `cp.async` loads for both A and B
   - 7 MMA warps handle compute with 4×4 MMA tiling
   - Use `setmaxnreg` to give MMA warps more registers (232) and DMA warp fewer (40)

4. **Add persistent tile scheduling**:
   - Launch exactly 170 CTAs (matching RTX 5090 SM count)
   - Each CTA loops over output tiles via atomic counter
   - Reduces launch overhead, improves SM utilization at all matrix sizes

### Non-goals

- Cluster multicast (requires SM100 pipeline infrastructure)
- TMA hardware instructions (not available from raw CUDA C++)
- Changing the `kernel_run` contract

### Expected Performance

- Transpose cost: ~0.02ms at 4096×4096 (32MB, well under bandwidth ceiling)
- GEMM improvement: eliminate ~50% of memory stalls from B loading
- Target: <0.66ms total, beating the reference's 0.666ms

## Testing

- Correctness: compare against `A.float() @ B.float()` with `allclose(atol=1e-1, rtol=1e-2)`
- Performance: measure via `make evaluate KERNEL=matmul CONFIG=mat-4096x4096`
- Validate at multiple sizes: 256, 1024, 4096, 8192

## File Changes

- `conf/fixtures/matmul/generated.cu` — the only file modified
