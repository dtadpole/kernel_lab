# FA4 TMA+wgmma Implementation Attempt — Findings

## Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA H100 SXM5 |
| Architecture | SM 9.0a (Hopper) |
| CUDA | 12.9 |

## Objective

Implement TMA-based wgmma flash attention following the FA4 CuTe DSL architecture to close the 3x performance gap (220→600+ TFLOPS).

## CuTe DSL Architecture Analysis

Studied the complete FA4 CuTe DSL source at `flash_attn.cute`. Key components:

| Feature | CuTe DSL Implementation |
|---------|------------------------|
| Thread model | 256 threads: 128 producer (56 regs) + 128 consumer (256 regs) |
| Data movement | TMA with 128B swizzle, mbarrier synchronization |
| Compute | wgmma.m64n64k16 RS variant, P from registers |
| Pipeline | 2-stage K/V with separate mbarrier pipelines |
| Overlap | Intra-warpgroup: QK GEMM N overlaps PV GEMM N-1 |
| Tile sizes | tile_m=128, tile_n=128, head_dim=128 |
| Config | mma_pv_is_rs=True, intra_wg_overlap=True |

## What Was Built

A complete TMA+wgmma kernel (~1100 lines):
- 256 threads with producer-consumer warp specialization
- `setmaxnreg.dec 56` / `setmaxnreg.inc 256` for register budget split
- `__noinline__` on producer and consumer for independent ptxas register allocation
- TMA via `cp.async.bulk.tensor.2d` with mbarrier `arrive_expect_tx` / `try_wait.parity`
- `cuTensorMapEncodeTiled` with `SWIZZLE_128B` and `boxDim=[64, BLOCK]` (split DIM=128 into 2 halves)
- wgmma with SWIZZLE_128B descriptor layout
- 2-stage double-buffered K/V pipeline
- Online softmax with `exp2.approx.ftz`

## Compile Results

**Excellent** — the architecture works:
```
consumer_warp_group: 0 spill stores, 0 spill loads
producer_warp_group: 0 spill stores, 0 spill loads
flash_attention_tma_wgmma: Used 255 registers, 0 spills, 1 barrier
```

## TMA Constraints Discovered

`cuTensorMapEncodeTiled` with `SWIZZLE_128B` requires `boxDim[0] * elementSize == 128 bytes`:
- `boxDim=[128,64] SWIZZLE_128B` → **CUDA_ERROR_INVALID_VALUE** (fails)
- `boxDim=[64,64] SWIZZLE_128B` → **CUDA_SUCCESS** (works)
- `boxDim=[128,64] SWIZZLE_NONE` → **CUDA_SUCCESS** (works but no swizzle)

For DIM=128 with bf16: must split each tile into 2 TMA loads of 64 elements each.

## wgmma PTX Constraints Discovered

RS variant trailing parameters: `p, scaleA, scaleB, tnspB`
- `scaleA` and `scaleB` must be 1 or -1 (NOT 0) — this caused a ptxas error
- `p=true` (setp.ne 1,0): accumulate mode (D += A@B)
- `p=false` (setp.ne 0,0): overwrite mode (D = A@B)
- Previous code incorrectly used scaleA=0 for overwrite — the correct approach is p=false with scaleA=1

## Runtime Issue

Kernel crashes with "Unknown Error" in consumer_warp_group at offset +0x1620 (deep in function). compute-sanitizer reports 4231 errors, all "Unknown Error" — likely an architectural constraint violation in mbarrier or wgmma usage.

Possible root causes:
1. mbarrier phase tracking mismatch between producer and consumer
2. wgmma descriptor mispointing (SMEM address or LBO/SBO values)
3. `setmaxnreg` interaction with mbarrier (register donation timing)
4. Missing `wgmma.fence` before first wgmma or `fence_view_async_shared` after SMEM writes

## Next Steps

The TMA infrastructure is partially proven (compiles clean, TMA encoding works). Need to debug incrementally:

1. **Minimal TMA test**: Just TMA load + mbarrier + read data back (no wgmma). Verify data arrives correctly in SMEM.
2. **Add wgmma on static data**: Pre-fill SMEM with known values, run wgmma, verify output. No TMA.
3. **Combine**: TMA load + wgmma on loaded data. One KV block only.
4. **Full loop**: Add softmax, KV loop, causal masking.

## Files

- `generated.cu.tma_attempt` — The full TMA+wgmma kernel (saved for reference)
- `generated.cu.baseline` — The mma.sync baseline (222 TFLOPS, working)
- `generated.cu` — Currently restored to baseline
