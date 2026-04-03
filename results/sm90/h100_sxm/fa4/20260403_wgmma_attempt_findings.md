# FA4 wgmma Optimization Attempt — Findings

## Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA H100 SXM5 |
| Architecture | SM 9.0a (Hopper) |
| CUDA | 12.9 |

## Objective

Migrate the FA4 generated kernel from SM80-compatible `mma.sync.m16n8k16` to SM90-native `wgmma.mma_async.m64n64k16` to close the 3x performance gap vs FA4 CuTe DSL / cuDNN.

## Baseline Performance

| Config | Generated (mma.sync) | FA4 CuTe DSL | cuDNN |
|--------|---------------------|--------------|-------|
| causal-b8-s4096 | 200 TFLOPS | 549 TFLOPS | 528 TFLOPS |
| noncausal-b8-s4096 | 224 TFLOPS | 658 TFLOPS | 625 TFLOPS |

## What Was Built

A complete wgmma-based flash attention kernel:
- 128 threads (1 warp group of 4 warps), no separate DMA warp
- `wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16` RS variant
- cp.async for global→shared data movement
- P staged through SMEM for D→A register layout conversion (softmax output → wgmma input)
- Double-buffered K and V
- 88KB shared memory (Q 16KB + K 32KB + V 32KB + P 8KB)
- 203 registers, 0 spills

## Performance (with Incorrect Results)

| Config | wgmma (wrong output) | mma.sync baseline |
|--------|---------------------|-------------------|
| causal-b8-s4096 | 249 TFLOPS | 200 TFLOPS |
| causal-b4-s8192 | 287 TFLOPS | 218 TFLOPS |
| causal-b2-s16384 | 312 TFLOPS | 224 TFLOPS |
| noncausal-b8-s4096 | 284 TFLOPS | 224 TFLOPS |
| noncausal-b4-s8192 | 313 TFLOPS | 228 TFLOPS |
| noncausal-b2-s16384 | 295 TFLOPS | 229 TFLOPS |

**25-40% faster** even with wrong results and no pipelining. Async MMA overlaps with softmax compute.

## Root Cause of Correctness Failure

### The wgmma SMEM Layout Requirement

The wgmma B operand (from SMEM descriptor) requires data in a **canonical interleaved layout**, NOT simple row-major. The CUTLASS canonical layout for `Major::MN` with `SWIZZLE_128B` is:

```
Swizzle<3,4,3> ∘ smem_ptr ∘ ((T,8,m),(8,k)) : ((1,T,LBO),(8T,SBO))
```

For bf16 (T=8), element B[n, k] is at offset:
```
offset = (n%8) + ((n/8)%8)*8 + (n/64)*LBO + (k%8)*64 + (k/8)*SBO
```

This is an **interleaved** layout where 8 N-positions are interleaved with 8 K-positions within each 128-byte SMEM block. Our cp.async loads produce simple **row-major** data:
```
offset = n * row_stride + k
```

These layouts are fundamentally different. Verified empirically:
- **V=ones test (all elements identical):** PASSES — interleaving doesn't affect uniform values
- **V=arange test (linearly increasing cols):** FAILS with column offset — wgmma reads wrong columns

### Correct wgmma PTX Syntax (Verified)

The RS variant (A=registers, B=SMEM descriptor) requires only **4 trailing parameters** after the predicate:
```ptx
wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16
    {d0..d31}, {a0..a3}, desc_b,
    p, scaleA, scaleB, tnspB;    // RS: 4 params (NOT 5)
```

The SS variant uses 5: `p, scaleA, scaleB, tnspA, tnspB`.

Compilation requires `-gencode arch=compute_90a,code=sm_90a` (NOT just `-arch=sm_90`).

### GmmaDescriptor Bitfield Layout (from CUTLASS)

```
[0:14)   start_address >> 4
[16:30)  leading_byte_offset >> 4
[32:46)  stride_byte_offset >> 4
[49:52)  base_offset (3 bits)
[62:64)  layout_type (0=INTERLEAVE, 1=SWIZZLE_128B, 2=SWIZZLE_64B, 3=SWIZZLE_32B)
```

## Path Forward

To achieve correct wgmma results, the SMEM data must match the canonical layout. Options:

1. **TMA (Tensor Memory Accelerator):** Hardware-accelerated global→shared loads that produce the correct interleaved layout. Requires host-side tensor descriptor setup. This is what CUTLASS/FlashAttention-4 uses.

2. **Explicit SMEM rearrangement:** cp.async load row-major → rearrange to interleaved in SMEM. Adds ~2μs per KV iteration but is simpler than TMA.

3. **Both A and B from SMEM (SS variant):** Put both Q and K in canonical layout, use SMEM descriptors for both. Eliminates ldmatrix for A operand.

## Key Learnings

1. **wgmma descriptors are NOT flexible** — they describe a fixed canonical layout, not arbitrary SMEM arrangements
2. **Row-major swizzle ≠ wgmma interleaved layout** — our Swizzle<3,4,3> only permutes columns within rows, while wgmma expects inter-row interleaving
3. **TMA is essential for wgmma** — practical wgmma kernels use TMA to produce the correct layout, not cp.async
4. **Performance potential is real** — even with wrong results, wgmma shows 25-40% improvement over mma.sync due to async tensor core execution
5. **The RS variant syntax has 4 trailing params**, not 5 — a subtle but critical difference from the SS variant
