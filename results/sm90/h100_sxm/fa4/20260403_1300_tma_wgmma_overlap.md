# FA4 TMA+wgmma with Intra-Warpgroup Overlap — 1.55x Speedup

## Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA H100 SXM5 |
| Architecture | SM 9.0a (Hopper) |
| CUDA | 12.9 |
| SMs | 132 |
| Peak BF16 Tensor Core | 989.5 TFLOPS |

## Summary

Rewrote the FA4 kernel from SM80-compatible mma.sync to SM90-native TMA+wgmma
with producer-consumer warp specialization and intra-warpgroup QK/PV overlap.
Achieved **1.55x average speedup** (220 -> 340 TFLOPS).

## Performance Comparison

```
+--------------------+---------+----------+---------+---------+
| Config             | mma.sync| TMA+wgmma| Speedup | cuDNN   |
|                    | TFLOPS  | TFLOPS   |         | TFLOPS  |
+--------------------+---------+----------+---------+---------+
| nc-b8-s4096        |   222   |   335    |  1.51x  |  ~646   |
| nc-b4-s8192        |   218   |   348    |  1.60x  |  ~607   |
| nc-b2-s16384       |   226   |   357    |  1.58x  |  ~593   |
| c-b8-s4096         |   191   |   290    |  1.52x  |  ~579   |
| c-b4-s8192         |   199   |   306    |  1.54x  |  ~615   |
| c-b2-s16384        |   206   |   318    |  1.54x  |  ~615   |
+--------------------+---------+----------+---------+---------+
| Average            |   210   |   326    |  1.55x  |  ~609   |
+--------------------+---------+----------+---------+---------+
```

Still at ~54% of cuDNN (~609 TFLOPS). Gap reduced from 35% to 54%.

## Architecture

### Before: mma.sync baseline (5 warps, 160 threads)
- 1 DMA warp (cp.async loads) + 4 MMA warps (mma.sync.m16n8k16)
- Synchronous tensor core ops: warp stalls during each mma.sync
- Only 8% of instructions were HMMA (tensor core)
- 168 registers, 0 spills

### After: TMA+wgmma (2 warp groups, 256 threads)
- Producer WG (128 threads, 56 regs): TMA loads Q/K/V via `cp.async.bulk.tensor.2d`
- Consumer WG (128 threads, 256 regs): `wgmma.mma_async.m64n64k16` RS variant
- Async tensor core ops: QK and PV overlap via `wgmma.wait_group(1)`
- 255 registers, 0 spills

### Key Techniques Implemented

1. **TMA (Tensor Memory Accelerator)**
   - `cuTensorMapEncodeTiled()` with `SWIZZLE_128B` for correct wgmma SMEM layout
   - `boxDim=[64, BLOCK]` (DIM=128 split into 2 half-loads per tile)
   - mbarrier-based producer-consumer synchronization
   - 2-stage double-buffered K/V pipeline

2. **wgmma.mma_async RS variant**
   - Q loaded via ldmatrix (A operand in registers)
   - K/V accessed via SMEM descriptors (B operand, SWIZZLE_128B layout_type)
   - P from registers for PV GEMM (no SMEM stage for P)

3. **Intra-warpgroup QK/PV overlap**
   - QK GEMM for block N and PV GEMM for block N-1 issued as separate commit groups
   - `wgmma.wait_group(1)` waits for QK only; PV continues in tensor cores
   - Softmax computed during PV execution (scalar pipeline overlaps tensor cores)
   - O rescaling deferred (`saved_rescale[]`) to avoid reading O while PV writes it

4. **Producer-consumer warp specialization**
   - `setmaxnreg.dec 56` for producer (donates registers)
   - `setmaxnreg.inc 256` for consumer (receives donated registers)
   - Producer `__noinline__`, consumer `__forceinline__` (critical: wgmma pipeline
     must not cross function boundaries or ptxas serializes all wgmma ops)

## Key Bugs Found and Fixed

1. **SWIZZLE_128B constraint**: `boxDim[0] * elementSize` must equal 128 bytes.
   For bf16 DIM=128: must split into 2 TMA loads of 64 elements each.

2. **mbarrier arrive count**: Consumer `mbarrier_arrive` for "empty" barriers
   must be guarded by `if (tid_in_wg == 0)` since init count = 1.

3. **V descriptor k-step offset**: For `tnspB=0`, K is within each row (contiguous).
   K-step advances by 32 bytes (column offset), NOT by 2048 bytes (row offset).

4. **wgmma scaleA/scaleB**: Must be 1 or -1, never 0. Overwrite mode uses
   predicate `p=false`, not `scaleA=0`.

5. **wgmma pipeline serialization (C7510)**: Consumer function must be
   `__forceinline__`, not `__noinline__`. Otherwise ptxas serializes all wgmma ops.

6. **O rescaling during PV (C7514)**: Cannot read/write O_lo/O_hi between
   QK commit and PV wait. Must defer O rescaling to before PV starts, using
   `saved_rescale[]` pattern.

## Compile Stats

```
flash_attention_tma_wgmma: 0 spill stores, 0 spill loads, 255 registers, 1 barrier
producer_warp_group:       0 spill stores, 0 spill loads
No ptxas performance warnings (C7510/C7514 resolved)
```

## Remaining Gap Analysis

At 326 TFLOPS average vs cuDNN ~609 TFLOPS (54%), the remaining gap comes from:

1. **Single consumer warp group**: FA4 CuTe DSL uses tile_m=128 with 2 consumer
   warp groups (each processing m=64). We use tile_m=64 with 1 consumer WG.
   Doubling to 2 WGs would double MMA parallelism.

2. **No inter-WG overlap**: With 2 consumer WGs, they can alternate QK/PV phases
   to keep tensor cores continuously fed.

3. **tile_n=64 vs tile_n=128**: FA4 uses tile_n=128 which reduces KV loop iterations
   by 2x, amortizing softmax overhead.

## Files

- `generated.cu` — The TMA+wgmma+overlap kernel (committed)
- `generated.cu.baseline` — The mma.sync baseline (220 TFLOPS)
- `generated.cu.tma_working` — Sequential TMA+wgmma without overlap (215 TFLOPS)
- `generated.cu.tma_attempt` — First TMA attempt with runtime bugs (reference)
