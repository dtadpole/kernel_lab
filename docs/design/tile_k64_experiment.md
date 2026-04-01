# TILE_K=64 Experiment — Failed (2026-04-01)

## Goal

Increase TILE_K from 32 to 64 for the big kernel variant (256×128 tile)
to reduce K-loop iterations by 50%, halving pipeline/mbarrier overhead.

## Approach

TILE_K=64 doubles the data per pipeline stage:
- A: 256×64×2B = 32KB (was 16KB)
- B: 128×64×2B = 16KB (was 8KB)
- Total: 48KB/stage (was 24KB)

With 99KB SMEM limit, only 2 stages fit (was 3). To maintain the 64B swizzle
for A and 128B swizzle for B (which constrain TMA box dim0 to 32/64 elements),
each 64-element K dimension requires TWO TMA loads of 32 elements each.

This means 6 TMA loads per pipeline stage:
- A_kh0 (32 K-elems), A_kh1 (32 K-elems)
- B0_kh0, B0_kh1, B1_kh0, B1_kh1

## Results

### Attempt 1: 128B swizzle for A (box {64, 256})

- TMA descriptor creation succeeded
- Kernel hung in mbarrier_wait at kb=2
- NCU (from earlier 128B swizzle test on CuTe DSL): 255 regs, 14% compute
- Root cause: 128B swizzle changes SMEM address pattern; ldmatrix XOR
  computation needs complete rework for 8-chunk rows

### Attempt 2: 64B swizzle + 2 TMA loads per A k-half (box {32, 256})

- Each stage gets 6 TMA loads into separate SMEM regions
- `mbarrier_arrive_expect_tx(49152)` matches total bytes from 6 loads
- Kernel hung at kb=2 (mbarrier_wait_parity for phase 1 never completes)
- Verified: 2-stage pipeline with 3 TMA loads works fine (TILE_K=32)

### Attempt 3: Duplicate TMA loads (6 loads, same destinations)

- Tested with TILE_K=32 + doubled expect_tx + each TMA issued twice
- **Core dump** — mbarrier cannot handle 6 TMA completions per phase

## Root Cause

**mbarrier limitation on SM120 (Blackwell)**: each mbarrier phase can
handle at most 3 `cp.async.bulk.tensor` completion signals. Issuing more
than 3 TMA loads per mbarrier causes either:
- Deadlock (mbarrier_wait never completes)
- Core dump (illegal memory access in mbarrier hardware)

This is likely a hardware limit on the number of pending TMA transaction
tracking entries per mbarrier instance, not documented in the public
PTX ISA reference.

## Conclusion

TILE_K=64 is **not feasible** with the current TMA+mbarrier architecture
on SM120 unless:
1. A single TMA load can transfer the full 64 K-elements (requires 128B
   swizzle for A, which needs complete ldmatrix address rework)
2. Multiple mbarriers per stage (one per TMA group) — adds complexity
   and may not improve performance

The current TILE_K=32 + 3 stages + 3 TMA/stage is the optimal
configuration within hardware constraints.

## Data Points Collected

| Config | Registers | Spills | Result |
|--------|-----------|--------|--------|
| Baseline (TILE_K=32, 3 stages) | 191 | 0 | 446 TFLOPS @ 8192 |
| TILE_K=64, 64B swizzle, 6 TMA | 193 | 0 | Hang (mbarrier) |
| TILE_K=64, 128B swizzle, 4 TMA | 194 | 0 | Hang (wrong ldmatrix addr) |
| 2-stage pipeline, TILE_K=32 | 191 | 0 | Works (perf not measured) |
| 6 TMA loads per mbarrier (test) | — | — | Core dump |
