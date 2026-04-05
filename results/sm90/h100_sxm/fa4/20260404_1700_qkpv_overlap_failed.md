# FA4 QK/PV Intra-WG Overlap — FAILED (Reverted)

## Objective

Overlap QK and PV WGMMA execution within a single warp group using the FA4
`mma_one_n_block_intrawg_overlap` pattern: issue QK[n+1] and PV[n] as
separate commit groups, use `wgmma.wait_group<1>()` to wait for QK while
PV runs, do softmax while PV tensor cores are busy.

## What Was Built

Restructured the KV loop into prologue/main/epilogue:
- **Prologue**: QK[0] → wait → softmax → pack P_packed[32]
- **Main loop**: QK[n+1] commit (group 1) → PV[n] commit (group 2) →
  wait<1> (QK done) → softmax (PV running) → wait<0> → pack P
- **Epilogue**: PV[last] → wait

Two implementations attempted:
1. `wgmma_wait_group<0>()` inside softmax before O rescale → **7% regression**
2. O rescale between wait<1> and wait<0> (FA4 pattern) → **18-20% regression**

## Root Cause: ptxas C7514

```
ptxas info: (C7514) Potential Performance Loss: wgmma.mma_async instructions
are serialized due to non wgmma instructions reading accumulator registers
of a wgmma between start and end of the pipeline stage
```

The O accumulator registers (`O_lo[32]`, `O_hi[32]`) are shared between:
- **PV WGMMA** (writes to O via scale_D=1 accumulation)
- **Softmax** (reads/writes O for online rescaling: `O *= exp2(old_max - new_max)`)

When softmax touches O registers between QK commit and PV commit (or between
PV commit and PV wait), ptxas detects the register conflict and forces the
WGMMA pipeline to serialize. The overlap becomes worse than sequential because
of added prologue/epilogue overhead and P_packed storage.

## Why FA4 Doesn't Have This Problem

FA4 uses **3 warp groups** (1 producer + 2 consumer):
1. Each consumer WG has its own O accumulator set
2. WG1 can do softmax+rescale on its O while WG2 does PV on its O
3. Consumer WGs use **240 registers/thread** (vs our 168-189)
4. Warp scheduler barriers coordinate WG execution order

The overlap is between **different warp groups**, not within a single WG's
pipeline. This avoids the C7514 serialization because each WG's accumulators
are independent.

## Performance (Attempt 2, FA4-style overlap)

| Config | Baseline (ms) | Overlap (ms) | Regression |
|--------|--------------|-------------|------------|
| causal-b8-s4096 | 1.350 | 1.596 | -18.3% |
| causal-b4-s8192 | 2.445 | 2.902 | -18.7% |
| causal-b2-s16384 | 4.663 | 5.534 | -18.7% |
| nc-b8-s4096 | 2.202 | 2.620 | -19.0% |
| nc-b4-s8192 | 4.171 | 5.021 | -20.4% |
| nc-b2-s16384 | 8.197 | 9.829 | -19.9% |

## Learning

**Intra-WG QK/PV overlap is fundamentally incompatible with 1-WG architecture**
when the O accumulator is shared between softmax and PV. The overlap requires
either:

1. **2+ consumer WGs** — each with independent O accumulators (FA4 approach)
2. **Separate O accumulator** — e.g., accumulate PV into a separate register
   set and merge after wait<0> (doubles register pressure, likely spills)
3. **rescale_O_before_gemm** — defer O rescale to before the PV GEMM using a
   stored scale factor, avoiding the register conflict within the pipeline stage

The next optimization should target **2 consumer WGs with correct inter-WG
barrier protocol** (FA4's NamedBarrierFwd pattern), not intra-WG overlap.
