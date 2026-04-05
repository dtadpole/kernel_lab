# FA4 Intra-WG QK/PV Overlap + TMA S2G Store — 511–648 TFLOPS (+10%)

## Hardware

- **GPU:** NVIDIA H100 SXM5 96GB HBM3
- **Architecture:** SM 9.0a (Hopper)
- **Peak BF16:** 800 TFLOPS (650W TDP)
- **Host:** devvm8490 (h8_3), GPU 4, CUDA 13.0

## Objective

Close the 17–20% gap between our generated CUDA kernel and Flash Attention 4
CuTe DSL by implementing the same intra-warpgroup QK/PV overlap pattern.

## What Changed

**Before:** Sequential QK → wait → softmax → PV → wait. The softmax computation
(exp2, warp shuffles, accumulator rescaling) blocks the tensor cores from
starting the PV GEMM.

**After:** Overlap QK[n] and PV[n-1] in the same consumer warp group using
`wgmma.wait_group` depth control:

```
PROLOGUE: QK[0] → wait_group<0> → softmax(first) → pack P to bf16x2
MAINLOOP: issue QK[n] → issue PV[n-1] → wait_group<1> (QK done)
          → softmax (runs while PV tensor cores execute!)
          → wait_group<0> (PV done) → rescale O → pack P
EPILOGUE: PV[last] → wait_group<0> → finalize → store O
```

Key implementation details:
1. **Pre-packed P values** (`uint32_t P_packed[32]`): Softmax result is packed
   to bf16x2 BEFORE the overlap, stored in separate registers from `S_acc`.
   This prevents read-write conflicts when QK overwrites `S_acc` while PV
   reads from `P_packed` concurrently.

2. **Deferred O rescaling**: O is rescaled AFTER PV completes (not during
   softmax). Mathematically correct because PV[n-1] used P values computed
   with the same running max — rescaling the entire O (including the just-added
   PV contribution) adjusts all terms to the new max.

3. **Scheduler barrier placement**: sync BEFORE QK, arrive AFTER PV issue
   (not after QK issue). Prologue has NO scheduler barriers — both consumer
   WGs do QK[0] independently.

4. **Two WGMMA groups in flight**: QK committed first, PV committed second.
   `wait_group<1>` ensures QK (first committed) has completed while PV
   (second committed) continues executing.

## Performance Comparison

| Config | Before (ms) | After (ms) | Before TF | After TF | Speedup |
|--------|------------|-----------|-----------|----------|---------|
| causal-b8-s4096 | 1.210 | 1.092 | 454.4 | 503.3 | +10.8% |
| causal-b4-s8192 | 2.177 | 1.979 | 505.1 | 555.5 | +10.0% |
| causal-b2-s16384 | 4.152 | 3.761 | 529.6 | 584.7 | +10.4% |
| nc-b8-s4096 | 1.971 | 1.797 | 557.7 | 612.0 | +9.7% |
| nc-b4-s8192 | 3.769 | 3.416 | 583.4 | 643.7 | +10.4% |
| nc-b2-s16384 | 7.340 | 6.698 | 599.2 | 656.6 | +9.6% |

## vs CuTe DSL Reference (GPU 7, previous session)

| Config | CuTe DSL TF | Gen CUDA TF | Gen/CuTe |
|--------|------------|-------------|----------|
| causal-b8-s4096 | 548.5 | 503.3 | 0.92× |
| causal-b4-s8192 | 612.1 | 555.5 | 0.91× |
| causal-b2-s16384 | 583.9 | 584.7 | **1.00×** |
| nc-b8-s4096 | 660.1 | 612.0 | 0.93× |
| nc-b4-s8192 | 692.9 | 643.7 | 0.93× |
| nc-b2-s16384 | 621.9 | 656.6 | **1.06×** |

## Resource Usage

- Registers: 168/thread, 0 spills (unchanged)
- SMEM: 160KB (unchanged)
- Barriers: 16 (unchanged)

## Correctness

All 6 configs: bit-identical output vs baseline (max_abs_error = 0.000000).

## TMA S2G O Store (added on top of overlap)

Replaced scalar packed BF16x2 stores with register→SMEM + TMA S2G:
- Register → Q SMEM with SWIZZLE_128B address computation
- 2-phase named barrier (256 SMEM writers + 32 TMA warp)
- Warp 4 does TMA `cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group`

| Config | Overlap only | + TMA S2G | Delta |
|--------|-------------|-----------|-------|
| causal-b8-s4096 | 503.3 | **510.9** | +1.5% |
| causal-b4-s8192 | 555.5 | **559.4** | +0.7% |
| causal-b2-s16384 | 584.7 | **585.1** | +0.1% |
| nc-b8-s4096 | 612.0 | **615.3** | +0.5% |
| nc-b4-s8192 | 643.7 | **648.2** | +0.7% |
| nc-b2-s16384 | 656.6 | 637.4 | -2.9% |

TMA S2G helps small configs where O store is a larger fraction of total time.
Slight regression on nc-b2-s16384 due to the 2-phase barrier overhead.

## Remaining Gap Analysis

The 7–8% gap on smaller configs (b8-s4096, b4-s8192) is likely due to:
1. **GPU-to-GPU variation**: CuTe DSL reference was measured on GPU 7, our
   numbers on GPU 4 (same host).
2. **CuTe DSL's smem_copy_atom**: More efficient register → SMEM mapping
   compared to our manual swizzle address computation.
