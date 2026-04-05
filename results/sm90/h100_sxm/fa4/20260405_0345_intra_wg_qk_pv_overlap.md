# FA4 Intra-WG QK/PV Overlap + TMA S2G + Tree Reductions — 511–657 TFLOPS (+10-12%)

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

## Tree Max/Sum Reductions (commit a1c275a)

Replaced sequential 31-deep FMNMX chain with 5-level tree reduction
using separate `tmax[16]`/`tsum[16]` temp arrays. Same instruction count
but breaks dependency chain: critical path ~124 → ~20 cycles.

## GPU 7 Benchmark (apples-to-apples with CuTe DSL reference)

| Config | Gen CUDA | CuTe DSL | Ratio |
|--------|----------|----------|-------|
| causal-b8-s4096 | 511 | 549 | 0.93× |
| causal-b4-s8192 | 562 | 612 | 0.92× |
| causal-b2-s16384 | 588 | 584 | **1.01×** |
| nc-b8-s4096 | 627 | 660 | 0.95× |
| nc-b4-s8192 | 657 | 693 | 0.95× |
| nc-b2-s16384 | 640 | 622 | **1.03×** |

## NCU Analysis (GPU 7, causal-b8-s4096)

- Tensor core (HMMA) utilization: 61.5%
- FMA pipe: 19.7%
- IPC: 0.50 inst/cycle
- SASS: 1,117 instructions per mainloop iteration (24 HGMMA, 1,093 scalar)

## Remaining Gap Analysis (SM90-fundamental)

The 5-8% gap on small/medium configs is **softmax-limited**:
- Softmax: ~1,093 scalar instructions per mainloop iteration
- PV WGMMA: ~512 cycles (16 m64n64k16 RS × 32 cycles each)
- Even with perfect overlap, softmax exceeds PV by ~580 cycles
- 35% of each mainloop iteration is tensor-core-idle waiting for softmax

SM90 lacks the following SM100+ features that would help:
- `redux.sync.max.f32` — hardware quad max reduction (saves 2 shuffles)
- `f32x2` packed arithmetic — process 2 softmax elements per instruction
- `fma.rn.ftz.f32x2` — fused multiply-add on pairs

CuTe DSL has the same algorithmic constraint but may benefit from:
- `stmatrix.sync.aligned.m8n8.x4` for O → SMEM (we use manual swizzled stores)
- LLVM backend instruction scheduling (vs nvcc)
