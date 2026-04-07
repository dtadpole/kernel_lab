---
name: FA4 4-WG spill analysis — 152 bytes blocks 400→500+ TF
description: 152 bytes ptxas spills cause 54M L1 requests (100% overhead). Eliminating would yield ~500 TF.
type: project
---

## Current State (2026-04-05)
- 4-WG correct on all configs, 309-398 TFLOPS
- 152 bytes spills (causal), 288 bytes (noncausal)
- 54.5M spill requests, 100% overhead, warp cycles 9.14 (vs 5.20 FA4 DSL)

## Why Spills Persist
All approaches tried have O_lo[32]+O_hi[32]+S_lo[32]+P_packed[32] = 128 during overlap.
No room for descriptors/loop vars → ptxas spills them.

## Approaches Tested
- A: Split O + n64 PV → 152 bytes spills (same as before, PV doubled)
- B: 2-pass DIM → 0 spills in dummy test but doubles mainloop
- O_acc SMEM staging → 0 spills but huge L1 traffic (99.92% hit rate)

## Theoretical Speedup from 0 Spills
If warp_cycles goes from 9.14 → 5.20 (FA4 level):
  2.58ms → 1.47ms → ~481 TF
With better scheduling from 4-WG active warps: ~500-550 TF

## Key Insight for Next Step
The 128-reg peak comes from the OVERLAP (PV+QK concurrent = O+S+P all live).
If we DON'T overlap PV and QK (sequential), peak = max(O+P, S+P) = 96.
But sequential loses tensor core overlap → ~30% slower.

The ideal: overlap PV and QK but with P_packed as the ONLY shared variable.
This requires PV to NOT use O_lo/O_hi as accumulators during QK.
Option: PV writes to TEMP registers, then add to O after QK finishes.
But m64n64k16 RS always accumulates (+= or =), can't write to temp without extra regs.

## Remaining Options
1. Accept 152 bytes spills → optimize other aspects (WG3 work, barrier latency)
2. Use approach B (2-pass DIM) → 0 spills but 2× mainloop
3. Non-overlap sequential (already tested: 305-395 TF, correct, 0 spills possible)
4. Investigate if ptxas can be tuned (register allocation hints, -maxrregcount, etc.)
