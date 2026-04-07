---
name: FA4 optimization plan from AVO paper
description: SM90 FA4 kernel optimization plan based on AVO paper (arxiv 2603.24517) ideas — 3 variants to try individually then combine
type: project
---

## Source
AVO paper (arxiv 2603.24517v1) — NVIDIA's agentic kernel evolution, 1668 TFLOPS on B200

## Current Baseline (generated.cu with TMA S2G)
- causal b8 s4096: 537 TFLOPS (0.96× FA4 DSL)
- causal b2 s16384: 614 TFLOPS (1.03× FA4 DSL)
- Peak: 76.7% of H100 800 TFLOPS

## Optimization Variants (try individually, then combine)

### V1: Correction/MMA Pipeline Overlap
Move O rescale from after `wgmma_wait_group<0>()` (PV done) to after QK issue.
Rescale's scalar FMULs run on ALU pipeline while QK runs on tensor pipeline = free overlap.
**Expected: +1% non-causal, +0.4% causal** (from AVO paper v29→v30)

Current flow: `wait PV → rescale O → pack P → sync → QK[n+1] → PV[n]`
New flow:    `wait PV → pack P → sync → QK[n+1] → rescale O → PV[n]`

### V2: Merge Prologue+Mainloop + Branchless Rescale
Eliminate separate prologue code. Single loop with branchless rescale:
`rescale = (kv_id == 0) ? 1.0f : exp2(old_max - new_max)`
Reduces I-cache pressure, code size. Enables lighter fence.
**Expected: reduces instruction footprint, may help I-cache bound configs**

### V3: Reduce wgmma_fence Calls
Mainloop has 2 wgmma_fence: before QK and before PV.
Try: (a) remove PV fence (rely on commit_group separation)
     (b) use lighter fence variant if available
     (c) merge QK+PV into single fence+commit sequence
**Expected: saves ~8 cycles per mainloop iteration**

## Results (2026-04-05)

### V3: Remove fence #3 → WINNER (+1-3% consistent)
Removed redundant `wgmma_fence()` after `wgmma_wait_group<1>()`. CUTLASS pattern uses only 2 fences.
Deployed to generated.cu.

### V1: Correction/MMA overlap → REJECTED (regression)
Moving rescale to after QK issue put rescale ON the critical path (QK→rescale→PV).
In the current design, rescale is already OFF the critical path (after PV wait).
AVO paper's approach assumes separate correction warps (Blackwell 4-WG), not applicable to SM90 3-WG.

### V2: Merge prologue+mainloop → SKIPPED
V3 already gives the improvement. Code size reduction from merging is low priority.

## Final Numbers (full benchmark, cold-L2, median of 20)
| Config | cuDNN | FA4 DSL | Generated | vs cuDNN | vs FA4 |
|--------|-------|---------|-----------|----------|--------|
| causal b8 s4096 | 555 | 554 | 535 | 0.96× | 0.97× |
| causal b4 s8192 | 553 | 549 | 554 | 1.00× | 1.01× |
| causal b2 s16384 | 552 | 578 | **605** | **1.10×** | **1.05×** |
| noncausal b8 s4096 | 612 | 667 | 594 | 0.97× | 0.89× |
| noncausal b4 s8192 | 587 | 622 | **620** | **1.05×** | 1.00× |
| noncausal b2 s16384 | 614 | 644 | 599 | 0.98× | 0.93× |

## Key Constraint
SM90 has 3 WGs max (not 4 like Blackwell). Fine-grained warp specialization (MMA/Softmax/Correction/Load) requires SMEM staging — risky, deferred.
