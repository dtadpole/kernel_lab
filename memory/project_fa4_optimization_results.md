---
name: FA4 optimization results — LPT + conditional rescale + hybrid softmax
description: Three optimizations applied to 3-WG FA4 kernel. V9 hybrid (causal tree + noncausal interleaved softmax) is current best.
type: project
---

## Applied Optimizations (from FA4 paper Section 3)

### 1. LPT Scheduling (Section 3.3) ✓
- Reverse q_block order for causal → heaviest blocks first
- +2-7% on causal configs

### 2. Conditional Rescale (Section 3.1.4) ✓ (causal only)
- Skip O_acc *= rescale when rescale == 1.0f
- Applied via `if (IS_CAUSAL && o_rescale[half] == 1.0f) continue;`

### 3. Hybrid Softmax (profile-driven) ✓
- Causal: tree reduction with rv[32] gather/scatter
- Noncausal: interleaved h0/h1 in-place on S_acc for ILP
- Selected via `if constexpr (IS_CAUSAL)`

### Tried but rejected:
- Poly exp2: -35% (FMA not bottleneck)
- Branchless rescale: -11% (breaks unrolling)
- Rescale-early: hurts causal
- Unified interleaved for causal: -7% (linear max worse for sparse tiles)

## Key insight from profile
cuDNN uses BLOCK_Q=64 (64x128x128 tiles) — shorter reduction chain, lower wait stall.
This is a fundamentally different architecture worth exploring next.
