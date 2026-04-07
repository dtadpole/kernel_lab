---
name: FA4 4-WG final conclusion — SM90 physical limit is 384 threads
description: 65536/168=390 threads max. 384 (3 WGs) is optimal. 4-WG impossible without spills on SM90.
type: project
---

## Definitive Finding (2026-04-05)

SM90 register file = 65536. Consumer needs 168 regs.
65536 / 168 = 390.1 → max 12 warps = 384 threads = 3 WGs.

Adding ANY threads beyond 384 forces ptxas to reduce regs below 168 → spills → slower.

## What Was Tested
- 512 threads: 128 regs, 1548B spill → 157 TF
- 416 threads: 128 regs (ptxas can't use 157 for 168-reg code), 1548B spill
- 416 + LB=384 trick: 168 regs but 168×416=69888>65536 → corruption
- n64 QK streaming softmax: 24 wgmma/iter, 208B spill → 399 TF (best 4-WG)
- 384 threads: 168 regs, 0 spill, 16 wgmma/iter → 548 TF (optimal)

## Why Blackwell Can Do 4-WG
Blackwell compiler supports per-WG register allocation at compile time.
192 regs for MMA, 80 for correction, 48 for load = 65536 total.
SM90 nvcc allocates uniformly → all threads get 128 at 512, or 168 at 384.

## Best SM90 Results
3-WG V3 + IS_CAUSAL template: 548 TF causal, 610 TF noncausal
FA4 CuTe DSL reference: 547-663 TF
cuDNN: 555-639 TF
