---
name: FA4 4-WG — SM90 entry regs too low at 512 threads (128 vs needed 168)
description: CUTLASS/DeepGEMM prove setmaxnreg works on SM90 at 3-WG (entry=168). 4-WG (entry=128) fails because consumer can't compile at 128 regs. Not a ptxas bug — a fundamental thread count constraint.
type: project
---

## Key Finding: setmaxnreg WORKS on SM90 — verified from CUTLASS source

CUTLASS SM90 warp-specialized GEMM (`sm90_gemm_allreduce_tma_warpspecialized_pingpong.hpp`):
- `LoadRegisterRequirement = 40` (setmaxnreg.dec 40)
- `MmaRegisterRequirement = 232` (setmaxnreg.inc 232)
- 3 WGs = 384 threads, entry = 65536/384 = 168 regs
- Balance: (168-40)×128 = 16384 freed, (232-168)×256 = 16384 acquired ✓

DeepGEMM (TensorRT-LLM, also SM90):
- `kNumTMARegisters = 40`, `kNumMathRegisters = 232`
- Same 3-WG architecture

## Why 4-WG fails

4 WGs = 512 threads → entry = 65536/512 = **128 regs**
Consumer FA kernel needs ~168-170 regs → **can't compile at 128**
→ ptxas either spills (1444B) or ignores setmaxnreg

3 WGs = 384 threads → entry = 65536/384 = **168 regs**
Consumer FA kernel needs ~168 regs → **fits exactly**
→ ptxas compiles at 168, setmaxnreg gives consumer 232 (extra headroom)

## This is NOT a ptxas bug
ptxas correctly handles setmaxnreg when entry regs ≥ consumer code requirements.
The constraint is: consumer code must compile at entry_regs = 65536/total_threads.
At 512 threads, entry=128 is simply too few for FA's 168-reg consumer.

## Confirmed optimal: 3-WG (384 threads, 168 regs)
Matches CUTLASS/DeepGEMM architecture exactly. 548 TF causal, 610 TF noncausal.
