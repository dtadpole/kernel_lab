---
name: optimize
description: Autonomous CUDA kernel optimization loop — profile, analyze, brainstorm, implement, verify
user-invocable: true
disable-model-invocation: true
argument-hint: <kernel> [gen=name] [ref=name] [arch=smXX] [gpu=N] [seed=auto|latest|init|vNNN]
---

# Optimize Kernel

Autonomous loop: bench → profile → analyze → implement → verify → bench.

## Arguments

| Arg | Required | Default | Description |
|-----|----------|---------|-------------|
| `$0` (kernel) | **yes** | — | `fa4`, `matmul`, etc. |
| `gen` | no | `cuda` | Target impl. Maps to `gen-{name}` slug |
| `ref` | no | all `ref-*` | Reference baseline(s) |
| `arch` | no | auto | GPU arch (e.g. `sm90`) |
| `gpu` | no | from CLAUDE.md | GPU index |
| `seed` | no | `auto` | Seed: `auto`, `latest`, `init`, or `vNNN` |

## Artifacts (read on demand)

| Artifact | When |
|----------|------|
| `artifacts/project-layout.md` | Phase 0 — slugs, directories, seeds |
| `artifacts/profiling-guide.md` | Phase 1c — NCU commands, metrics |
| `artifacts/analysis-guide.md` | Phase 2 — docs, roofline, external search |
| `artifacts/results-format.md` | Phase 6 — results file template |

## Toolbox — Code Constraints

Write raw CUDA C/C++ code only. Inline PTX is allowed and encouraged for
hardware-specific instructions (wgmma, cp.async.bulk, setmaxnreg, mbarrier, TMA).
Verify PTX availability via `/ik:docs`.

**FORBIDDEN**: CUTLASS, cuDNN, cuBLAS, Thrust, CUB, or any high-level GPU library.
You must implement all kernel logic from scratch — WGMMA scheduling, TMA loads,
mbarrier synchronization, shared memory management, epilogue stores, etc.
The only allowed includes are: cuda_runtime.h, cuda_bf16.h, cuda_fp16.h, mma.h,
and standard C/C++ headers.

## Phase 0: Initialize

Ensure gen/ has code. Seed from gem if empty. No cross-run access.

## The Loop

### 1. Understand
- **1a.** Run `/ik:bench` — authoritative baseline. Fix any ✗ first.
- **1b.** Read gen + ref code, previous results, compile output.
- **1c.** NCU profile 1-2 configs with largest gap. See `artifacts/profiling-guide.md`.

### 2. Analyze
Classify bottleneck (compute/memory/latency). Consult `/ik:docs`, check
roofline, search external insights, study prior attempts.
See `artifacts/analysis-guide.md`.

### 3. Brainstorm
Each idea must be grounded (data-backed), specific (exact change),
measurable (predicts NCU effect), feasible. Pick highest impact first.

### 4. Plan (write before coding)
Write a short plan as text output BEFORE writing any code:
- What optimization? (e.g., "add TMA store epilogue")
- What changes? (e.g., "replace scalar stores with cp.async.bulk")
- Expected impact? (e.g., "eliminate lg_throttle stall, ~3% improvement")
This plan is visible to the Supervisor. Do NOT skip this step.

### 5. Implement
One change at a time. New code = new turn.
Write incrementally — skeleton first, then add optimizations.
Do NOT write a full 500-line kernel in one shot.

### 6. Verify
1. Compile via `/ik:exec`. Check regs, spills, SMEM.
2. Trial ALL configs — ANY correctness failure → fix ×3 or revert.
3. Profile same configs as 1c for comparison.

### 7. Record + Bench (on improvement)
Write results file. Run `/ik:bench`. Print and STOP.

### 8. Retry (no improvement)
Revert to baseline. Record learning. Loop to Phase 2.
Stop after 4 failed ideas — write findings summary.

## Key Principles

- **ik:bench is sole authority** — only bench results are official
- **Correctness first** — if ANY config shows ✗ in benchmark, STOP optimizing
  and fix correctness. Performance is meaningless without correctness.
  Do NOT request another bench until all configs pass ✓.
- **One change at a time** — isolate variables
- **Stop on improvement** — bench, print, stop
- **Keep trying on failure** — revert and try next idea immediately
- **Profile failed attempts** — extract learning before reverting
