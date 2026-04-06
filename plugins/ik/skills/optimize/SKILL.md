---
name: optimize
description: Autonomous CUDA kernel optimization loop — profile, analyze, brainstorm, implement, verify
user-invocable: true
disable-model-invocation: true
argument-hint: <kernel> [gen=name] [ref=name] [arch=smXX] [gpu=N] [seed=auto|latest|clear|init|vNNN]
---

# Optimize Kernel

Autonomous optimization loop for CUDA kernels. Profiles current performance,
identifies gaps, brainstorms data-driven ideas, implements the most promising
one, verifies improvement, and commits if successful.

## Arguments

| Arg | Required | Default | Description |
|-----|----------|---------|-------------|
| `$0` (kernel) | **yes** | — | Kernel name: `fa4`, `matmul`, etc. |
| `gen` | no | `cuda` | Optimization target (exactly one). Maps to `gen-{name}` slug |
| `ref` | no | all `ref-*` | Reference baseline(s). Default: all in `data/ref/{kernel}/` |
| `arch` | no | auto | GPU arch (e.g. `sm90`). Auto-detected if omitted |
| `gpu` | no | from CLAUDE.md | GPU index (`CUDA_VISIBLE_DEVICES`) |
| `seed` | no | `auto` | Seed strategy: `auto`, `latest`, `init`, or `vNNN` |

GPU is **session-sticky**: once set, all subsequent ik invocations reuse it.

## Artifacts (read on demand)

Detailed reference material lives in `plugins/ik/skills/optimize/artifacts/`.
Read the relevant artifact when you reach the phase that needs it:

| Artifact | When to read |
|----------|-------------|
| `project-layout.md` | Phase 0 — slug mapping, directory layout, seed strategies |
| `profiling-guide.md` | Phase 1c / 5.4 — NCU commands, metrics to collect, bottleneck classification |
| `analysis-guide.md` | Phase 2 — docs consultation, roofline, external search, studying prior impls |
| `results-format.md` | Phase 6 — results file template and required sections |

## Optimization Toolbox

CUDA optimizations can use **any combination** of:
- **Pure CUDA C++** — standard CUDA API, shared memory, warp shuffles
- **Inline PTX assembly** — `asm volatile("...")` for instructions not exposed
  by CUDA intrinsics (e.g., `wgmma.mma_async`, `cp.async.bulk`, `fence.proxy`,
  `setmaxnreg`). Use when CUDA C++ cannot express the needed instruction.
- **CUTLASS / CuTe headers** — for high-level collective operations

Prefer CUDA C++ when possible. Use inline PTX only when the optimization
requires instructions not available through CUDA intrinsics. Always verify
PTX instruction availability for the target arch via `/ik:docs`.

## Phase 0: Initialize Gen Code

Ensure gen/ has code to optimize:
```python
from cuda_exec.impls import _ensure_gen_dir, list_gems, reseed_gen

gen_path = _ensure_gen_dir(kernel, arch)
# If no code: check for gems in this run → seed from chosen gem
# If no gems: write kernel from scratch using refs and docs as reference
```

**No cross-run access.** Only gems within the current run are valid seeds.
**No external seeding.** Never copy from `.worktrees/`, `legacy/`, or other runs.

## The Loop

Run **autonomously** until improvement or ideas exhausted.

### Phase 1: Understand (Bench → Code → Selective Profile)

#### 1a. Full Benchmark

Run `/ik:bench` first to establish authoritative baseline numbers:
```bash
.venv/bin/python -m cuda_exec.formal bench.kernel={kernel} bench.arch={arch} bench.gpu={gpu}
```
This gives ground-truth performance across ALL configs with correctness checks.
If any config shows ✗, fix before optimizing.

#### 1b. Read Code and Study History

With bench numbers in hand, **read and understand** the kernel source code:
- Read the gen implementation source (resolved via `resolve_impl`)
- Read ref implementations for comparison (`data/ref/{kernel}/`)
- Read ALL previous results files in `results/{arch}/{gpu_name}/{kernel}/`
- Review compile output: SASS, PTX, register usage, spill bytes, shared memory

Identify **where the gap is** from code + bench data before profiling.

#### 1c. Selective NCU Profiling

**Only after** identifying targets from bench results and code analysis, profile
1-2 specific configs to get hardware-level data. Do NOT profile everything.

Pick configs based on bench results:
- The config with the **largest gap** vs best `ref-*` (biggest opportunity)
- One **representative** config if needed

Read `plugins/ik/skills/optimize/artifacts/profiling-guide.md` for NCU commands
and metrics to collect.

### Phase 2: Analyze the Gap

With bench data + code understanding + selective NCU profiles:

1. **Classify bottleneck**: compute-bound, memory-bound, or latency-bound
2. **Consult NVIDIA docs** (source of truth) — `/ik:docs` + web search
3. **Check roofline** — `docs/roofline/` specs, calculate achieved % of peak
4. **Search for external insights** — GTC talks, papers, CUTLASS GitHub
5. **Study previous attempts** — results files document dead ends

Read `plugins/ik/skills/optimize/artifacts/analysis-guide.md` for detailed
guidance on each analysis step.

### Phase 3: Brainstorm Ideas

**Invoke `/10x-engineer:brainstorming`** to systematically explore the
optimization space. Feed it the bench data, NCU profiles, code analysis,
and prior results as context. The brainstorming skill ensures divergent
thinking before convergent selection.

Each idea must be:
- **Grounded** — backed by profiling data, docs, or prior results
- **Specific** — exact code change, not vague direction
- **Measurable** — predicts which NCU metric improves
- **Feasible** — implementable in current code

Rank by expected impact. Pick the single most promising idea first.

### Phase 4: Plan and Implement

**Invoke `/10x-engineer:writing-plans`** to create a detailed implementation
plan for the chosen idea. The plan should specify which lines change, expected
NCU effect, and risks.

Then **invoke `/10x-engineer:subagent-driven-development`** to execute the plan
if it contains independent sub-tasks that can be parallelized.

1. **Snapshot baseline** — gen/ scratch is the working copy; gem is the baseline
2. **Write plan** via brainstorming + writing-plans skills
3. **Implement** — modify the target impl source file
4. **Increment turn** — new code = new turn

### Phase 5: Verify

1. **Compile** via `/ik:exec` (new turn). Check registers, spills, shared mem.
   Up to 3 compile retries on failure.
2. **Trial ALL configs** — correctness is a **hard gate**:
   - Every config must pass. ANY failure → REJECT.
   - Up to 3 fix attempts. Still fails → **revert immediately**, next idea.
   - **Never weaken tolerances** to make a failing kernel pass.
3. **Output performance table** (mandatory after trial-all)
4. **Selective profile** — same 1-2 configs as Phase 1c, for comparison
5. **Compare** new vs baseline latency (preliminary — bench is authoritative)

```
  Trial ALL configs
        │
   Correctness?
    ┌───┴───┐
  FAIL    PASS
    │       │
  Fix×3   ┌──────┬──────┐
    │     Better  Same  Worse
  Still?    │      │      │
    │     Ph.6  Analyze  Revert
  REJECT  (bench) & retry & retry
```

### Phase 6: Record + Bench (only on improvement)

1. **Write results** file — see `plugins/ik/skills/optimize/artifacts/results-format.md`
2. **Run `/ik:bench`** — the sole authority. If bench shows any ✗, go back.
3. **Print results and STOP.** User decides whether to continue.

### Phase 7: Retry (no improvement or regression)

1. **Revert** to baseline (re-seed from gem)
2. **Record learning** — what was tried, why it failed, NCU evidence
3. **Loop back** to Phase 2 with new evidence
4. **Stop conditions**: after 4 total failed ideas, write findings summary and stop

## Optimization Targets

The optimization loop aims for TWO targets (both must be met for a kernel
to be considered "done"):

1. **+10% above native baseline** — the gen kernel's own first gem (v001)
   must improve by at least 10% TFLOPS at the largest config by the end
   of the session.
2. **+10% above reference** — the gen kernel must BEAT the best ref-*
   implementation (e.g., cuBLAS) by at least 10% TFLOPS at the largest
   config.

If target 2 is unreachable (e.g., cuBLAS is already at 92% of peak), aim
to close within 5% of the reference first, then iterate. Document why the
gap exists and what would be needed to close it.

## Key Principles

- **Stop on improvement** — bench, print, stop. Don't keep going.
- **Keep trying on failure** — revert and try next idea immediately. Don't ask.
- **Autonomous** — make decisions, implement, evaluate. Surface results only.
- **Docs are source of truth** — consult before and after profiling
- **One change at a time** — isolate variables
- **ik:bench is sole authority** — only bench results are official
- **Correctness first** — never sacrifice for performance
- **Never commit a regression**
- **Profile failed attempts** — extract learning before reverting
