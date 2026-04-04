---
name: optimize-kernel
description: Autonomous CUDA kernel optimization loop — profile, analyze, brainstorm, implement, verify, commit
user-invocable: true
disable-model-invocation: true
argument-hint: <kernel> [--impl generated|reference] [--target smXX]
---

# Optimize Kernel

Autonomous optimization loop for CUDA kernels. Profiles current performance,
identifies gaps, brainstorms data-driven ideas, implements the most promising
one, verifies improvement, and commits if successful.

## Arguments

| Position | Name | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `$0` | kernel | **yes** | — | Kernel name: `fa4`, `matmul`, etc. |
| `--impl` | implementation | no | `generated` | Which code to optimize: `generated` (hand-written CUDA) or `reference` (CuTe DSL) |
| `--target` | target arch | no | auto-detect | GPU arch target, e.g. `sm90`, `sm120`. Auto-detected from local GPU if omitted |

Parse `$ARGUMENTS` to extract these. Example invocations:
```
/optimize-kernel fa4
/optimize-kernel matmul --impl reference
/optimize-kernel fa4 --target sm90
```

## Auto-Detection

If `--target` is not provided, detect from local GPU:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```
Map compute capability to arch: `12.0` -> `sm120`, `9.0` -> `sm90`, `8.0` -> `sm80`.

## Project Layout

Before starting, know where things live:

| What | Path |
|------|------|
| Fixture configs | `data/fixtures/{arch}/{kernel}/configs.json` |
| CuTe DSL reference | `data/fixtures/{arch}/{kernel}/cutedsl.py` |
| cuDNN wrapper | `data/fixtures/{arch}/{kernel}/cudnn.py` (if exists) |
| Generated code | `data/generated/{arch}/{kernel}/generated.cu` |
| Results | `results/{arch}/{gpu_name}/{kernel}/` |
| Roofline specs | `docs/roofline/` |
| NVIDIA docs (local) | Use `/kb:docs` to search indexed CUDA Toolkit docs |
| NVIDIA docs (online) | Web search for official NVIDIA docs, PTX ISA, tuning guides |

## The Loop

Run this loop **autonomously without user input** until a performance
improvement is achieved. Each iteration consists of the phases below.

### Phase 1: Find the Facts

#### 1a. Gather Current Performance

Read the latest results files in `results/{arch}/{gpu_name}/{kernel}/` to
understand current performance across all configs. Sort by date to find the
most recent.

Then run a fresh evaluation to get ground-truth numbers for this session:

1. Use `/cuda:exec` to **compile** the current generated code
2. Use `/cuda:exec` to **evaluate** across **ALL** configs in `data/fixtures/{arch}/{kernel}/configs.json`
3. Record latency for generated code vs reference (CuTe DSL) vs cuDNN/cuBLAS

**After every full evaluation, output a performance comparison table** like the
example below. This table is mandatory after evaluate-all in both Phase 1 and
Phase 5. Use box-drawing characters for the table border.

```
┌────────────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────┬──────────┐
│ NVIDIA H100 (h8_3)     │   cuDNN 9.19.0   │   FA4 CuTe DSL   │  Generated CUDA  │ FA4 DSL  │ Gen CUDA │
│ GPU4, torch 2.11+cu128 │  TFLOPS   (ms)   │  v4.0.0b7  (ms)  │  TFLOPS   (ms)   │ vs cuDNN │ vs cuDNN │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ mha-causal-b8-s4096    │  565.5  (0.972)  │  549.2  (1.001)  │  381.7  (1.440)  │  0.97×   │  0.67×   │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ mha-causal-b4-s8192    │  599.8  (1.833)  │  558.6  (1.968)  │  388.7  (2.829)  │  0.93×   │  0.65×   │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ mha-causal-b2-s16384   │  619.3  (3.551)  │  591.2  (3.720)  │  412.8  (5.327)  │  0.95×   │  0.67×   │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ mha-noncausal-b8-s4096 │  625.2  (1.759)  │  647.1  (1.699)  │  432.9  (2.540)  │  1.04×   │  0.69×   │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ mha-noncausal-b4-s8192 │  616.3  (3.568)  │  633.3  (3.472)  │  452.2  (4.863)  │  1.03×   │  0.73×   │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ mha-noncausal-b2-s16384│  597.2  (7.365)  │  655.2  (6.712)  │  469.4  (9.370)  │  1.10×   │  0.79×   │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ % of peak (best cfg)   │      78.2%       │      81.9%       │      58.7%       │          │ 800.0TF  │
└────────────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────┴──────────┘
```

Column definitions:
- **cuDNN / FA4 CuTe DSL / Generated CUDA**: TFLOPS (effective throughput) and (ms) latency
- **FA4 DSL vs cuDNN**: Speedup ratio (>1.0 = FA4 CuTe DSL is faster)
- **Gen CUDA vs cuDNN**: Speedup ratio (>1.0 = generated is faster)
- **% of peak**: Best config's TFLOPS / GPU theoretical peak
- Header row 1: GPU model, host name. Row 2: GPU index, torch version
- Reference columns show library version: cuDNN version, flash-attn-4 version
- Reference columns show library version: cuDNN version, flash-attn-4 version

#### 1b. Profile Selectively

**Do NOT profile all configs.** Profiling is expensive. Pick 1-2 configs that
are most representative or most interesting based on the evaluation results:
- The config with the **largest gap** vs reference/cuDNN (biggest opportunity)
- One **representative** config (e.g., the most common batch/seq_len combo)

Use `/cuda:exec` to **profile** with NCU on the selected config(s):
- Profile the **generated** kernel
- Profile the **reference** kernel (CuTe DSL side) on the same config

Collect key NCU metrics:
- Compute throughput (SM %)
- Memory throughput (DRAM %)
- Achieved occupancy
- Warp stall reasons
- L1/L2 hit rates
- Instructions per cycle
- Register usage, spill loads/stores

#### 1c. Examine Assembly

Use `/cuda:inspect` to review:
- **SASS** — actual GPU instructions, look for inefficiencies
- **PTX** — compiler input, check for unnecessary barriers or redundant ops
- **Resource usage** — registers per thread, shared memory, spill bytes

Pay attention to:
- Register spills (spill stores/loads > 0 is a red flag)
- Instruction mix (ratio of compute vs memory vs control)
- Barrier patterns (excessive `bar.sync` or `membar`)
- Bank conflicts in shared memory access patterns

### Phase 2: Analyze the Gap

If there is a performance gap between the implementation being optimized and
the best baseline:

#### 2a. Compare Profiles

Side-by-side the NCU metrics between generated and reference/cuDNN. Identify
which hardware resource is the bottleneck:
- **Compute-bound**: SM % is high, memory % is low
- **Memory-bound**: DRAM % is high, SM % is low
- **Latency-bound**: Both are low — look at warp stalls

#### 2b. Consult NVIDIA Documentation (Source of Truth)

**Documentation is the ground truth. Always consult docs before forming
hypotheses.** Use two complementary channels:

**Local indexed docs** — `/kb:docs`:
- CUDA C++ Programming Guide (memory model, warp-level primitives, async copy)
- PTX ISA Reference (instruction semantics, latency, constraints)
- CUDA Best Practices Guide / Tuning Guide
- Architecture-specific features (e.g., SM90 WGMMA, SM120 tcgen05)

**Online NVIDIA docs** — web search:
- Search `site:docs.nvidia.com` or `site:developer.nvidia.com` for specific
  instructions, intrinsics, or hardware features
- PTX ISA changelog for new instructions on the target arch
- CUTLASS/CuTe source-level documentation on GitHub
- Nsight Compute metrics interpretation guides

**What to look for:**
- Exact instruction latencies and throughput for the target SM
- Memory hierarchy behavior (L1/L2 sector size, cache policies)
- Async copy / TMA programming constraints
- Warp scheduling and instruction interleaving rules
- Architecture-specific limitations or opportunities

When profiling data contradicts your assumptions, go back to docs to understand
why. The docs are authoritative — NCU data confirms or reveals, docs explain.

#### 2c. Check Roofline

Read `docs/roofline/` specs for the target GPU. Calculate:
- Arithmetic intensity of the kernel (FLOPs / bytes transferred)
- Theoretical peak for this workload given compute vs memory bound
- Current achieved % of roofline ceiling
- Whether optimization should target compute, memory, or latency

#### 2d. Search for External Insights

Use web search for **grounded, verifiable** information:
- NVIDIA GTC talks and whitepapers with published benchmarks
- Peer-reviewed papers with reproducible results
- Official NVIDIA blog posts with performance data
- CUTLASS / FlashAttention GitHub issues and commit messages

**Prioritize source-of-truth**: NVIDIA official docs > published papers >
reproducible benchmarks > blog posts > forum posts. Discard any advice that
lacks measurable evidence or conflicts with official documentation.

#### 2e. Study Previous Implementations and Results

Two key sources of institutional knowledge beyond docs and profiles:

**Previous implementations** — compare different implementations of the same
kernel to understand what techniques each uses:
- Compare CuTe DSL reference (`data/fixtures/{arch}/{kernel}/cutedsl.py`) vs
  hand-written CUDA (`data/generated/{arch}/{kernel}/generated.cu`) — what
  instructions, tile sizes, and pipeline strategies does each use?
- Use `git log` to find earlier versions of the generated code:
  ```bash
  git log --oneline --all -- data/generated/{arch}/{kernel}/generated.cu
  ```
- Read previous versions for ideas:
  ```bash
  git show <commit>:data/generated/{arch}/{kernel}/generated.cu
  ```

**Previous results** — the `results/{arch}/{gpu_name}/{kernel}/` folder is a
knowledge base of past optimization attempts. Each results file documents:
- What was tried and why
- NCU profiling data before/after
- What worked, what didn't, and root cause analysis
- Architectural insights discovered during optimization

Always read ALL results files for the target kernel before brainstorming.
Failed experiments are especially valuable — they document constraints and
dead ends that should not be revisited.

### Phase 3: Brainstorm Ideas

Based on all gathered evidence, generate a list of **concrete, specific**
optimization ideas. Each idea must be:

1. **Grounded** — backed by profiling data, documentation, or prior results
2. **Specific** — not "improve memory access" but "coalesce V tile loads by
   transposing the shared memory layout from [BLOCK_KV][DIM] to [DIM][BLOCK_KV]"
3. **Measurable** — clear prediction of which NCU metric should improve
4. **Feasible** — implementable in the current code structure

Rank ideas by expected impact and confidence. Pick the single most promising
idea to implement first.

### Phase 4: Plan and Implement

1. **Snapshot baseline** — before modifying any code, save the current file:
   ```bash
   cp data/generated/{arch}/{kernel}/generated.cu data/generated/{arch}/{kernel}/generated.cu.baseline
   ```
   This enables clean revert if the attempt fails.
2. **Write a plan** — describe exactly what code changes are needed, which
   files to modify, and what the expected effect is. Include:
   - Which NCU metric you expect to improve and by how much
   - Which lines of code are changing and why
   - What could go wrong (register pressure, correctness, bank conflicts)
3. **Implement** — make the code changes in `data/generated/{arch}/{kernel}/generated.cu`
   (or the CuTe DSL reference code if `--impl reference`)
4. **Increment turn** — new code = new turn in cuda_exec

### Phase 5: Verify

1. **Compile** the modified code via `/cuda:exec`
   - Check compile output: register count, spill bytes, shared memory
   - If compile fails, fix and retry (up to 3 compile attempts)
   - If register spills increased substantially, reconsider the approach
2. **Evaluate** across **ALL** configs (every config, no exceptions)
   - Check correctness first — all configs must pass
   - If correctness fails, fix and retry (up to 3 correctness attempts)
   - **Output the performance comparison table** (same format as Phase 1a)
     after evaluate-all completes — this is mandatory
3. **Profile selectively** — NCU profiling is expensive. Only profile the
   **same 1-2 config(s)** profiled in Phase 1b, to compare the specific NCU
   metrics targeted by this optimization. Do NOT profile all configs.
4. **Compare** new latency vs baseline from Phase 1

#### Decision Point

```
                    ┌─────────────────────┐
                    │ Evaluate & Profile  │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         Improved      No change      Regression
              │              │              │
              v              v              v
          Phase 6       Analyze why    Revert code
          (commit)      ┌────┘         to baseline
                        │              ┌────┘
                        v              v
                   Profile the    Analyze why
                   attempt to     it regressed
                   understand     ┌────┘
                        │              │
                        v              v
                   Update mental   Update mental
                   model with      model with
                   new evidence    new evidence
                        │              │
                        └──────┬───────┘
                               v
                        Return to Phase 3
                        (next idea on list)
```

- **Improvement confirmed** (any config improved, no config regressed > 2%) → **Phase 6** (commit)
- **No improvement or regression** → **Phase 7** (revert, learn, retry)
- **Correctness failure after 3 fix attempts** → **Phase 7** (revert, move to next idea)

### Phase 6: Commit Results (only if improvement confirmed)

**Gate: only enter this phase if Phase 5 confirmed improvement.** If there was
no improvement or a regression, skip to Phase 7 instead.

#### 6a. Write Results

Create a results file at:
```
results/{arch}/{gpu_name}/{kernel}/YYYYMMDD_HHMM_{description}.md
```

Include:
- Hardware specs
- Objective and approach
- Before/after performance table (all configs)
- Key changes made
- NCU metrics comparison (before vs after)
- What worked and why

#### 6b. Commit and Push

```bash
git add data/generated/{arch}/{kernel}/generated.cu   # or reference file
git add results/{arch}/{gpu_name}/{kernel}/*.md
git commit -m "perf: {kernel} — {short description of optimization}

{1-2 sentence summary of what changed and the improvement}"
git push
```

#### 6c. Report

Print a summary:
```
=== Optimization Complete ===
Kernel: {kernel}
Target: {arch} ({gpu_name})
Change: {description}
Result: {improvement summary across configs}
Commit: {hash}
```

### Phase 7: Retry Loop (no improvement or regression)

**Enter this phase when Phase 5 shows no improvement or a regression.**

#### 7a. Revert to Baseline

```bash
cp data/generated/{arch}/{kernel}/generated.cu.baseline data/generated/{arch}/{kernel}/generated.cu
```

For no-improvement cases: profile the failed attempt BEFORE reverting to
extract learning from the NCU data. For regressions: revert immediately.

#### 7b. Update Mental Model

Record what was tried and why it didn't work:
- Which NCU metrics changed (or didn't)
- What the profiling data revealed about the failed hypothesis
- Any new constraints or insights discovered

#### 7c. Loop Back

Return to **Phase 2** (re-analyze with new evidence) and **Phase 3** (pick
the next idea from the ranked list). Then proceed through Phase 4 → Phase 5
again.

#### 7d. Stop Conditions

- **After 3 failed ideas**: Profile baseline one final time with fresh eyes.
  Consult NVIDIA docs for anything missed. Try one last idea informed by all
  accumulated evidence.
- **After 4 total failed ideas**: Revert to baseline, write a findings summary
  to results (what was tried, what was learned, why nothing worked), and stop.
  **Never commit a regression.**

## Key Principles

- **Docs are source of truth** — always consult NVIDIA documentation before
  and after profiling. NCU data shows *what*, docs explain *why*
- **Data-driven** — every decision backed by profiling data or documentation
- **One change at a time** — isolate variables, measure each change independently
- **Correctness first** — never sacrifice correctness for performance
- **Never commit a regression** — revert to baseline if performance degrades
- **Profile failed attempts** — a failed idea still produces useful profiling
  data. Extract the learning before reverting
- **Record everything** — results files capture institutional knowledge, even
  for failed attempts (document what didn't work and why)
