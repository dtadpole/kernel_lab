---
name: optimize
description: Autonomous CUDA kernel optimization loop — profile, analyze, brainstorm, implement, verify, commit
user-invocable: true
disable-model-invocation: true
argument-hint: <kernel> [--impl generated|reference] [--target smXX] [--gpu N]
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
| `--target` | target arch | no | auto-detect | GPU arch target, e.g. `sm90`, `sm120`. Auto-detected from GPU if omitted |
| `--gpu` | GPU index | no | from CLAUDE.md | GPU device index (sets `CUDA_VISIBLE_DEVICES`). Uses host assignment from CLAUDE.md if omitted |

Parse `$ARGUMENTS` to extract these. Example invocations:
```
/ik:optimize fa4
/ik:optimize matmul --impl reference
/ik:optimize fa4 --target sm90
/ik:optimize fa4 --gpu 4
```

When `--gpu N` is provided, set `CUDA_VISIBLE_DEVICES=N` on all `/ik:exec` and `/ik:bench` commands.

**GPU is session-sticky**: once a GPU index is set by ANY ik skill (ik:exec, ik:bench, ik:optimize), ALL subsequent ik skill invocations in the same session MUST use that same GPU — unless the user explicitly provides a new `--gpu` value to override it.

## Auto-Detection

If `--target` is not provided, detect from GPU:
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
| NVIDIA docs (local) | Use `/ik:docs` to search indexed CUDA Toolkit docs |
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

1. Use `/ik:exec` to **compile** the current generated code
2. Use `/ik:exec` to **trial** across **ALL** configs in `data/fixtures/{arch}/{kernel}/configs.json`
3. Record latency for generated code vs reference (CuTe DSL) vs cuDNN/cuBLAS

**After every full evaluation, output a performance comparison table** like the
example below. This table is mandatory after trial-all in both Phase 1 and
Phase 5. Use box-drawing characters for the table border.

**Use `/ik:bench` for output.** Do not create a custom table — run bench and
use its output format (dynamic columns, correctness indicators, TFLOPS).
If any config shows ✗, the optimization is REJECTED.

#### 1b. Profile Selectively

**Do NOT profile all configs.** Profiling is expensive. Pick 1-2 configs that
are most representative or most interesting based on the evaluation results:
- The config with the **largest gap** vs reference/cuDNN (biggest opportunity)
- One **representative** config (e.g., the most common batch/seq_len combo)

Use `/ik:exec` to **profile** with NCU on the selected config(s):
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

Review from the compile output:
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

**Local indexed docs** — `/ik:docs`:
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

1. **Compile** the modified code via `/ik:exec`
   - Check compile output: register count, spill bytes, shared memory
   - If compile fails, fix and retry (up to 3 compile attempts)
   - If register spills increased substantially, reconsider the approach

2. **Correctness gate** — this is a **hard requirement**, not optional
   - **Trial** across **ALL** configs (every config, no exceptions)
   - **Every config must pass correctness.** If ANY config fails → REJECT.
   - If correctness fails, fix and retry (up to 3 attempts)
   - After 3 failed attempts → **revert immediately**, move to next idea
   - **Never skip correctness.** Never commit code that fails correctness.
   - **Never weaken tolerances** to make a failing kernel pass.

3. **Output the performance comparison table** (same format as Phase 1a)
   after trial-all completes — this is mandatory

4. **Profile selectively** — NCU profiling is expensive. Only profile the
   **same 1-2 config(s)** profiled in Phase 1b, to compare the specific NCU
   metrics targeted by this optimization. Do NOT profile all configs.

5. **Compare** new latency vs baseline from Phase 1. Note: this comparison is
   preliminary — only `/ik:bench` results are official (see Phase 6).

#### Decision Point

```
                    ┌─────────────────────┐
                    │   Trial ALL configs  │
                    └────────┬────────────┘
                             │
                    ┌────────┴────────┐
                    │ Correctness?    │
                    └────────┬────────┘
                             │
                   FAIL ─────┼───── PASS
                     │               │
                     v               │
                  Fix & retry        │
                  (up to 3x)         │
                     │               │
                  Still fails?       │
                     │               │
                     v               │
                  REJECT:            │
                  revert to          │
                  baseline           │
                                     │
                      ┌──────────────┼──────────────┐
                      │              │              │
                 Improved      No change      Regression
                      │              │              │
                      v              v              v
                  Phase 6       Analyze why    Revert code
                  (bench)       then retry     to baseline
```

- **Correctness fails** → REJECT immediately. Revert. No exceptions.
- **Correctness passes + improvement** → Phase 6 (record + bench)
- **Correctness passes + no improvement** → Phase 7 (revert, learn, retry)
- **Correctness passes + regression** → Phase 7 (revert, learn, retry)

### Phase 6: Record Results and Bench (only if improvement confirmed)

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

#### 6b. Final bench and STOP

**Run `/ik:bench` as the last step.** This is mandatory.
The bench output is the authoritative record of performance and correctness.
`ik:bench` handles snapshotting, gem creation, and committing automatically.
If bench shows any ✗, go back and fix.

**After a successful bench, print the performance comparison table and STOP.**
Do not continue optimizing. The user will decide whether to run another round.

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

- **Stop on improvement** — when an optimization improves performance, run
  `/ik:bench`, print the results, and stop. Do not continue optimizing.
- **Keep trying on failure** — when an idea doesn't improve performance,
  revert and immediately try the next idea. Do not ask the user.
- **Autonomous execution** — this is a fully autonomous loop. Make decisions,
  implement changes, evaluate results, and iterate. Only surface results when
  you have something concrete to show (improvement benched or all ideas
  exhausted).
- **Docs are source of truth** — always consult NVIDIA documentation before
  and after profiling. NCU data shows *what*, docs explain *why*
- **Data-driven** — every decision backed by profiling data or documentation
- **One change at a time** — isolate variables, measure each change independently
- **ik:bench is the sole authority** — only `/ik:bench` results are official.
  `ik:exec` trials are preliminary checks; the `improved` field from `ik:bench`
  is the only valid signal to stop the loop
- **Correctness first** — never sacrifice correctness for performance
- **Never commit a regression** — revert to baseline if performance degrades
- **Profile failed attempts** — a failed idea still produces useful profiling
  data. Extract the learning before reverting
- **Record everything** — results files capture institutional knowledge, even
  for failed attempts (document what didn't work and why)
