---
name: bench
description: Formal benchmark — atomic compile + trial ALL configs for comprehensive kernel assessment
user-invocable: true
argument-hint: <kernel> [--arch smXX] [--gpu N] [--impls impl1 impl2 ...]
---

# Formal Benchmark

**`ik:bench` is the sole official formal benchmark.** All performance results,
improvement decisions, and gem records come exclusively from `ik:bench`. Results
from `ik:exec` or `ik:optimize` trials are preliminary and never authoritative.

Atomic compile + trial of ALL configs for a kernel. Not for iterative
development — use `/ik:exec` for that.

## What it does

1. **Snapshot** sources + configs to `~/kernel_lab_kb` (before any compile/trial)
2. Discovers all implementations via `cuda_exec.impls` from the snapshot
3. Loads ALL configs from the snapshot
4. Compiles + trials each `.cu` gen impl against the primary reference
5. `.py` gen impls are measured on the reference side during `.cu` trials
6. Writes per-impl `results.json` + `report.md` to `kernel_lab_kb`
7. If any config beats the previous best, creates a **gem** (versioned record)

All compile/trial runs against the snapshot copy — never the original files.

## Usage

```bash
cd /home/zhenc/kernel_lab
```

### Run benchmark

```bash
.venv/bin/python -m cuda_exec.formal bench.kernel=matmul bench.arch=sm90 bench.gpu=5
.venv/bin/python -m cuda_exec.formal bench.kernel=fa4 bench.arch=sm90 bench.gpu=5
```

### Specific implementations

```bash
.venv/bin/python -m cuda_exec.formal bench.kernel=matmul bench.arch=sm90 bench.gpu=5 'bench.impls=[ref-cublas,gen-cuda]'
```

### Custom paths

```bash
# Custom KB repo location
.venv/bin/python -m cuda_exec.formal bench.kernel=vecadd bench.arch=sm90 bench.kb_repo=~/other_kb

# Custom runtime (compile/trial intermediates)
.venv/bin/python -m cuda_exec.formal bench.kernel=vecadd bench.arch=sm90 bench.runtime_root=/tmp/bench_runtime

# Custom data root (skip snapshot, read from specified directory)
.venv/bin/python -m cuda_exec.formal bench.kernel=vecadd bench.arch=sm90 bench.data_root=~/other_project/data
```

### Custom timeout

```bash
.venv/bin/python -m cuda_exec.formal bench.kernel=fa4 bench.arch=sm90 bench.gpu=5 bench.timeout=600
```

## Output Structure

### kernel_lab_kb (git repo, text only)

```
ik_bench/
  runs/<kernel>/<arch>/<YYYYMMDD_HHMMSS>/
    command.json                    # invocation parameters
    data/ref/<kernel>/              # snapshot of source files
    data/gen/<arch>/<kernel>/
    data/configs/<kernel>.json
    impls/<impl_slug>/
      compile/                      # ptxas + resource usage text
      results.json                  # structured results (compact)
      report.md                     # human-readable table
  gems/<kernel>/<arch>/<impl_slug>/
    v001_<YYYYMMDD_HHMMSS>/         # first gem = first run
    v002_<YYYYMMDD_HHMMSS>/         # only created when a config beats v001
    ...                             # gems are never deleted
```

### Local runtime (not in git)

```
~/.cuda_exec/                       # ik:exec runtime (unchanged)
~/.cuda_exec_bench/                 # ik:bench runtime (isolated)
  <kernel>/<arch>/<YYYYMMDD_HHMMSS>/
    workspace/, artifacts/, logs/, state/
```

## Gem Rules

A new gem is created when at least one config is faster than the latest gem,
subject to noise thresholds:

- Absolute improvement > 0.002 ms
- Relative improvement > 0.2%
- Both must be exceeded (AND)
- First run is always a gem
- Gems are never deleted — they form a progression of improvements

## Response Format

The response is produced by `cuda_exec.formal.formal_benchmark()`:

```json
{
  "kernel": "matmul",
  "arch": "sm90",
  "num_configs": 6,
  "impls_requested": ["ref-cublas", "gen-cuda", "gen-cutedsl"],
  "refs": ["ref-cublas"],
  "gens": ["gen-cuda", "gen-cutedsl"],
  "improved": true,
  "gems": {
    "gen-cuda": { "version": 2, "improved_configs": ["mat-8192x8192"], ... }
  },
  "results": {
    "gen-cuda": {
      "impl": "gen-cuda",
      "compile_ok": true,
      "trial_ok": true,
      "compile_result": { ... },
      "trial_result": {
        "all_ok": true,
        "configs": {
          "mat-256x256": {
            "status": "ok",
            "correctness": { "passed": true, ... },
            "performance": { "latency_ms": { "median": 0.0108, ... } },
            "reference": { "performance": { "latency_ms": { "median": 0.033 } } },
            "generated": { "performance": { "latency_ms": { "median": 0.011 } } },
            "cudnn": { "performance": { "latency_ms": { "median": 0.031 } } }
          }
        }
      }
    }
  }
}
```

**Key fields for improvement detection:**
- `improved` (bool): `true` if any impl set a new gem (beat previous best)
- `gems` (dict): per-impl gem info, only for impls that improved. Empty `{}` if no improvement.

**`ik:bench` is the sole authority on improvements.** Results from `ik:exec` or
`ik:optimize` trials are preliminary — only `ik:bench` gems are official.

## Output Format

After the bench completes, **always output a performance comparison table** using
box-drawing characters. This is mandatory — never skip it.

**Columns are dynamic** — generated from the discovered implementations. Each
`ref-*` and `gen-*` impl slug becomes a column. The table adapts to whatever
implementations exist for the kernel+arch combination.

Example for `matmul/sm90` with `ref-cublas`, `gen-cutedsl`, `gen-cuda`:

```
┌────────────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────┬──────────┬───┐
│ NVIDIA H100 SXM5       │  ref-cublas      │  gen-cutedsl     │  gen-cuda        │ cutedsl  │ gen-cuda │   │
│ GPU4, torch 2.11+cu128 │  TFLOPS   (ms)   │  TFLOPS   (ms)   │  TFLOPS   (ms)   │ vs ref   │ vs ref   │   │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┼───┤
│ mat-256x256            │    1.0  (0.033)  │    0.4  (0.091)  │    3.1  (0.011)  │  0.37×   │  3.07×   │ ✗ │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┼───┤
│ mat-8192x8192          │  737.7  (1.491)  │  574.5  (1.915)  │  751.1  (1.464)  │  0.78×   │  1.02×   │ ✗ │
└────────────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────┴──────────┴───┘
```

**How to build the table:**
- **Discover columns** from `result["impls_requested"]` — each impl slug is a column
- **First ref-* impl** is the baseline for speedup ratios
- **TFLOPS + (ms)** for each impl: extract from `reference`, `generated`, `cudnn` fields in trial results
- **Speedup columns**: each non-baseline impl vs the first ref-* baseline
- **Correctness**: `✓` or `✗` per config row, from `correctness.passed`
- **Header row 1**: GPU model, host name
- **Header row 2**: GPU index, torch version
- **TFLOPS**: calculated from the kernel's FLOPs formula (2*M*N*K for matmul, etc.)

---

## Key differences from ik:exec

| | ik:exec | ik:bench |
|---|---|---|
| Source files | Original `data/` | Snapshot in `kernel_lab_kb` |
| Runtime | `~/.cuda_exec/` | `~/.cuda_exec_bench/` (isolated) |
| Compile | Separate step, Solver controls | Bundled, automatic |
| Configs | Solver picks subset | ALL configs, mandatory |
| Impls | Single impl per session | ALL impls (or specified subset) |
| Profile | Yes (NCU deep dive) | No |
| Turn mgmt | Solver manages turns | One-shot, auto-generated |
| Results | In `~/.cuda_exec/` only | `kernel_lab_kb` runs + gems |
| Used by | Solver (development) | Judge (formal assessment) |

## Rules

- **ALL configs** — every config in the fixture is trialed, no exceptions
- **At least one ref-* impl** — enforced by `resolve_impls()`
- **No profiling** — bench is for measurement, not diagnosis
- **Atomic** — compile failure stops immediately, no partial results
- **Snapshot-first** — sources are snapshotted before compile, never evaluate original files
- **Read-only intent** — bench does not modify source files

**GPU is session-sticky**: once a GPU index is set by ANY ik skill (ik:exec, ik:bench, ik:optimize), ALL subsequent ik skill invocations in the same session MUST use that same GPU — unless the user explicitly provides a new `--gpu` value to override it.

## Hydra Config

All settings in `conf/bench/default.yaml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `kernel` | required | Kernel name (matmul, vecadd, fa4) |
| `arch` | required | GPU architecture (sm90, sm120) |
| `impls` | `all` | "all" or list of impl slugs |
| `timeout` | `300` | Per-config timeout (seconds) |
| `gpu` | `null` | GPU index (sets CUDA_VISIBLE_DEVICES; null = use env) |
| `kb_repo` | `~/kernel_lab_kb` | KB repo path for runs + gems |
| `runtime_root` | `~/.cuda_exec` | Local runtime for intermediates |
| `data_root` | `null` | Source data root (null = project data/) |
