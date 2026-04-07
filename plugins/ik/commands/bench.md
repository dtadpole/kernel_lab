---
name: bench
description: Formal benchmark — atomic compile + trial ALL configs for comprehensive kernel assessment
user-invocable: true
argument-hint: <kernel> [gpu=N] [arch=smXX] [impls=impl1,impl2]
---

# Formal Benchmark

**`ik:bench` is the sole official formal benchmark.** All performance results,
improvement decisions, and gem records come exclusively from `ik:bench`. Results
from `ik:exec` or `ik:optimize` trials are preliminary and never authoritative.

Atomic compile + trial of ALL configs for a kernel. Not for iterative
development — use `/ik:exec` for that.

## Implementation Slug Resolution

Implementations are discovered dynamically from the directory structure.
A **slug** has the format `{source}-{name}`:

| Source | Directory | Example slug | Example path |
|--------|-----------|--------------|--------------|
| `ref` | `ref/{kernel}/` | `ref-pytorch` | `ref/matmul/cublas.py` |
| `gen` | `gen/{arch}/{kernel}/` | `gen-cuda` | `gen/sm90/matmul/cuda.cu` |

**Forward resolution** (slug → files): `resolve_impl(kernel, arch, slug)`
- Tries `{name}.py` first, then `{name}.cu` in the source directory
- Auto-includes helper files (`.py` helpers for `.py` entry points, `.h`/`.cuh` for `.cu`)

**Reverse discovery** (directory → all slugs): `list_impls(kernel, arch)`
- Scans `ref/{kernel}/` — every `.py` or `.cu` file becomes `ref-{stem}`
- Scans `gen/{arch}/{kernel}/` — `.cu` files become `gen-{stem}`, `.py` files with `class Model` become `gen-{stem}`

When `impls` is `all` (default), both directions are used: reverse discovery
finds all slugs, then forward resolution loads their files.

Both functions live in `cuda_exec/impls.py` and accept `data_root` to work
against either original `data/` or a snapshot copy.

## What it does

1. **Snapshot** sources + configs to `~/kernel_lab_kb` (before any compile/trial)
2. **Discovers** all implementation slugs from the snapshot via `list_impls()`
3. **Resolves** each slug to its source files via `resolve_impl()`
4. Loads ALL configs from the snapshot
5. Compiles + trials each `.cu` gen impl against the primary `ref-*` impl
6. `.py` gen impls are measured on the reference side during `.cu` trials
7. Writes per-impl `results.json` + `report.md` to `kernel_lab_kb`
8. If any config beats the previous best, creates a **gem** (versioned record)

All compile/trial runs against the snapshot copy — never the original files.

## Environment Resolution

The benchmark auto-resolves the host environment via `cuda_exec.host_env`
by matching the current hostname against `conf/hosts/default.yaml`:

| What | Source | Override |
|------|--------|----------|
| `arch` | `env.torch_cuda_arch` or nvidia-smi | `bench.arch=sm90` |
| `gpu` | `benchmark.cuda_visible_devices` | `bench.gpu=5` |
| `CUDA_HOME` | `env.cuda_home` | (env var) |
| `LD_PRELOAD` | `env.ld_preload` | (env var) |

No manual environment setup is needed — `formal.py` reads the host config and
applies the right settings automatically.

## Usage

```bash
cd /home/zhenc/kernel_lab
```

### Run benchmark (only kernel is required)

```bash
.venv/bin/python -m cuda_exec.formal bench.kernel=matmul
.venv/bin/python -m cuda_exec.formal bench.kernel=fa4
```

### Overrides

```bash
# Specific GPU, arch, impls, or timeout
.venv/bin/python -m cuda_exec.formal bench.kernel=matmul bench.gpu=5
.venv/bin/python -m cuda_exec.formal bench.kernel=matmul bench.arch=sm90
.venv/bin/python -m cuda_exec.formal bench.kernel=matmul 'bench.impls=[ref-pytorch,gen-cuda]'
.venv/bin/python -m cuda_exec.formal bench.kernel=fa4 bench.timeout=600
```

## Output Structure

### kernel_lab_kb (git repo, text only)

```
runs/
  runs/<kernel>/<arch>/<YYYYMMDD_HHMMSS>/
    command.json                    # invocation parameters
    ref/<kernel>/              # snapshot of source files
    gen/<arch>/<kernel>/
    configs/<kernel>.json
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
  "kernel": "<kernel>",
  "arch": "<arch>",
  "num_configs": 6,
  "impls_requested": ["ref-<name>", "gen-<name1>", "gen-<name2>"],
  "refs": ["ref-<name>"],
  "gens": ["gen-<name1>", "gen-<name2>"],
  "improved": true,
  "gems": {
    "gen-<name1>": { "version": 2, "improved_configs": ["<config_slug>"], ... }
  },
  "results": {
    "gen-<name1>": {
      "impl": "gen-<name1>",
      "compile_ok": true,
      "trial_ok": true,
      "compile_result": { ... },
      "trial_result": {
        "all_ok": true,
        "configs": {
          "<config_slug>": {
            "status": "ok",
            "correctness": { "passed": true, ... },
            "performance": { "latency_ms": { "median": 0.0108, ... } },
            "reference": { "performance": { "latency_ms": { "median": 0.033 } } },
            "generated": { "performance": { "latency_ms": { "median": 0.011 } } }
          }
        }
      }
    }
  }
}
```

Slugs in the response (`impls_requested`, `refs`, `gens`, `results` keys, `gems` keys)
are all dynamically discovered — they match whatever `ref-*` and `gen-*` files exist
in the data directories.

**Key fields for improvement detection:**
- `improved` (bool): `true` if any impl set a new gem (beat previous best)
- `gems` (dict): per-impl gem info, only for impls that improved. Empty `{}` if no improvement.

**`ik:bench` is the sole authority on improvements.** Results from `ik:exec` or
`ik:optimize` trials are preliminary — only `ik:bench` gems are official.

## Output Format

After the bench completes, **always output the performance comparison table
printed by `formal.py`**. This is mandatory — never skip it. The table is
generated by `format_results_table()` and printed to stdout automatically.

The table format:

```
**NVIDIA H100 SXM R&R SKU 96GB** | host: h8_4 | driver: 595.45.04 | CUDA: 13.2 | peak: 800.0 TFLOPS (BF16 TC)

| Config                  | ref-cudnn              | ref-cutedsl                  | ref-pytorch            | peak-cuda              |
|                         | cuDNN 9.19.0           | FA4 4.0.0b7, DSL 4.4.2      | PyTorch 2.11.0+cu130   |                        |
|                         | ms    TFLOPS           | ms    TFLOPS   speedup       | ms    TFLOPS   speedup | ms    TFLOPS   speedup |
|-------------------------|------------------------|------------------------------|------------------------|------------------------|
| mha-causal-b8-s4096     |  0.978   562.1         |  0.988   556.6 ✓ 0.99x       |   1.01   545.2 ✓ 0.97x |  0.911   603.1 ✓ 1.07x |
| mha-causal-b4-s8192     |   1.85   594.0         |   1.80   611.6 ✓ 1.03x       |   1.84   597.2 ✓ 1.01x |   1.67   657.9 ✓ 1.11x |
| mha-causal-b2-s16384    |   3.59   611.9         |   3.46   635.6 ✓ 1.04x       |   3.69   595.6 ✓ 0.97x |   3.12   704.8 ✓ 1.15x |
| mha-noncausal-b8-s4096  |   1.63   673.3         |   1.64   669.7 ✓ 0.99x       |   1.72   640.6 ✓ 0.95x |   1.59   689.9 ✓ 1.02x |
| mha-noncausal-b4-s8192  |   3.20   687.5         |   3.28   669.7 ✓ 0.97x       |   3.42   643.7 ✓ 0.94x |   3.09   711.0 ✓ 1.03x |
| mha-noncausal-b2-s16384 |   6.36   691.4         |   6.45   681.7 ✓ 0.99x       |   7.18   612.4 ✓ 0.89x |   6.10   720.9 ✓ 1.04x |
|-------------------------|------------------------|------------------------------|------------------------|------------------------|
| % of peak               |                  86.4% |                        85.2% |                  80.5% |                  90.1% |

Golden: ref-cudnn — ✓/✗ = correctness vs golden
```

**Table structure:**

1. **Environment line** (above table): GPU model, host slug, driver version,
   CUDA toolkit version, peak TFLOPS
2. **Header row 1**: impl slugs (column names)
3. **Header row 2**: version labels per impl — auto-detected:
   - `ref-cudnn`: cuDNN version (e.g. `cuDNN 9.19.0`)
   - `ref-cutedsl`: FA4 + CUTLASS DSL versions (e.g. `FA4 4.0.0b7, DSL 4.4.2`)
   - `ref-pytorch`: PyTorch version with CUDA suffix (e.g. `PyTorch 2.11.0+cu130`)
   - `peak-cuda` / `gen-cuda`: empty (hand-written code, no library version)
4. **Header row 3**: `ms    TFLOPS   speedup` sub-labels
5. **Data rows**: `ms  TFLOPS ✓/✗ speedup` per config per impl
6. **% of peak row**: best TFLOPS / GPU peak
7. **Footer**: golden identity + correctness legend

**Speedup**: 2 decimal places (e.g. `1.07x`, `0.97x`) — important for
detecting 1-2% improvements. Golden column has no speedup (it IS the baseline).

**Correctness**: ✓/✗ vs golden. Golden column has no marker.

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

**GPU is session-sticky**: once a GPU index is set by ANY ik skill (ik:exec, ik:bench, ik:optimize), ALL subsequent ik skill invocations in the same session MUST use that same GPU — unless the user explicitly provides a new `gpu=` value to override it.

## Hydra Config

All settings in `conf/bench/default.yaml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `kernel` | required | Kernel name (matmul, vecadd, fa4) |
| `arch` | `auto` | GPU architecture — auto-detected from host config / nvidia-smi. Override: sm90, sm120 |
| `impls` | `all` | "all" or list of impl slugs |
| `timeout` | `300` | Per-config timeout (seconds) |
| `gpu` | `null` | GPU index — auto-resolved from host config `benchmark.cuda_visible_devices` when null |
| `kb_repo` | `~/kernel_lab_kb` | KB repo path for runs + gems |
| `runtime_root` | `~/.cuda_exec` | Local runtime for intermediates |
| `data_root` | `null` | Source data root (null = project data/) |
