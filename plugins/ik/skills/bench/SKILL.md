---
name: bench
description: Formal benchmark — atomic compile + trial ALL configs for comprehensive kernel assessment
user-invocable: true
argument-hint: <kernel> [--arch smXX] [--impls impl1 impl2 ...]
---

# Formal Benchmark

Atomic compile + trial of ALL configs for a kernel. Used by the Formal Evaluator
(Judge) agent for comprehensive, authoritative assessment. Not for iterative
development — use `/ik:exec` for that.

## What it does

1. Discovers all implementations via `cuda_exec.impls` (dynamic slug resolution)
2. Loads ALL configs from `data/configs/{kernel}.json`
3. For each `.cu` gen impl: compiles + trials every config against the primary reference
4. `.py` gen impls are measured on the reference side during `.cu` trials
5. Returns per-implementation results with correctness + latency

## Usage

```bash
cd /home/zhenc/kernel_lab
```

### Run benchmark (auto-detects GPU arch)

```bash
CUDA_VISIBLE_DEVICES=5 .venv/bin/python -c "
from cuda_exec.formal import formal_benchmark
import json, subprocess

cc = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
    text=True
).strip().split('\n')[0]
arch = 'sm' + cc.replace('.', '')

result = formal_benchmark(kernel='$KERNEL', arch=arch)
print(json.dumps(result, indent=2, default=str))
"
```

### CLI mode

```bash
CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m cuda_exec.formal matmul sm90
CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m cuda_exec.formal fa4 sm90 --impls ref-cublas gen-cuda
```

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
            "correctness": { "passed": false, ... },
            "performance": { "latency_ms": { "median": 0.0108, ... } },
            "reference": { "performance": { "latency_ms": { "median": 0.033 } } },
            "generated": { "performance": { "latency_ms": { "median": 0.011 } } },
            "cudnn": { "performance": { "latency_ms": { "median": 0.031 } } }
          }
        }
      }
    },
    "gen-cutedsl": {
      "impl": "gen-cutedsl",
      "compile_ok": null,
      "trial_ok": null,
      "note": "Python impl — measured as reference/cudnn side when .cu impls are trialed"
    }
  }
}
```

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
| Compile | Separate step, Solver controls | Bundled, automatic |
| Configs | Solver picks subset | ALL configs, mandatory |
| Impls | Single impl per session | ALL impls (or specified subset) |
| Profile | Yes (NCU deep dive) | No |
| Turn mgmt | Solver manages turns | One-shot, auto-generated |
| Used by | Solver (development) | Judge (formal assessment) |

## Rules

- **ALL configs** — every config in the fixture is trialed, no exceptions
- **At least one ref-* impl** — enforced by `resolve_impls()`
- **No profiling** — bench is for measurement, not diagnosis
- **Atomic** — compile failure stops immediately, no partial results
- **Read-only intent** — bench does not modify source files
