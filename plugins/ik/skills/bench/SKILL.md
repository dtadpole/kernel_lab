---
name: bench
description: Formal benchmark — atomic compile + trial ALL configs for comprehensive kernel assessment
user-invocable: true
argument-hint: <kernel> [--arch smXX]
---

# Formal Benchmark

Atomic compile + trial of ALL configs for a kernel. Used by the Formal Evaluator
(Judge) agent for comprehensive, authoritative assessment. Not for iterative
development — use `/ik:exec` for that.

## What it does

1. Loads ALL source files (reference, generated, optional cuDNN) from fixtures
2. Loads ALL configs from `data/fixtures/{arch}/{kernel}/configs.json`
3. Compiles the kernel (bundled, not a separate step)
4. Trials every config (correctness + latency comparison)
5. Returns a single comprehensive verdict

## Usage

```bash
cd /home/zhenc/kernel_lab
```

### Auto-detect GPU arch

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -c "
from cuda_exec.formal import formal_benchmark
import json, subprocess

# Auto-detect arch from GPU
cc = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
    text=True
).strip().split('\n')[0]
arch = 'sm' + cc.replace('.', '')

result = formal_benchmark(kernel='$KERNEL', arch=arch)
print(json.dumps(result, indent=2, default=str))
"
```

### Explicit arch

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -c "
from cuda_exec.formal import formal_benchmark
import json

result = formal_benchmark(kernel='fa4', arch='sm90')
print(json.dumps(result, indent=2, default=str))
"
```

### CLI mode

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m cuda_exec.formal fa4 sm90
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m cuda_exec.formal matmul sm90 --run-tag bench_v2
```

## Response Format

```json
{
  "compile_ok": true,
  "trial_ok": true,
  "kernel": "fa4",
  "arch": "sm90",
  "num_configs": 8,
  "compile_result": { ... },
  "trial_result": {
    "all_ok": true,
    "configs": {
      "config-slug-1": {
        "status": "ok",
        "correctness": { "passed": true, ... },
        "performance": { "latency_ms": { ... }, ... }
      },
      ...
    }
  }
}
```

## Key differences from ik:exec

| | ik:exec | ik:bench |
|---|---|---|
| Compile | Separate step, Solver controls | Bundled, automatic |
| Configs | Solver picks subset | ALL configs, mandatory |
| Profile | Yes (NCU deep dive) | No |
| Turn mgmt | Solver manages turns | One-shot |
| Used by | Solver (development) | Judge (formal assessment) |

## Rules

- **ALL configs** — every config in the fixture is trialed, no exceptions
- **No profiling** — bench is for measurement, not diagnosis
- **Atomic** — compile failure stops immediately, no partial results
- **Read-only intent** — bench does not modify source files
