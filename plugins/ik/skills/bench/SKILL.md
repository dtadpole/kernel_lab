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

## Output Format

After the bench completes, **always output a performance comparison table** using
box-drawing characters. This is mandatory — never skip it.

```
┌────────────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────┬──────────┐
│ NVIDIA H100 (h8_3)     │   cuDNN 9.19.0   │   CuTe DSL ref   │  Generated CUDA  │ DSL ref  │ Gen CUDA │
│ GPU4, torch 2.11+cu128 │  TFLOPS   (ms)   │  TFLOPS   (ms)   │  TFLOPS   (ms)   │ vs cuDNN │ vs cuDNN │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ mha-causal-b8-s4096    │  565.5  (0.972)  │  549.2  (1.001)  │  381.7  (1.440)  │  0.97×   │  0.67×   │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ mha-causal-b4-s8192    │  599.8  (1.833)  │  558.6  (1.968)  │  388.7  (2.829)  │  0.93×   │  0.65×   │
└────────────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────┴──────────┘
```

**Column definitions:**
- **cuDNN / CuTe DSL ref / Generated CUDA**: TFLOPS (effective throughput) and (ms) median latency
- **DSL ref vs cuDNN**: Speedup ratio (>1.0 = DSL reference is faster than cuDNN)
- **Gen CUDA vs cuDNN**: Speedup ratio (>1.0 = generated is faster than cuDNN)
- Header row 1: GPU model, host name
- Header row 2: GPU index, torch version
- If cuDNN baseline is not available, omit cuDNN column and show "vs DSL ref" instead
- TFLOPS calculation: use the kernel's FLOPs formula from the fixture config (2*M*N*K for matmul, etc.)

**Correctness indicator:** Append a checkmark or cross to each config row:
- `✓` if `correctness.passed == True`
- `✗` if `correctness.passed == False`

---

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
