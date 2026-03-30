# Eval Harness and Support Extraction Design

## Goal

Split cuda_exec's generated-side runtime into a reusable C++ harness (`eval_harness.cu`) and extract shared Python utilities into `eval_support.py`, giving kernel authors a clean interface and eliminating duplicated code between evaluate.py and profile.py.

## Background

Today, every generated `.cu` file is monolithic: it contains the kernel, env parsing, memory management, CUDA event timing, statistics, and JSON output formatting. evaluate.py and profile.py both duplicate shared infrastructure (device locking, watchdog, GPU cleanup, reference loading). This design fixes both problems.

---

## 1. Generated-side contract (C++ harness)

### Kernel author interface

The kernel author writes a single `.cu` file containing:

1. One or more `__global__` kernel functions
2. Four `extern "C"` interface functions

```cpp
#include "eval_harness.h"

// --- Kernel(s) ---
__global__ void vector_add(const float* x, const float* y, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) out[tid] = x[tid] + y[tid];
}

// --- Required interface ---

// Called once before timing. Allocate device memory, copy inputs to GPU.
extern "C" int kernel_setup(const RuntimeConfig* cfg) { ... }

// Called for each warmup and timed trial. Launch the kernel on the given stream.
// Must NOT allocate or free memory. Must NOT synchronize.
extern "C" int kernel_run(cudaStream_t stream) { ... }

// Called once after timing. Copy output to host, return pointer and element count.
extern "C" void kernel_output(float** data, int* size) { ... }

// Called at exit. Free all device and host memory.
extern "C" void kernel_cleanup(void) { ... }
```

### RuntimeConfig struct (provided by eval_harness.h)

```cpp
typedef struct {
    char config_slug[256];
    char config_json[4096];    // raw CUDA_EXEC_CONFIG_JSON
    int  input_size;           // CUDA_EXEC_PARAM_INPUT_SIZE
    int  rank;                 // CUDA_EXEC_PARAM_RANK
    char shape_kind[64];       // CUDA_EXEC_PARAM_SHAPE_KIND
    char shape_json[1024];     // CUDA_EXEC_PARAM_SHAPE (JSON array)
    int  num_warmups;          // default 5
    int  num_trials;           // default 10
} RuntimeConfig;
```

Fields are populated from environment variables by the harness at startup.

### Harness responsibilities (eval_harness.cu)

`eval_harness.cu` provides `main()` and handles:

1. Parse env vars into `RuntimeConfig`
2. Create CUDA stream
3. Call `kernel_setup(&cfg)` — fail if non-zero return
4. Warmup: call `kernel_run(stream)` + `cudaStreamSynchronize(stream)` x `num_warmups`
5. Timed trials: for each trial, record CUDA events around `kernel_run(stream)`, synchronize, measure elapsed time
6. Call `kernel_output(&data, &size)` — collect output
7. Format and print JSON to stdout:
   ```json
   {
     "config_slug": "...",
     "output": { "result": [...] },
     "performance": {
       "metadata": { "rank": ..., "shape_kind": "...", "input_size": ..., "shape": [...] },
       "latency_ms": { "min": ..., "median": ..., "max": ..., "mean": ... },
       "runs": 10
     }
   }
   ```
8. Call `kernel_cleanup()`
9. Destroy stream, exit

### CUDA event timing (precision requirement)

```cpp
cudaEvent_t start_ev, end_ev;
cudaEventCreate(&start_ev);
cudaEventCreate(&end_ev);
cudaEventRecord(start_ev, stream);
kernel_run(stream);
cudaEventRecord(end_ev, stream);
cudaEventSynchronize(end_ev);
float ms;
cudaEventElapsedTime(&ms, start_ev, end_ev);
```

This matches kbEvalCli's `time_execution_with_cuda_event` pattern. GPU-side timing only — no host-side overhead contaminates the measurement.

### NCU compatibility

The harness binary is a standard CUDA executable. NCU profiles it as: `ncu ./binary`. All kernel launches (warmup + timed) are visible to NCU. No special mode needed — NCU aggregates launches of the same kernel automatically.

---

## 2. Reference-side contract (Python)

No changes to the reference contract. Documenting here for completeness since both sides are now formally specified.

### Required exports

```python
class Model(torch.nn.Module):
    def __init__(self, *init_inputs):
        ...
    def forward(self, *inputs) -> torch.Tensor:
        ...

def get_inputs(config: dict) -> list[torch.Tensor]:
    ...

def get_init_inputs() -> list[Any]:
    ...
```

### Rules

- `Model` must be a subclass of `torch.nn.Module`
- `get_inputs(config)` receives the config `params` dict and returns a list of tensors
- `get_init_inputs()` returns constructor arguments for `Model`
- The entry file must be named `reference.py`
- Reference execution is always in-process Python (loaded by evaluate.py/profile.py via importlib)

---

## 3. eval_support.py — shared Python utilities

### Functions extracted from evaluate.py

| Function | Purpose |
|---|---|
| `set_seed(seed)` | `torch.manual_seed` + `torch.cuda.manual_seed` |
| `acquire_device_lock(device)` | `fcntl.flock(LOCK_EX\|LOCK_NB)` per-GPU lock |
| `release_device_lock(lock_fd)` | Unlock and close fd |
| `gpu_cleanup(device)` | `empty_cache` + `reset_peak_memory_stats` + `synchronize` |
| `watchdog_handler(signum, frame)` | Raises `TimeoutError` |
| `load_reference_entry(reference_root)` | Find `reference.py` under root |
| `load_reference_module(reference_path)` | `importlib` dynamic load |
| `normalize_reference_contract(module)` | Validate Model/get_inputs/get_init_inputs |
| `extract_config_payload(env_json)` | Parse `CUDA_EXEC_CONFIG_JSON` → params dict |
| `tensor_to_jsonable(value)` | Tensor → nested list |
| `flatten_numeric(value)` | Nested list → flat float list |
| `infer_shape(value)` | Nested list → shape tuple |
| `allclose_check(ref, gen, atol, rtol)` | Tolerance check matching torch.allclose |
| `measure_reference(module, config, device, ...)` | Full reference measurement with CUDA event timing |

### Constants extracted

```python
DEFAULT_SEED = 42
NUM_CORRECTNESS_TRIALS = 3
NUM_WARMUP_RUNS = 5
NUM_PERF_TRIALS = 10
ATOL = 1e-02
RTOL = 1e-02
```

### Import changes

**evaluate.py**: imports from `eval_support` instead of defining locally. Keeps only: `_run_generated()`, `_verify_correctness()`, `_comparison_payload()`, `main()`.

**profile.py**: (removed — profile is now NCU-only, handled by `profile.sh` and inline NCU commands in `tasks.py`).

---

## 4. compile.sh changes

### New flag: `--harness`

```bash
compile.sh --source generated.cu --output binary --harness /path/to/eval_harness.cu
```

When `--harness` is provided:

- Steps 1-4 (PTX, ptxas, cuobjdump, nvdisasm) analyze `--source` only, with `-I` pointing to the harness header directory
- Step 5 (binary link): `nvcc eval_harness.cu generated.cu -I<harness_dir> -o binary`
- Step 6 (new): Symbol validation — verify the binary exports `kernel_setup`, `kernel_run`, `kernel_output`, `kernel_cleanup` via `nm`. Fail with clear error if any symbol is missing.

When `--harness` is NOT provided:

- Current behavior unchanged (monolithic single-file compile)

### Include path

`eval_harness.h` lives alongside `eval_harness.cu` in `cuda_exec/scripts/`. compile.sh derives the include path from the `--harness` argument: `dirname "$HARNESS"`.

---

## 5. File inventory

### New files

| File | Purpose |
|---|---|
| `cuda_exec/scripts/eval_harness.cu` | C++ harness main() |
| `cuda_exec/scripts/eval_harness.h` | RuntimeConfig struct + function declarations |
| `cuda_exec/scripts/eval_support.py` | Shared Python utilities |

### Modified files

| File | Change |
|---|---|
| `cuda_exec/scripts/evaluate.py` | Import from eval_support, remove extracted functions |
| `cuda_exec/scripts/profile.py` | Import from eval_support instead of evaluate |
| `cuda_exec/scripts/compile.sh` | Add `--harness` flag + symbol validation step |
| `cuda_exec/tasks.py` | Pass `--harness` path when invoking compile.sh |
| `cuda_exec/DESIGN.md` | Add kernel interface contracts section |
| `cuda_exec/CLAUDE.md` | Update module descriptions |
| `cuda_exec/tests/fixtures/generated/generated_runtime_launch.cu` | Convert to harness interface |

### Unchanged files

| File | Reason |
|---|---|
| `cuda_exec/tests/fixtures/generated/generated.cu` | Stays monolithic — fast test fixture with fake timing |
| `cuda_exec/tests/fixtures/reference/reference.py` | Reference contract unchanged |

---

## 6. Backward compatibility

- `--harness` flag is opt-in. Without it, compile.sh works exactly as before.
- `generated.cu` (fake timing fixture) stays monolithic for fast integration tests.
- `generated_runtime_launch.cu` converts to harness format as proof of the new interface.
- tasks.py passes `--harness` when the harness file exists; callers can still submit monolithic .cu files.

---

## 7. Input data generation

Both sides generate inputs independently from config parameters:

- Reference: `get_inputs(config)` in Python
- Generated: `kernel_setup(cfg)` in C++ using the same deterministic patterns

For the current vector-add fixture, both use `arange(0, input_size)`. This works because:
- Input generation is deterministic given config
- Correctness comparison checks output values, not inputs
- Both sides agree on the same input convention per kernel

Known limitation: for kernels with complex input patterns, the C++ and Python sides must independently implement matching input generation. Long-term, serializing inputs from Python to the binary via files would be more robust but is out of scope for this design.

---

## 8. JSON output contract

The harness binary outputs structured JSON to stdout. Two consumers parse this:

**evaluate.py** reads:
- `payload.get("output", {}).get("result", [])` — kernel output for correctness comparison
- `payload.get("performance", {}).get("latency_ms", {})` — timing for speedup calculation

**profile.py** reads:
- `payload.get("performance", {})` — timing summary
- `payload.get("output", {})` — optional output metadata

The harness must produce this exact structure. The `output.result` field is a flat list of floats (the harness always flattens). evaluate.py's `_flatten_numeric()` also flattens the reference output before comparison, so shape matching is done separately via `_infer_shape()` on the reference side. The harness reports `output_shape` in metadata for cross-checking.
