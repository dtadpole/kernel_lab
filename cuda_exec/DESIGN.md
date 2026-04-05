# cuda_exec Design

This file is the source of truth for `cuda_exec` conventions.

The design goal is simple:

- keep agent inputs small
- keep runtime layout clean
- make workflow rules explicit
- separate scratch files from kept results
- support many runtime configs per compile

---

## 1. Core mental model

Use these four concepts:

1. **workspace = inputs + scratch**
2. **artifacts = kept results**
3. **logs = process output**
4. **state = workflow record**

### `workspace`

`workspace` is the working area for the current turn.

It contains:

- staged inputs
- scratch/intermediate files
- the initial cwd for launched processes

### `artifacts`

`artifacts` are files worth keeping.

Examples:

- compiled binaries
- profiler reports
- explicit result files worth preserving

### `logs`

`logs` store process output:

- combined logs
- stdout captures
- stderr captures

### `state`

`state` stores workflow records:

- stage manifests
- runtime config records
- per-config results
- references to kept artifacts

---

## 2. Turn-root layout

```text
~/.cuda_exec/<run_tag>/<version>/<direction_id>_<direction_slug>/turn_<turn>/
  workspace/
    inputs/
      reference/
      generated/
  artifacts/
  logs/
  state/
```

Override with `CUDA_EXEC_ROOT` env var for tests/isolation.

`turn_root` means the whole per-turn directory.
`workspace` means only the working directory under that turn root.

---

## 3. Workflow convention

Workflow order:

1. `compile`
2. `trial` (optional)
3. `profile` (optional)
4. `execute` for special/tooling cases only

Rules:

- `compile` must run first for a turn
- `compile` may run only once per turn
- do not reuse the same turn to upload a different file set for compile
- if new reference/generated inputs arrive after compile, start a new turn
- `trial` and `profile` require compile state from the same turn
- old turns are immutable

---

## 4. Compile input convention

`compile` takes inline file maps:

- `reference_files: Dict[str, str]`
- `generated_files: Dict[str, str]`

Rules:

- `reference_files` must be non-empty
- `generated_files` must be non-empty
- keys must be relative paths (no absolute paths, no `.`/`..` traversal)
- values are file contents
- `generated_files` must contain exactly one `.cu` file keyed as `generated.cu`
- `generated_files` may also include headers or helper files
- `reference_files` must include `cutedsl.py` (the entry point)

The toolkit writes these inputs under:

```text
workspace/inputs/reference/<relative_path>
workspace/inputs/generated/<relative_path>
```

---

## 5. Code-level compile vs config-level trial/profile

### Compile is code-level

`compile` builds the code artifact for the turn.

- compile happens once per turn
- compile should not vary across runtime configs
- code should be general enough to support all configs for that turn

### Trial and profile are config-level

`trial` and `profile` run the compiled artifact against one or more runtime configs.

That means one compile can fan out into many configs.

For `trial`, the runtime shape is comparison-first:

- load exactly one reference Python module from `workspace/inputs/reference/`
- require the module contract `Model`, `get_inputs(config)`, and `get_init_inputs()`
- run the reference side through that module contract
- run the generated side through the compiled primary artifact
- return per-config `reference`, `generated`, `correctness`, `performance`, `artifacts`, and `logs`

For `profile`, Nsight Compute (`ncu`) is used exclusively:

- callers specify `side: "generated" | "reference"` to choose which kernel to profile
- `generated`: NCU profiles the compiled CUDA binary via `profile.sh`
- `reference`: NCU profiles the reference Python/CuTe DSL kernel, filtering by kernel name regex to skip PyTorch JIT overhead
- return per-config `summary`, `artifacts`, and `logs`

---

## 6. Trial contract alignment notes (`cuda_exec` vs kbEval)

`cuda_exec/scripts/trial.py` is aligned with `kbEvalCli.py` across timing, verification, run parameters, and system-level behavior. The one intentional divergence is execution mode: `cuda_exec` runs the generated side as a compiled CUDA binary subprocess, while kbEvalCli runs both sides in-process.

### Aligned areas

**Timing** — CUDA event timing via `torch.cuda.Event(enable_timing=True)` with `start_event.record()` / `end_event.record()` / `torch.cuda.synchronize()` / `elapsed_time()`.

**Verification** — `allclose` with `atol=1e-02`, `rtol=1e-02`. Multi-trial correctness with 3 trials and deterministic seed rotation.

**Run parameters** — warmup=5, timing trials=10, correctness trials=3. Latency stats: `mean`, `std`, `min`, `max`, `median`.

**System level** — `fcntl.flock` per-GPU device lock, `signal.alarm` watchdog, GPU cleanup in `finally` block.

### Remaining intentional divergences

1. **Generated-side interface** — kbEvalCli expects `ModelNew(nn.Module)` in Python; `cuda_exec` runs the generated side as a compiled CUDA binary subprocess.

2. **`get_inputs` signature** — kbEval uses `get_inputs()` with no config argument; `cuda_exec` uses `get_inputs(config)` for config-driven evaluation.

---

## 6a. CuTe DSL reference benchmarking methodology

CuTe DSL (`cutlass.cute`) reference kernels require specific practices to get accurate timing. Naive usage produces misleading results because host-side overhead is captured in GPU event measurements.

### Problem: per-call overhead inflates event timing

`@cute.jit` functions re-generate their MLIR program on every call to verify the kernel content matches the cache. CUDA events record GPU timestamps — if the GPU sits idle waiting for the host to submit the kernel launch, the elapsed time includes that idle gap.

### Solution: `cute.compile()` + explicit stream

1. **Pre-compile with `cute.compile()`** — Returns a fixed `JitExecutor` that skips MLIR re-verification.
2. **Pass `cuda_driver.CUstream` explicitly** — Ensures the kernel runs on the same CUDA stream as the PyTorch timing events.

### Reference implementation pattern

```python
from cuda.bindings import driver as cuda_driver
import cutlass.cute as cute

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._compiled = None

        @cute.kernel
        def my_kernel(x_ptr, out_ptr, n):
            ...  # device code

        self.kernel = my_kernel

        @cute.jit
        def launch(x_ptr, out_ptr, n, blocks, threads, stream: cuda_driver.CUstream):
            self.kernel(x_ptr, out_ptr, n).launch(
                grid=[blocks, 1, 1],
                block=[threads, 1, 1],
                stream=stream,
            )

        self._jit_fn = launch

    def _ensure_compiled(self, *example_args):
        if self._compiled is not None:
            return
        fake_stream = cute.runtime.make_fake_stream()
        self._compiled = cute.compile(self._jit_fn, *example_args, fake_stream)

    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        blocks = ...
        self._ensure_compiled(x, out, n, blocks, self.threads)
        stream = cuda_driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        self._compiled(x, out, n, blocks, self.threads, stream)
        return out
```

### Do NOT put `torch.cuda.synchronize()` inside `forward()`

Reference `forward()` must not synchronize internally. Synchronization is the responsibility of the external timing harness.

---

## 7. Runtime config convention

`trial` and `profile` accept slug-keyed config maps:

- `configs: Dict[config_slug, Dict[str, Any]]`

The config slug is the stable identity for a config across both input and output.
The config body is intentionally flexible and kernel-specific.

For each config, the toolkit writes a config record under `state/configs/` and exports runtime
information through environment variables:

- `CUDA_EXEC_CONFIG_ID` (set to the config slug)
- `CUDA_EXEC_CONFIG_PATH`
- `CUDA_EXEC_CONFIG_JSON`
- `CUDA_EXEC_PARAM_<KEY>` for top-level config values

---

## 8. Attempt convention

Stage outputs use uniform attempt naming:

```text
state/compile.attempt_001.json
logs/compile.attempt_001.log

state/trial.attempt_001.json
state/profile.attempt_001.json
```

Config-specific outputs carry both attempt and config identity:

```text
logs/trial.attempt_001.config_<slug>.log
artifacts/trial.attempt_001.config_<slug>.comparison.json
artifacts/profile.attempt_001.config_<slug>.ncu-rep
```

Even though compile runs only once per turn, it still uses `attempt_001` for naming uniformity.

---

## 9. Stage outputs

### Compile

Kept results in `artifacts/`: binary, PTX, CUBIN, resource-usage report, SASS dump.

Process output in `logs/`: per-tool stdout/stderr (nvcc-ptx, ptxas, resource-usage, nvdisasm).

Workflow record: `state/compile.attempt_001.json`

### Trial

Per-config: logs under `logs/`, comparison artifacts under `artifacts/`.

Workflow record: `state/trial.attempt_###.json`

### Profile

Per-config: logs under `logs/`, NCU report under `artifacts/`.

Workflow record: `state/profile.attempt_###.json`

### Execute

Process output: `logs/execute.attempt_###.{log,stdout,stderr}`

`execute` does **not** write a stage state file. It is a tool-style execution path, not a workflow-record stage.

---

## 10. Response convention

Responses should stay stage-specific and minimal. `state` is internal — not part of default responses.

Rules:

- use relative paths only, never absolute
- files returned as dicts keyed by relative path with `content`, `encoding` (utf8/base64), `truncated`
- do not expose internal `state` paths, full artifact catalogs, or internal bookkeeping objects

Response shapes are defined in `models.py` (the canonical source). Key models:

- `CompileResponse` — `metadata`, `all_ok`, `attempt`, `artifacts`, `tool_outputs`
- `TrialResponse` — `metadata`, `all_ok`, `attempt`, `configs: Dict[slug, TrialConfigOutput]`
- `ProfileResponse` — `metadata`, `all_ok`, `attempt`, `configs: Dict[slug, ProfileConfigOutput]`
- `ExecuteResponse` — `metadata`, `all_ok`, `attempt`, `logs`

---

## 11. CWD convention

The toolkit guarantees that the **initial cwd** for launched processes is `<turn_root>/workspace/`.

The toolkit does not guarantee that an invoked program will remain inside that directory if the program itself changes cwd or writes to absolute paths.

### Test isolation

Runtime root can be redirected with `CUDA_EXEC_ROOT=<temporary-directory>`.

Retention helper: `cuda_exec/scripts/prune_temp_runs.py` deletes preserved run directories older than 7 days. Directories containing `keep` in name or a `KEEP` marker file are skipped.

---

## 12. Reference Python contract

A reference Python file must export:

- `class Model(torch.nn.Module)`
- `get_inputs(config: dict) -> list`
- `get_init_inputs(config: dict) -> list`

`cuda_exec` owns the evaluator/profiler runtime. Reference files only need to expose a runnable module interface.

---

## 13. Kernel interface contract (BF16-only)

All generated CUDA kernels and Python reference implementations use **BF16 (`__nv_bfloat16` / `torch.bfloat16`)** exclusively.  There is no float32 path.

### Generated side (CUDA)

Kernel authors implement a single function with this exact signature:

```cpp
#include <cuda_bf16.h>

extern "C" int kernel_run(__nv_bfloat16** inputs, int num_inputs,
                          __nv_bfloat16** outputs, int num_outputs,
                          int n, cudaStream_t stream);
```

- `inputs` / `outputs`: arrays of device pointers to BF16 buffers, pre-allocated by the harness.
- `n`: number of elements per buffer (= `input_size` from config).
- `stream`: CUDA stream to launch on.  Must not synchronize.
- Return `0` on success, non-zero on failure.

The eval harness (`eval_harness.cu`) provides `main()`, env-based config parsing, deterministic BF16 input generation, CUDA event timing with warmup, and structured JSON output.  It links with the kernel file via `compile.sh --harness`.

### Reference side (Python)

See Section 12 (Reference Python contract).

### Harness detection

`compile.sh` and `tasks.py` auto-detect harness mode by checking whether the source file contains `kernel_run`.

### Measurement environment contract

**The harness/support layer owns the measurement environment.  Fixture files must not.**

Fixture files (`cutedsl.py`, `generated.cu`) implement only kernel logic:
- `kernel_run()` — launch kernel(s) on the given stream, return immediately.
- `Model.forward()` — run the kernel, return output tensor.  Must not synchronize.

Harness responsibilities (never fixture responsibilities):
- L2 cache flush before each timed trial
- Warmup count and iteration control
- CUDA event timing (start/end events, synchronization)
- Trial count and statistical aggregation
- Device locking and GPU cleanup

| Path | Role | L2 flush | Warmup | Timing |
|------|------|:---:|:---:|:---:|
| `eval_harness.cu` | Generated trial + NCU profile | Yes | Yes | CUDA events |
| `eval_support.py` | Reference trial | Yes | Yes | CUDA events |
| `profile_reference.py` | Reference NCU profile | Yes | Yes | — (NCU captures) |
