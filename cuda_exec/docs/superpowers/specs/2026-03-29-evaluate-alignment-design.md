# Evaluate Pipeline Alignment with kbEvalCli.py

## Context

`cuda_exec/scripts/evaluate.py` is the comparison runner for the evaluate stage. It currently diverges from `~/triton-ag/kbEvalCli.py` in several key areas: timing mechanism, correctness verification, device safety, and runtime parameters. This spec aligns evaluate.py with kbEvalCli.py across all six areas while preserving cuda_exec's subprocess architecture (compiled CUDA binary).

## Alignment Areas

### 1. Execution Mode — No change

- Reference: in-process via `importlib` (same concept as kbEvalCli's `load_model_and_inputs`)
- Generated: subprocess (compiled `.bin` binary) — this is cuda_exec's core value, not changing
- kbEvalCli loads both in-process; cuda_exec keeps the subprocess split

### 2. Configuration & Structure

- **Seed control**: Add `_set_seed(seed)` calling `torch.manual_seed(seed)` + `torch.cuda.manual_seed(seed)` before reference execution. Matches kbEvalCli's `set_seed()`.
- **torch.no_grad()**: Wrap all reference model execution in `torch.no_grad()` context. Matches kbEvalCli.
- **Explicit device placement**: Move `init_inputs`, `model`, and `inputs` to CUDA device explicitly via `.cuda(device=device)` pattern, matching kbEvalCli instead of relying on reference module internals.
- **get_inputs(config)**: Keep the config parameter (cuda_exec design decision). kbEvalCli uses parameterless `get_inputs()`.

### 3. Verification Logic

- **Shape check**: Infer tensor shape from nested list structure. Compare shapes before value comparison. Fail fast on mismatch, matching kbEvalCli's `output.shape != output_new.shape` check.
- **allclose tolerance**: Replace `abs(a-b) <= 1e-5` with `abs(a-b) <= atol + rtol * abs(expected)` where `atol=1e-02, rtol=1e-02`, matching `torch.allclose` parameters in kbEvalCli.
- **Multi-trial correctness**: Run reference N times (default `num_correctness_trials=3`) with seed rotation, matching kbEvalCli's `num_verify_trials=3`. Generated runs once (subprocess constraint). Each reference output is compared against the generated output.
- **Reporting**: `max_diff`, `avg_diff`, `output_shape`, `trials` (e.g. `"2/3"`), `total_trials`, `passed_trials` — matching kbEvalCli's `CorrectnessResult` fields.

### 4. Timing

- Replace `time.perf_counter()` with `torch.cuda.Event(enable_timing=True)`:
  ```python
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)
  start_event.record()
  output = model(*inputs)
  end_event.record()
  torch.cuda.synchronize(device=device)
  elapsed_ms = start_event.elapsed_time(end_event)
  ```
- Matches `time_execution_with_cuda_event()` in kbEvalUtil.py.

### 5. Run Parameters

- **Warmup**: 5 runs before timing measurement (matching `num_warmups=5`)
- **Timing trials**: 10 measurement runs (matching `num_perf_trials=10`)
- **Correctness trials**: 3 trials with seed rotation (matching `num_verify_trials=3`)
- **Timing stats**: `mean`, `std`, `min`, `max`, `num_trials` (adding `std`, matching `get_timing_stats`)
- **Device metadata**: `hardware` (`torch.cuda.get_device_name`), `device` (str) added to performance metadata

### 6. System Level

- **Device locking**: `fcntl.flock(LOCK_EX | LOCK_NB)` per GPU device, lock file at `~/.cuda_exec/.lock_cuda_{device_index}`. On lock failure, raise error (evaluate.py is single-shot subprocess, no retry loop). Matches kbEvalCli's `FileLock`.
- **Watchdog**: `signal.alarm(timeout)` with SIGALRM handler that raises `TimeoutError`. Simpler than kbEvalCli's `ArmableWatchdog` but effective since evaluate.py is already a subprocess managed by tasks.py.
- **Cleanup**: After evaluation (in `finally` block):
  ```python
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats(device=device)
  torch.cuda.synchronize(device=device)
  ```
  Matches `graceful_eval_cleanup()`.

## Code Changes

### evaluate.py — new/modified functions

| Function | Change |
|----------|--------|
| `_set_seed(seed)` | **New** — torch.manual_seed + torch.cuda.manual_seed |
| `_infer_shape(nested_list)` | **New** — infer tensor shape from nested list |
| `_allclose_check(ref, gen, atol, rtol)` | **New** — shape check + allclose on flattened values |
| `_acquire_device_lock(device)` | **New** — fcntl file lock per device |
| `_release_device_lock(lock_fd)` | **New** — release fcntl lock |
| `_gpu_cleanup(device)` | **New** — empty_cache + sync + reset stats |
| `_measure_reference(...)` | **Rewrite** — CUDA event timing, warmup, seed, no_grad, explicit .cuda(), std in stats |
| `_comparison_payload(...)` | **Rewrite** — shape check, allclose, multi-trial fields |
| `main()` | **Modify** — device lock, watchdog, cleanup, multi-trial |

### models.py

| Model | Change |
|-------|--------|
| `LatencySummary` | Add `std: float \| None = None` |
| `CorrectnessSummary` | Add `output_shape`, `trials`, `total_trials`, `passed_trials` |

### tests/test_e2e_service.py

- Update evaluate test assertions for new correctness and performance fields

### DESIGN.md / CLAUDE.md

- Document CUDA event timing, allclose tolerance, warmup/trial params, device locking

## What this does NOT include

- No error exception hierarchy (evaluate.py reports errors via JSON, not exceptions)
- No `measure_reference` only mode (profile endpoint covers this)
- No `torch.compile` reference option
- No multi-device selection (single-device assumption)
- No stale lock cleanup (short-lived subprocess)
