# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`cuda_exec` is a FastAPI service for remote CUDA kernel compilation, evaluation, and profiling. It compiles generated CUDA kernels, compares them against Python reference implementations, and profiles performance — all behind a structured HTTP API designed for agent consumption.

## Environment setup

The venv at `cuda_exec/.venv` is managed with **uv**:
```bash
cd /home/centos/kernel_lab/cuda_exec
uv venv .venv --python 3.12
uv pip install torch pydantic fastapi 'uvicorn[standard]' nvidia-cutlass-dsl psutil ninja
uv pip install 'flash-attn-4>=4.0.0b5' --no-build-isolation
```

## Commands

### Run the service
```bash
cd /home/centos/kernel_lab
cuda_exec/.venv/bin/python -m uvicorn cuda_exec.main:app --host 127.0.0.1 --port 8000
```

### Run integration tests
```bash
cd /home/centos/kernel_lab
python -m pytest cuda_exec/tests/test_e2e_service.py -v
```
Tests provision their own uv venv per suite run under `~/temp/`, start a real uvicorn process with `CUDA_EXEC_ROOT` redirected, hit HTTP endpoints, and preserve run directories for inspection. A CUDA GPU and the CUDA toolkit at `/usr/local/cuda` are required.

### Run a single test
```bash
python -m pytest cuda_exec/tests/test_e2e_service.py -v -k test_name
```

### Prune old test run directories
```bash
python cuda_exec/scripts/prune_temp_runs.py          # deletes runs > 7 days old
python cuda_exec/scripts/prune_temp_runs.py --dry-run # preview only
```

## Architecture

### Workflow pipeline

The service enforces a strict stage order per turn: **compile -> evaluate -> profile**. Compile runs once per turn. Evaluate and profile fan out over slug-keyed runtime configs against the compiled artifact.

### Module responsibilities

- **`main.py`** — Thin FastAPI router. Maps HTTP endpoints (`/compile`, `/evaluate`, `/profile`, `/execute`, `/files/read`, `/healthz`) to task functions. Transforms internal results into public response models.
- **`models.py`** — Pydantic request/response contracts. Documents the public API shape including `FilePayload`, `CompileArtifacts`, `EvaluateConfigOutput`, `ProfileConfigOutput`. This is the canonical place for API contract semantics.
- **`runner.py`** — Runtime layout and subprocess execution. Owns the four-directory turn structure (`workspace/`, `artifacts/`, `logs/`, `state/`), path resolution, file capture for responses, and the `CUDA_EXEC_ROOT` override mechanism.
- **`tasks.py`** — Orchestration logic for each stage. Enforces workflow rules (compile-once, compile-before-evaluate), writes manifests, invokes shell/Python scripts, and assembles structured results with correctness/performance summaries.
- **`scripts/compile.sh`** — 5-step CUDA compile pipeline: PTX generation, ptxas assembly, resource usage dump, SASS dump via nvdisasm, and binary linking. `--harness` flag links with `eval_harness.cu` and validates the `kernel_run` symbol.
- **`scripts/eval_harness.cu`** — BF16-only evaluation harness. Provides `main()`, env-based config, BF16 input generation, CUDA event timing, and JSON output. Kernel authors implement only `kernel_run` — no custom headers or structs required.
- **`scripts/eval_support.py`** — Shared Python utilities for evaluate and profile: device locking, watchdog, GPU cleanup, reference loading/measurement, correctness checking.
- **`scripts/evaluate.py`** — Comparison runner aligned with `kbEvalCli.py`. CUDA event timing, `allclose` verification (atol/rtol=1e-02), multi-trial correctness (3 trials, seed rotation), warmup=5, timing trials=10, per-GPU device lock, SIGALRM watchdog, GPU cleanup. Loads reference Python module (must export `Model(nn.Module)`, `get_inputs(config)`, `get_init_inputs()`), runs the generated compiled binary, compares outputs.
- **`scripts/fmt_eval.py`** — Formats evaluate.py JSON output as a compact terminal summary (correctness + latency + speedup).
- **`scripts/profile.sh`** — Nsight Compute (`ncu`) capture wrapper. Used by the `/profile` endpoint and Makefile `profile-ncu-generated` / `profile-ncu-reference` targets.

### Turn-root disk layout

```
~/.cuda_exec/<run_tag>/<version>/<direction_id>_<direction_slug>/turn_<turn>/
  workspace/inputs/{reference,generated}/
  artifacts/
  logs/
  state/
```
Override with `CUDA_EXEC_ROOT` env var for tests/isolation.

### Key design decisions

- **Compile is code-level; evaluate/profile are config-level.** One compile fans out to many configs.
- **Reference side is always Python** (`torch.nn.Module` contract); generated side is the compiled CUDA binary.
- **Config payloads are intentionally flexible** — the service owns transport shape, the kernel owns semantic shape.
- **Public responses use `all_ok`** at the top level and per-config `status` fields. Internal `state/` is not exposed in default responses.
- **Files in responses** are relative-path-keyed dicts of `FilePayload` with `encoding` (utf8/base64) and `truncated` metadata.
- **Profile is NCU-only.** The `/profile` endpoint runs Nsight Compute with `--set detailed`. Callers specify `side: "generated" | "reference"` to choose which kernel to profile.
- **Bearer token authentication** gates all endpoints except `/healthz`. Key file at `~/.keys/cuda_exec.key`, overridable via `CUDA_EXEC_KEY_PATH` env var. Service refuses to start without a valid key.
- **Fixed entry file names** — reference entry must be `cutedsl.py`, generated entry must be `generated.cu`. Additional helper files may use any name.
- **BF16-only kernel interface** — All inputs/outputs use `__nv_bfloat16` (CUDA) / `torch.bfloat16` (Python). Generated kernels export `extern "C" int kernel_run(__nv_bfloat16**, int, __nv_bfloat16**, int, int, cudaStream_t)`. No custom headers needed — only `#include <cuda_bf16.h>`.
- **Evaluate pipeline aligned with kbEvalCli.py** — CUDA event timing, `allclose` with atol/rtol=1e-02, multi-trial correctness (3 trials with seed rotation), warmup=5, timing trials=10. System-level: per-GPU `fcntl` device lock, `signal.alarm` watchdog, GPU cleanup in `finally`. Only intentional divergence: generated side runs as compiled binary subprocess (not in-process Python).
- **L2 cache flush before every timed trial** — The harness/support layer (NOT fixture files) flushes L2 cache before each timed trial using the Triton do_bench / NVBench pattern: allocate a buffer equal to L2 cache size, `memset`/`zero_()` it before each trial. This prevents warm-L2 artifacts from inflating performance numbers.
- **Measurement environment is the harness's responsibility** — Fixture files (`cutedsl.py`, `generated.cu`) implement only the kernel logic. The measurement environment (L2 flush, warmup count, trial count, timing, device locking) is set by the outer harness layer:
  - **Generated evaluate**: `eval_harness.cu` controls warmup, L2 flush, CUDA event timing
  - **Reference evaluate**: `eval_support.py` controls warmup, L2 flush, CUDA event timing
  - **Reference NCU profiling**: `profile_reference.py` wraps `cutedsl.py` with L2 flush + controlled warmup/trials
  - **Generated NCU profiling**: `eval_harness.cu` (same binary, NCU captures kernel invocations)
  - Fixture files must NOT set their own L2 flush, warmup counts, or timing. Their `main()` is for standalone smoke testing only.

### DESIGN.md

`DESIGN.md` is the authoritative source of truth for conventions, JSON request/response examples, and detailed contract specifications. Consult it for response shapes, naming conventions, and the full attempt/config naming scheme.

### Hardware roofline data

`docs/roofline/` contains per-GPU peak compute and memory bandwidth specs for roofline analysis. See `docs/roofline/README.md` for a quick comparison table covering A100, H100, B200, RTX 5090, and RTX PRO 6000 Blackwell. Always reference these when reporting kernel performance (TFLOPS achieved vs. peak, % of roofline ceiling).
