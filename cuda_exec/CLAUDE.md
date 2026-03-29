# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`cuda_exec` is a FastAPI service for remote CUDA kernel compilation, evaluation, and profiling. It compiles generated CUDA kernels, compares them against Python reference implementations, and profiles performance — all behind a structured HTTP API designed for agent consumption.

## Commands

### Run the service
```bash
cd /home/centos/kernel_lab
.venv/bin/python -m uvicorn cuda_exec.main:app --host 127.0.0.1 --port 8000
```
The venv lives at `cuda_exec/.venv` and has `fastapi` + `uvicorn` installed from `requirements.txt`.

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
- **`scripts/compile.sh`** — 5-step CUDA compile pipeline: PTX generation, ptxas assembly, resource usage dump, SASS dump via nvdisasm, and binary linking.
- **`scripts/evaluate.py`** — Comparison runner. Loads reference Python module (must export `Model(nn.Module)`, `get_inputs(config)`, `get_init_inputs()`), runs the generated compiled binary, compares outputs.
- **`scripts/profile.py`** — Profile runner supporting three modes (`generated_only`, `reference_only`, `dual`) under the `comparison_runtime` backend.
- **`scripts/profile.sh`** — Nsight Compute (`ncu`) capture wrapper, scoped to `generated_only` mode only.

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
- **`profiler_backend="ncu"`** is a parallel path to `comparison_runtime`, intentionally limited to `generated_only` mode.
- **Bearer token authentication** gates all endpoints except `/healthz`. Key file at `~/.keys/cuda_exec.key`, overridable via `CUDA_EXEC_KEY_PATH` env var. Service refuses to start without a valid key.

### DESIGN.md

`DESIGN.md` is the authoritative source of truth for conventions, JSON request/response examples, and detailed contract specifications. Consult it for response shapes, naming conventions, and the full attempt/config naming scheme.
