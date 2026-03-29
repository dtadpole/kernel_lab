# cuda_exec

FastAPI-based remote CUDA execution service.

## What this directory contains

- service entrypoint: `main.py`
- request/response models: `models.py`
- runtime helpers: `runner.py`
- hardened task flows: `tasks.py`
- shell helpers: `scripts/`

## Design summary

- compile is code-level and runs once per turn
- evaluate/profile are config-level and may run many configs per compile
- compile inputs use inline file maps via `reference_files` and `generated_files`
- evaluate/profile configs use `Dict[config_slug, Dict[str, Any]]`
- config payloads are intentionally kernel-specific and flexible
- public responses return stage-relevant `artifacts` and `logs` as relative-path keyed dictionaries
- top-level public responses use `all_ok`
- evaluate/profile responses mirror request shape with `configs: Dict[config_slug, ...]`
- evaluate config outputs carry `status`, `reference`, `generated`, `correctness`, `performance`, `artifacts`, and `logs`
- profile requests support `mode` plus `profiler_backend`
- profile config outputs carry `status`, `summary`, `reference`, `generated`, `reference_summary`, `generated_summary`, `artifacts`, and `logs`
- internal state is kept for compile/evaluate/profile bookkeeping, but not exposed in default public responses
- the runtime mental model is:
  - `workspace = inputs + scratch`
  - `artifacts = kept results`
  - `logs = process output`
  - `state = workflow record`

## Detailed design

See:

- `/home/centos/kernel_lab/cuda_exec/DESIGN.md`

That file now contains the authoritative request/response JSON examples for:

- compile
- evaluate
- profile
- execute

Current profile note:

- `profiler_backend="comparison_runtime"` is the default behavior-first runtime
- `profiler_backend="ncu"` is available in parallel and is intentionally scoped to `mode="generated_only"`
- long-term design: `ncu` stays a generated-side capture backend; cross-side comparison remains under `comparison_runtime`
