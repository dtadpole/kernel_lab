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
- compile inputs use `Dict[relative_path, content]`
- public responses return stage-relevant `artifacts` and `logs` as relative-path keyed dictionaries
- internal state is kept for compile/evaluate/profile bookkeeping, but not exposed in default public responses
- the runtime mental model is:
  - `workspace = inputs + scratch`
  - `artifacts = kept results`
  - `logs = process output`
  - `state = workflow record`

## Detailed design

See:

- `/home/centos/kernel_lab/cuda_exec/DESIGN.md`
