# cuda_exec

FastAPI-based remote CUDA execution service.

## What this directory contains

- service entrypoint: `main.py`
- request/response models: `models.py`
- runtime helpers: `runner.py`
- hardened task flows: `tasks.py`
- shell helpers: `scripts/`

## Documentation

`README.md` is intentionally short.

The detailed design, workflow rules, runtime layout, config model, and naming conventions live in:

- `/home/centos/kernel_lab/cuda_exec/DESIGN.md`

## Current design direction

- compile is code-level and runs once per turn
- evaluate/profile are config-level and may run many configs per compile
- the preferred agent mental model is:
  - `workspace = inputs + scratch`
  - `artifacts = kept results`
  - `logs = process output`
  - `state = workflow record`
