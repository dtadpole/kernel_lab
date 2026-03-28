# kernel_lab

A Python library workspace for kernel optimization experiments and execution tooling.

## Current component

- `cuda_exec/` — CUDA execution service package

## Goals

This repository is intended to host tooling around:

- kernel compilation
- correctness evaluation
- benchmarking
- profiling
- CUDA Toolkit tool execution

## Repository layout

```text
kernel_lab/
  cuda_exec/
  README.md
  LICENSE
  pyproject.toml
  .gitignore
```

## Quick start

Create a virtual environment and install the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Owner

- d.t.p

## License

MIT — see `LICENSE`.
