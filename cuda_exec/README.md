# cuda_exec

FastAPI-based remote CUDA execution service.

## Purpose

`cuda_exec` is intended to be deployed as its own remote service. It manages its own
Python dependencies and its own `uv` environment independently from the rest of the
`kernel_lab` repo.

## API surface (v0)

### `GET /healthz`
Simple health check.

### `POST /compile`
Run a compile command in a working directory.

### `POST /evaluate`
Run an evaluation command in a working directory.

### `POST /profile`
Run a profiler (`ncu` or `nsys`) against a target command.

### `POST /execute`
Run a CUDA Toolkit binary directly.

Current rule:
- `binary_path` must point under `/usr/local/cuda/bin`
- intended for direct execution of tools such as `ptxas`, `nvdisasm`, `cuobjdump`, `ncu`, etc.

## Deployment / local run

Create a dedicated environment for `cuda_exec`:

```bash
cd /home/centos/kernel_lab/cuda_exec
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Run the service from the repo root so `cuda_exec` imports resolve naturally:

```bash
cd /home/centos/kernel_lab
source cuda_exec/.venv/bin/activate
uvicorn cuda_exec.main:app --host 0.0.0.0 --port 8000 --reload
```
