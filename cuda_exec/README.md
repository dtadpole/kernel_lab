# cuda_exec

FastAPI-based remote CUDA execution service.

## Purpose

`cuda_exec` is intended to be deployed as its own remote service. It manages its own
Python dependencies and its own `uv` environment independently from the rest of the
`kernel_lab` repo.

## Metadata contract (required)

All command-style request and response payloads must include a required `metadata` object.
Within `metadata`, all of the following fields are required:

- `version`
- `direction_id`
- `direction_slug`
- `turn`

Recommended shape:

```json
{
  "metadata": {
    "version": "v1",
    "direction_id": 3,
    "direction_slug": "warp_specialized_async_pipeline",
    "turn": 12
  }
}
```

## API surface (v0)

### `GET /healthz`
Simple health check.

### `POST /compile`
Run a compile command in a working directory.

Request shape:

```json
{
  "metadata": {
    "version": "v1",
    "direction_id": 3,
    "direction_slug": "warp_specialized_async_pipeline",
    "turn": 12
  },
  "workdir": "/home/centos/kernel_lab",
  "command": ["bash", "-lc", "make -C /home/centos/cuda_example binary"],
  "env": {},
  "timeout_seconds": 300,
  "artifacts": []
}
```

### `POST /evaluate`
Run an evaluation command in a working directory.

Request shape:

```json
{
  "metadata": {
    "version": "v1",
    "direction_id": 3,
    "direction_slug": "warp_specialized_async_pipeline",
    "turn": 12
  },
  "workdir": "/home/centos/kernel_lab",
  "command": ["bash", "-lc", "python evaluator.py"],
  "env": {},
  "timeout_seconds": 300,
  "expected_outputs": []
}
```

### `POST /profile`
Run a profiler (`ncu` or `nsys`) against a target command.

Request shape:

```json
{
  "metadata": {
    "version": "v1",
    "direction_id": 3,
    "direction_slug": "warp_specialized_async_pipeline",
    "turn": 12
  },
  "profiler": "ncu",
  "workdir": "/home/centos/cuda_example",
  "target_command": ["./build/bin/vector_add_inline_ptx_profile"],
  "profiler_args": ["--set", "default", "--target-processes", "all"],
  "env": {},
  "timeout_seconds": 1800
}
```

### `POST /execute`
Run a CUDA Toolkit binary directly.

Current rule:
- `binary_path` must point under `/usr/local/cuda/bin`
- intended for direct execution of tools such as `ptxas`, `nvdisasm`, `cuobjdump`, `ncu`, etc.

Request shape:

```json
{
  "metadata": {
    "version": "v1",
    "direction_id": 3,
    "direction_slug": "warp_specialized_async_pipeline",
    "turn": 12
  },
  "binary_path": "/usr/local/cuda/bin/ptxas",
  "args": ["--version"],
  "workdir": "/home/centos/kernel_lab",
  "env": {},
  "timeout_seconds": 60
}
```

### Response shape
All command-style endpoints return the required `metadata` object back in the response:

```json
{
  "metadata": {
    "version": "v1",
    "direction_id": 3,
    "direction_slug": "warp_specialized_async_pipeline",
    "turn": 12
  },
  "ok": true,
  "kind": "compile",
  "command": ["bash", "-lc", "echo hi"],
  "workdir": "/tmp",
  "returncode": 0,
  "duration_seconds": 0.123,
  "stdout": "...",
  "stderr": "..."
}
```

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
