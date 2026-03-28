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

## Response contract (fixed structure)

Command-style responses now contain two structured sections:

1. `output`
   - `stdout`
   - `stderr`

2. `files`
   - structured returned files with path, name, encoding, and content

If the caller wants files returned, it should specify them in `return_files`.
For `compile`, `artifacts` are also collected and returned as files.

Current file capture behavior:
- text files are returned as UTF-8 when possible
- binary files are returned as Base64
- files larger than 1 MiB are truncated in the response and marked with `truncated: true`

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
  "artifacts": ["build/output.ptx"],
  "return_files": ["build/logs/compile.log"]
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
  "expected_outputs": [],
  "return_files": ["outputs/result.json"]
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
  "timeout_seconds": 1800,
  "return_files": ["/tmp/vector_add_inline_ptx-ncu.ncu-rep"]
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
  "timeout_seconds": 60,
  "return_files": []
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
  "output": {
    "stdout": "...",
    "stderr": "..."
  },
  "files": [
    {
      "path": "/tmp/result.txt",
      "name": "result.txt",
      "exists": true,
      "size_bytes": 42,
      "encoding": "utf8",
      "truncated": false,
      "content": "hello\n",
      "error": null
    }
  ]
}
```

## CLI scripts

Under `cuda_exec/scripts/` there are now three command-line helpers:

- `compile.py`
- `evaluate.py`
- `profile.py`

These scripts are meant to show the exact underlying CUDA Toolkit invocation and can also execute it.

### Compile examples

Show the exact `nvcc` command without executing it:

```bash
cd /home/centos/kernel_lab
source cuda_exec/.venv/bin/activate
python cuda_exec/scripts/compile.py nvcc \
  --workdir /home/centos/cuda_example \
  --source vector_add_inline_ptx.cu \
  --output /tmp/vector_add_inline_ptx.ptx \
  --mode ptx \
  --arch native \
  --dry-run-only
```

Show the exact `ptxas` command without executing it:

```bash
python cuda_exec/scripts/compile.py ptxas \
  --workdir /home/centos/cuda_example \
  --input build/intermediate/vector_add_inline_ptx.ptx \
  --output /tmp/vector_add_inline_ptx.cubin \
  --arch sm_120 \
  --verbose \
  --dry-run-only
```

### Evaluate example

```bash
python cuda_exec/scripts/evaluate.py \
  --workdir /home/centos/cuda_example \
  -- bash -lc './build/bin/vector_add_inline_ptx --bench --warmup 10 --iterations 20'
```

### Profile examples

Show the exact `ncu` command without executing it:

```bash
python cuda_exec/scripts/profile.py ncu \
  --workdir /home/centos/cuda_example \
  --set-name default \
  --target-processes all \
  --export /tmp/vector_add_inline_ptx-ncu \
  --dry-run-only \
  -- ./build/bin/vector_add_inline_ptx_profile
```

Show the exact `nsys` command without executing it:

```bash
python cuda_exec/scripts/profile.py nsys \
  --workdir /home/centos/cuda_example \
  --trace cuda,nvtx,osrt \
  --output /tmp/vector_add_inline_ptx-nsys \
  --force-overwrite \
  --dry-run-only \
  -- ./build/bin/vector_add_inline_ptx_profile
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
