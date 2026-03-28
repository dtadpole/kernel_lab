# kernel_lab

A Python library workspace for kernel optimization experiments and CUDA execution tooling.

## Current component

- `cuda_exec/` — FastAPI-based remote execution service package

## Service name

- chosen name: `cuda_exec`

Other viable names for the 4th "run any CUDA Toolkit binary" endpoint family were:
- `execute`
- `run_binary`
- `run_tool`
- `tool_exec`
- `invoke`

For now, the API uses **`/execute`** because it is the shortest and clearest.

## API surface (v0)

### `GET /healthz`
Simple health check.

### `POST /compile`
Run a compile command in a working directory.

Request shape:

```json
{
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
- this is intended for direct execution of CUDA Toolkit tools such as `ptxas`, `nvdisasm`, `cuobjdump`, `ncu`, etc.

Request shape:

```json
{
  "binary_path": "/usr/local/cuda/bin/ptxas",
  "args": ["--version"],
  "workdir": "/home/centos/kernel_lab",
  "env": {},
  "timeout_seconds": 60
}
```

### Response shape
All command-style endpoints currently return a shared response shape:

```json
{
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

## Local development

This repo uses `requirements.txt` and `uv`.

### Create environment

```bash
cd /home/centos/kernel_lab
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Run the service

```bash
cd /home/centos/kernel_lab
source .venv/bin/activate
uvicorn cuda_exec.main:app --host 0.0.0.0 --port 8000 --reload
```

## Files

```text
kernel_lab/
  cuda_exec/
    __init__.py
    main.py
    models.py
    runner.py
  README.md
  requirements.txt
  LICENSE
  .gitignore
```

## Owner

- d.t.p

## License

MIT — see `LICENSE`.
