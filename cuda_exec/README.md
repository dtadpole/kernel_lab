# cuda_exec

FastAPI-based remote CUDA execution service.

## Purpose

`cuda_exec` is intended to be deployed as its own remote service. It manages its own
Python dependencies and its own `uv` environment independently from the rest of the
`kernel_lab` repo.

## Metadata contract (required)

All command-style request and response payloads must include a required `metadata` object.
Within `metadata`, all of the following fields are required:

- `run_tag`
- `version`
- `direction_id`
- `direction_slug`
- `turn`

Recommended shape:

```json
{
  "metadata": {
    "run_tag": "blackwell_agent_a",
    "version": "v1",
    "direction_id": 3,
    "direction_slug": "warp_specialized_async_pipeline",
    "turn": 12
  }
}
```

## Workspace convention (required)

The API does **not** accept an explicit working directory.

Instead, `cuda_exec` resolves a deterministic workspace from metadata using this convention:

```text
~/.code_exec/<run_tag>/<version>/<direction_id>_<direction_slug>/turn_<turn>/
```

On this machine, that means:

```text
/home/centos/.code_exec/<run_tag>/<version>/<direction_id>_<direction_slug>/turn_<turn>/
```

Within each turn directory, `cuda_exec` creates these fixed subdirectories:

```text
workspace/
outputs/
logs/
profiles/
state/
tmp/
```

All command execution currently happens with cwd set to:

```text
.../turn_<turn>/workspace/
```

Accordingly, command-style responses return:

- `workspace_path`

instead of a caller-specified `workdir`.

Relative artifact paths such as:

- `outputs/...`
- `logs/...`
- `profiles/...`
- `state/...`
- `tmp/...`

are interpreted relative to the turn root, not relative to `workspace/`.

## Simplified interface design

The interface is now intentionally asymmetric:

- `compile` accepts source inputs and produces shared turn state
- `evaluate` and `profile` default to consuming that shared compile state
- `execute` remains the only general-purpose command interface

This is meant to reduce redundant parameter passing for agent callers.

## Request model (current hardened direction)

### `CompileRequest`
Convention-driven. No free-form command and no per-request environment variables.

```json
{
  "metadata": {
    "run_tag": "blackwell_agent_a",
    "version": "v1",
    "direction_id": 3,
    "direction_slug": "warp_specialized_async_pipeline",
    "turn": 12
  },
  "timeout_seconds": 300,
  "original_files": ["/abs/path/baseline_kernel.cu"],
  "generated_files": ["/abs/path/candidate_kernel.cu"],
  "return_files": []
}
```

Current compile convention:
- stage `original_files` into `workspace/original/`
- stage `generated_files` into `workspace/generated/`
- choose exactly one `.cu` file, preferring `generated_files` over `original_files`
- invoke the hardened Bash script:
  - `/home/centos/kernel_lab/cuda_exec/scripts/compile.sh`
- the Bash script calls:

```text
/usr/local/cuda/bin/nvcc -arch=native -std=c++17 -O3 -lineinfo <absolute_source> -o <turn_root>/outputs/<stem>
```

Compile also writes shared turn state to:

- `state/compile.json`

That manifest includes the default artifact id:

- `compile:primary_binary`

### `EvaluateRequest`
Convention-driven. No free-form command and no per-request environment variables.

```json
{
  "metadata": {
    "run_tag": "blackwell_agent_a",
    "version": "v1",
    "direction_id": 3,
    "direction_slug": "warp_specialized_async_pipeline",
    "turn": 12
  },
  "timeout_seconds": 300,
  "target_artifact_id": null,
  "return_files": []
}
```

Current evaluate convention:
- if `target_artifact_id` is omitted, use `compile:primary_binary`
- resolve the target from `state/compile.json`
- execute exactly one target artifact
- evaluate remains Python-driven

### `ProfileRequest`
Convention-driven. Hardened to NCU in the current version.

```json
{
  "metadata": {
    "run_tag": "blackwell_agent_a",
    "version": "v1",
    "direction_id": 3,
    "direction_slug": "warp_specialized_async_pipeline",
    "turn": 12
  },
  "timeout_seconds": 1800,
  "target_artifact_id": null,
  "return_files": []
}
```

Current profile convention:
- if `target_artifact_id` is omitted, use `compile:primary_binary`
- resolve the target from `state/compile.json`
- invoke the hardened Bash script:
  - `/home/centos/kernel_lab/cuda_exec/scripts/profile.sh`
- the Bash script calls:

```text
/usr/local/cuda/bin/ncu --set default --target-processes all --force-overwrite --export <turn_root>/profiles/<stem>-ncu <absolute_target>
```

### `ExecuteRequest`
General CUDA Toolkit execution interface. This is the only request type that currently accepts free-form command/environment input.

```json
{
  "metadata": {
    "run_tag": "blackwell_agent_a",
    "version": "v1",
    "direction_id": 3,
    "direction_slug": "warp_specialized_async_pipeline",
    "turn": 12
  },
  "timeout_seconds": 300,
  "command": ["/usr/local/cuda/bin/ptxas", "--version"],
  "env": {},
  "return_files": []
}
```

Current execute rule:
- `command[0]` must point under `/usr/local/cuda/bin`

## Response contract (fixed structure)

Command-style responses contain three important structured sections:

1. `artifacts`
   - structured artifact refs that the agent can reuse
2. `output`
   - `stdout`
   - `stderr`
3. `files`
   - structured returned files with path, name, encoding, and content

If the caller wants additional files returned, it should specify them in `return_files`.

Current file capture behavior:
- text files are returned as UTF-8 when possible
- binary files are returned as Base64
- files larger than 1 MiB are truncated in the response and marked with `truncated: true`

### Example response shape

```json
{
  "metadata": {
    "run_tag": "blackwell_agent_a",
    "version": "v1",
    "direction_id": 3,
    "direction_slug": "warp_specialized_async_pipeline",
    "turn": 12
  },
  "ok": true,
  "kind": "compile",
  "command": ["/usr/bin/env", "bash", "/home/centos/kernel_lab/cuda_exec/scripts/compile.sh", "--source", "...", "--output", "..."],
  "workspace_path": "/home/centos/.code_exec/blackwell_agent_a/v1/3_warp_specialized_async_pipeline/turn_12/workspace",
  "returncode": 0,
  "duration_seconds": 0.123,
  "artifacts": [
    {
      "artifact_id": "compile:primary_binary",
      "kind": "binary",
      "path": "outputs/candidate_kernel",
      "description": "Default executable artifact produced by compile"
    }
  ],
  "output": {
    "stdout": "...",
    "stderr": "..."
  },
  "files": [
    {
      "path": "/home/centos/.code_exec/blackwell_agent_a/v1/3_warp_specialized_async_pipeline/turn_12/state/compile.json",
      "name": "compile.json",
      "exists": true,
      "size_bytes": 512,
      "encoding": "utf8",
      "truncated": false,
      "content": "{...}",
      "error": null
    }
  ]
}
```

## API surface (v0)

### `GET /healthz`
Simple health check.

### `POST /compile`
Runs the hardened compile flow and writes `state/compile.json`.

### `POST /evaluate`
Runs the hardened evaluate flow. Defaults to `compile:primary_binary` from `state/compile.json`.

### `POST /profile`
Runs the hardened profile flow. Defaults to `compile:primary_binary` from `state/compile.json`.

### `POST /execute`
Runs a caller-specified CUDA Toolkit command.

## CLI scripts

Under `cuda_exec/scripts/` there are now:

- `compile.sh`
- `evaluate.py`
- `profile.sh`

These scripts follow the same metadata-derived workspace convention as the API.
They do **not** accept an explicit working directory.

### Compile example

```bash
cd /home/centos/kernel_lab
source cuda_exec/.venv/bin/activate
bash cuda_exec/scripts/compile.sh \
  --source /abs/path/candidate_kernel.cu \
  --output /home/centos/.code_exec/blackwell_agent_a/v1/3_warp_specialized_async_pipeline/turn_12/outputs/candidate_kernel
```

### Evaluate example

```bash
python cuda_exec/scripts/evaluate.py \
  --run-tag blackwell_agent_a \
  --version v1 \
  --direction-id 3 \
  --direction-slug warp_specialized_async_pipeline \
  --turn 12
```

### Profile example

```bash
bash cuda_exec/scripts/profile.sh \
  --target /home/centos/.code_exec/blackwell_agent_a/v1/3_warp_specialized_async_pipeline/turn_12/outputs/candidate_kernel \
  --export-prefix /home/centos/.code_exec/blackwell_agent_a/v1/3_warp_specialized_async_pipeline/turn_12/profiles/candidate_kernel-ncu
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
