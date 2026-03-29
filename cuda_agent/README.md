# cuda_agent

MCP-based CUDA kernel optimization agent. Uses the `cuda_exec` service via MCP tools to iteratively compile, evaluate, and improve CUDA kernels.

## Quick start

1. Ensure `cuda_exec` is running:
   ```bash
   cd /home/centos/kernel_lab
   cuda_exec/.venv/bin/python -m uvicorn cuda_exec.main:app --host 127.0.0.1 --port 8000
   ```

2. Set your API key:
   ```bash
   export ANTHROPIC_API_KEY=sk-...
   ```

3. Run the agent:
   ```bash
   cuda_agent/.venv/bin/python -m cuda_agent \
       --run-tag optim_001 \
       --version v1 \
       --direction-id 7 \
       --direction-slug vector-add \
       --reference-dir cuda_exec/tests/fixtures/reference/ \
       --generated-file cuda_exec/tests/fixtures/generated/vector_add_runtime_launch.cu \
       --configs-file cuda_exec/tests/fixtures/configs/vector_add_shapes.json
   ```

## Prerequisites

- `cuda_exec` service running and accessible
- `ANTHROPIC_API_KEY` environment variable set
- Bearer token key file at `~/.keys/cuda_exec.key`

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Anthropic API key |
| `CUDA_EXEC_URL` | `http://127.0.0.1:8000` | cuda_exec service URL |
| `CUDA_EXEC_KEY_PATH` | `~/.keys/cuda_exec.key` | Path to bearer token key file |

## MCP Tools Reference

The CUDA Toolkit MCP server (`mcp_server.py`) exposes 5 tools that map 1:1 to `cuda_exec` HTTP endpoints. All tools require a `metadata` dict with keys: `run_tag`, `version`, `direction_id`, `direction_slug`, `turn`.

### cuda_compile

Compile CUDA source files using the nvcc -> ptxas -> cuobjdump -> nvdisasm toolchain.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metadata` | `dict` | (required) | Turn identity |
| `reference_files` | `dict[str, str]` | (required) | `{relative_path: content}` — reference Python sources |
| `generated_files` | `dict[str, str]` | (required) | `{relative_path: content}` — exactly one `.cu` file |
| `timeout_seconds` | `int` | 180 | Max wall-clock seconds |

**Returns:** `{all_ok, attempt, artifacts: {binary, ptx, cubin, resource_usage, sass}, tool_outputs}`

**Workflow:** Must be called exactly once per turn, before `cuda_evaluate` or `cuda_profile`. New source code requires a new turn.

### cuda_evaluate

Evaluate a compiled kernel for correctness and performance against the reference.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metadata` | `dict` | (required) | Turn identity (must match compile turn) |
| `configs` | `dict[str, dict]` | (required) | `{config_slug: config_body}` — runtime configs |
| `timeout_seconds` | `int` | 180 | Max wall-clock seconds |

**Returns:** `{all_ok, configs: {slug: {status, correctness: {passed, max_abs_error, ...}, performance: {latency_ms: {min, median, max, mean}, ...}}}}`

**Workflow:** Requires successful `cuda_compile` on same turn. Large fields (reference, generated, artifacts, logs) are stripped by the MCP server; use `cuda_read_file` for full details.

### cuda_profile

Profile kernel latency with configurable mode and backend.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metadata` | `dict` | (required) | Turn identity (must match compile turn) |
| `configs` | `dict[str, dict]` | (required) | `{config_slug: config_body}` |
| `mode` | `str` | `"generated_only"` | `"generated_only"` / `"reference_only"` / `"dual"` |
| `profiler_backend` | `str` | `"comparison_runtime"` | `"comparison_runtime"` / `"ncu"` (ncu: generated_only only) |
| `timeout_seconds` | `int` | 180 | Max wall-clock seconds |

**Returns:** `{all_ok, configs: {slug: {status, summary, reference_summary, generated_summary}}}`

**Workflow:** Requires successful `cuda_compile` on same turn. Use `cuda_read_file` to fetch `.ncu-rep` files for detailed Nsight Compute reports.

### cuda_execute

Run an ad-hoc command on the remote CUDA service.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metadata` | `dict` | (required) | Turn identity |
| `command` | `list[str]` | (required) | Argv list, e.g. `["/usr/local/cuda/bin/nvcc", "--version"]` |
| `env` | `dict[str, str]` | `{}` | Extra environment variables |
| `timeout_seconds` | `int` | 180 | Max wall-clock seconds |

**Returns:** `{all_ok, attempt, logs: {relative_path: file_payload}}`

### cuda_read_file

Read a file from a turn's directory tree on demand.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metadata` | `dict` | (required) | Turn identity (identifies which turn) |
| `path` | `str` | (required) | Relative path starting with `artifacts/`, `logs/`, or `state/` |
| `max_bytes` | `int \| None` | `None` | Optional byte cap |

**Returns:** `{metadata, file: {path, inline, content, encoding, truncated}}`

**Example paths:**
- `artifacts/compile.attempt_001.vector_add.ptx`
- `logs/evaluate.attempt_001.config_size_1024.log`
- `state/compile.attempt_001.json`

## Logging

Tool invocation logs are written per-run to:

```
~/.cuda_agent/<run_tag>/<version>/<direction_id>_<direction_slug>/tool_use.jsonl
```

Each line is a JSON object with `timestamp`, `tool`, `input`, and `output` fields (truncated to 1000 chars).
