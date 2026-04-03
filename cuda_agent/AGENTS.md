# cuda_agent

This file provides guidance when working with code in this directory.

## What this is

`cuda_agent` is an MCP-based optimization agent that iteratively improves CUDA kernels by calling the `cuda_exec` remote execution service. It uses the Anthropic Agent SDK (`claude-agent-sdk`) to run a Claude-powered optimization loop: compile -> evaluate -> analyze -> modify code -> repeat.

## Commands

### Run the agent
```bash
cd /home/centos/kernel_lab
.venv/bin/python -m cuda_agent \
    --run-tag optim_001 \
    --version v1 \
    --direction-id 7 \
    --direction-slug vector-add \
    --reference-dir data/fixtures/sm120/vecadd/ \
    --generated-file data/generated/sm120/vecadd/generated.cu \
    --configs-file data/fixtures/sm120/vecadd/configs.json \
    --max-iterations 5
```

### Prerequisites
- `cuda_exec` service must be running at the target URL (default `http://127.0.0.1:8000`)
- `ANTHROPIC_API_KEY` must be set in the environment
- Bearer token key file at `~/.keys/cuda_exec.key` (or override via `CUDA_EXEC_KEY_PATH`)

### Test plugin MCP server standalone
```bash
cd /home/centos/kernel_lab
PYTHONPATH="$PWD" .venv/bin/python plugins/cuda/mcp_server.py
```

## Architecture

### Module responsibilities

- **`agent.py`** — Agent orchestration. Loads the `cuda` plugin MCP server for toolkit execution and runs the optimization loop. Documentation search uses `python -m doc_retrieval` CLI via Bash. Claude manages iteration internally.
- **`prompts.py`** — System prompt encoding workflow rules, convergence criteria, and CUDA optimization techniques. Initial prompt template formatting.
- **`task.py`** — `OptimizationTask` dataclass holding all inputs for an optimization run.
- **`cli.py`** — CLI argument parsing, file reading, task construction.
- **`__main__.py`** — Entry point for `python -m cuda_agent`.

### Plugin MCP server (under `plugins/`)

- **`plugins/cuda/mcp_server.py`** — 9 execution tools: `compile`, `evaluate`, `profile`, `execute`, `read_file`, `get_compile_data`, `get_evaluate_data`, `get_profile_data`, `get_data_point`

Documentation search is handled via `python -m doc_retrieval find/read/browse` CLI (no MCP server).

### Key design decisions

- **Loose coupling** — MCP servers make HTTP calls to cuda_exec. No imports from `cuda_exec` package.
- **Single long agent run** — Claude manages the optimization loop internally rather than an outer Python loop.
- **MCP for execution, CLI for docs** — Toolkit execution uses an MCP server; documentation search uses the `doc_retrieval` CLI via Bash.
- **Binary content filtering** — The cuda MCP server replaces base64-encoded payloads with placeholders to keep tool results within context limits.
- **Bearer token auth** — The cuda MCP server reads the same key file as cuda_exec (`~/.keys/cuda_exec.key`).
- **Local data store** — Every tool call saves its full raw request and response (before compaction) to `~/.cuda_agent/<run_tag>/<version>/<direction_id>_<direction_slug>/turn_<turn>/`.
