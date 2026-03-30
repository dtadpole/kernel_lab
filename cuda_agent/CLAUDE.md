# CLAUDE.md

This file provides guidance to Claude Code when working with code in this directory.

## What this is

`cuda_agent` is an MCP-based optimization agent that iteratively improves CUDA kernels by calling the `cuda_exec` remote execution service. It uses the Anthropic Agent SDK (`claude-agent-sdk`) to run a Claude-powered optimization loop: compile -> evaluate -> analyze -> modify code -> repeat.

## Commands

### Run the agent
```bash
cd /home/centos/kernel_lab
cuda_agent/.venv/bin/python -m cuda_agent \
    --run-tag optim_001 \
    --version v1 \
    --direction-id 7 \
    --direction-slug vector-add \
    --reference-dir conf/fixtures/vecadd/ \
    --generated-file conf/fixtures/vecadd/generated.cu \
    --configs-file conf/fixtures/vecadd/configs.json \
    --max-iterations 5
```

### Prerequisites
- `cuda_exec` service must be running at the target URL (default `http://127.0.0.1:8000`)
- `ANTHROPIC_API_KEY` must be set in the environment
- Bearer token key file at `~/.keys/cuda_exec.key` (or override via `CUDA_EXEC_KEY_PATH`)

### Run the MCP server standalone (for testing)
```bash
cd /home/centos/kernel_lab
cuda_agent/.venv/bin/python -m cuda_agent.mcp_server
```
The server uses stdio transport. It reads `CUDA_EXEC_URL` and `CUDA_EXEC_KEY_PATH` from environment.

## Architecture

### Module responsibilities

- **`mcp_server.py`** — FastMCP stdio server. Wraps 5 cuda_exec HTTP endpoints as action tools (`cuda_compile`, `cuda_evaluate`, `cuda_profile`, `cuda_execute`, `cuda_read_file`), 4 local data retrieval tools (`cuda_get_compile_data`, `cuda_get_evaluate_data`, `cuda_get_profile_data`, `cuda_get_data_point`), and 2 document search tools (`cuda_search_docs`, `cuda_lookup_doc_section`) for querying indexed NVIDIA CUDA Toolkit documentation. Strips base64 binary content from responses to avoid context bloat. Persists raw request/response for every tool call to a local data store. Handles bearer token auth.
- **`agent.py`** — Agent orchestration. Creates a `claude-agent-sdk` session with the MCP server and runs the optimization loop. Claude manages iteration internally.
- **`prompts.py`** — System prompt encoding workflow rules, convergence criteria, and CUDA optimization techniques. Initial prompt template formatting.
- **`task.py`** — `OptimizationTask` dataclass holding all inputs for an optimization run.
- **`cli.py`** — CLI argument parsing, file reading, task construction.
- **`__main__.py`** — Entry point for `python -m cuda_agent`.

### Key design decisions

- **Loose coupling** — The MCP server makes HTTP calls to cuda_exec. No imports from `cuda_exec` package.
- **Single long agent run** — Claude manages the optimization loop internally rather than an outer Python loop. More flexible; Claude can adapt strategy mid-run.
- **Binary content filtering** — The MCP server replaces base64-encoded payloads with placeholders to keep tool results within context limits.
- **Bearer token auth** — The MCP server reads the same key file as cuda_exec (`~/.keys/cuda_exec.key`).
- **Local data store** — Every tool call saves its full raw request and response (before compaction) to `~/.cuda_agent/<run_tag>/<version>/<direction_id>_<direction_slug>/turn_<turn>/`. The `cuda_get_data_point` tool reads from this store.
