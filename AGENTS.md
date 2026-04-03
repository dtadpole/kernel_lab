# kernel_lab

A repository for kernel optimization experiments and related tooling.

## Current components

- `cuda_exec/` — FastAPI-based remote CUDA execution service
- `cuda_agent/` — MCP-based CUDA kernel optimization agent (uses cuda_exec via MCP tools)
- `doc_retrieval/` — NVIDIA CUDA Toolkit document retrieval system (BM25 + dense search)
- `plugins/` — Claude Code / Agent SDK plugins (MCP servers + Skills)
  - `plugins/kb/` — Knowledge search (doc retrieval MCP server)
  - `plugins/cuda/` — CUDA Toolkit execution service (compile/evaluate/profile MCP server)

## Repo-level conventions

### 1. Python environment

- All components share a single `uv`-managed virtual environment at `.venv`
- Dependencies are defined in the root `pyproject.toml`
- Setup: `uv venv .venv --python 3.12 && uv pip install -e "."`
- Plugins use `.venv/bin/python` to run their MCP servers

### 2. Metadata is mandatory

For command-style API requests/responses, `metadata` is required.
Required fields:

- `run_tag`
- `version`
- `direction_id`
- `direction_slug`
- `turn`

### 3. `cuda_exec` workflow is convention-driven

High-level rules:

- compile first
- compile once per turn
- evaluate/profile depend on compile state from the same turn
- evaluate/profile are config-level, not code-level
- new files require a new turn
- old turns are immutable

### 4. Settled `cuda_exec` interface conventions

These are stable project-level decisions and should remain in repo docs, not only in assistant memory.

#### Runtime mental model

- `workspace = inputs + scratch`
- `artifacts = kept results`
- `logs = process output`
- `state = workflow record`

#### Turn-root layout

`cuda_exec` resolves runtime locations by convention from request metadata. It does not accept a caller-specified working directory.

Runtime root:

```text
~/.cuda_exec/<run_tag>/<version>/<direction_id>_<direction_slug>/turn_<turn>/
```

Within each turn:

```text
turn_<turn>/
  workspace/
  artifacts/
  logs/
  state/
```

#### Compile inputs

`compile` takes inline file maps, not file lists:

- `reference_files: Dict[relative_path, content]`
- `generated_files: Dict[relative_path, content]`

All public request/response file names should use relative paths.

#### Stage model

- `compile` is code-level
- `evaluate` / `profile` are runtime-config-level
- one compile may fan out into many configs
- runtime configs are passed as `configs: Dict[config_slug, Dict[str, Any]]`
- the same config slug is the stable identity on both request and response
- the config body is intentionally kernel-specific and flexible; do not overfit it to one workload family
- reference Python code for `evaluate` / `profile` should follow the explicit module contract: `Model(torch.nn.Module)`, `get_init_inputs()`, `get_inputs(config)`
- `cuda_exec` intentionally borrows that kbEval-style reference contract even though the generated side still runs through the compiled artifact path rather than a Python `ModelNew` class

#### Public response boundary

Default public responses should stay small and only expose stage-relevant artifacts/logs.
Internal workflow state is kept for compile/evaluate/profile bookkeeping but is not part of the default public response.
Public request/response file names use relative paths, and public returned files are shaped as relative-path keyed dictionaries.
For evaluate/profile specifically, public responses mirror request shape and use `configs: Dict[config_slug, ...]` instead of result lists.
Top-level public responses use `all_ok` for aggregate success. Per-config outputs use `status` plus structured summaries instead of raw log-only results.

- evaluate config output: `status` + `reference` + `generated` + `correctness` + `performance` + `artifacts` + `logs`
- profile config output: `status` + `summary` + `artifacts` + `logs`
- profile uses Nsight Compute exclusively; callers specify `side: "generated" | "reference"`

#### Execute boundary

`execute` is logs-only from the public API perspective:

- keep `logs/execute.attempt_###.*`
- do not expose execute state in the public response
- let the caller decide what execute outputs matter

### 5. `cuda_exec` documentation split

- `cuda_exec/DESIGN.md` is the source of truth for detailed design
- `cuda_exec/README.md` stays short
- this `AGENTS.md` stays at repo-level only
- `cuda_exec/models.py` documents the public request/response contract
- `cuda_exec/runner.py` documents runtime-layout semantics
- `cuda_exec/main.py` stays thin and keeps only lightweight endpoint/helper docstrings

### 6. `cuda_exec/tests` is integration-only

- `cuda_exec/tests/` is reserved for end-to-end integration tests
- do not put unit tests there
- tests should start a real uvicorn service in a subprocess and call HTTP interfaces with realistic payloads
- tests should isolate runtime side effects via a temporary `CUDA_EXEC_ROOT`
- prefer placing temporary test roots under `~/temp/`
- create one top-level temp directory per integration-suite invocation rather than one sibling temp directory per test class or service lifecycle
- prefix the run directory name with `YYYY-MM-DD-HH-MM-`
- then use a kebab-case slug plus PID in the subfolder name
- if multiple service processes are started during the suite, reuse that same top-level run directory and namespace per-service logs/runtime roots inside it
- preferred isolation direction: provision the uvicorn Python environment from `cuda_exec/requirements.txt` using `uv`, with the environment itself created under a temporary folder for the test run
- when using a temp-folder uv-managed environment, prefer naming it `<temp-run-dir>/.venv`
- preserve run environments and intermediate outputs by default for later inspection; do not rely on immediate deletion after each run
- cleanup should happen via a separate retention process (for example pruning runs older than 7 days)
- the standard helper is `cuda_exec/scripts/prune_temp_runs.py`
- its default behavior is to delete preserved run directories older than 7 days; support `--dry-run`; and skip directories marked for keep
- integration test runs should invoke this helper before starting the temporary uvicorn service
- current tests may still use the repo-local `.venv`, but the temp-folder `uv`-managed `.venv` is the preferred future-tightening path
- expected lower-level CUDA failures are allowed during early integration coverage, as long as the interface behavior itself is exercised
- current integration config coverage should include roughly 4–6 configs spanning multiple 1D sizes plus representative 2D and 3D shape metadata
- prefer storing integration config sets in fixture files under `conf/fixtures/` instead of embedding them directly in the main test module
- fixture config slugs should make semantic sense for the sample workload; for vector-add fixtures, prefer size/shape/rank-based slugs rather than unrelated causal/noncausal labels
- for vector-add integration fixtures, the config body itself should stay pertinent: shape/rank/input_size metadata is enough, and unrelated transformer-style fields should be omitted

### 7. `cuda_agent` conventions

- `cuda_agent` does not import from `cuda_exec` — it communicates via HTTP through MCP servers
- `cuda_agent` requires `cuda_exec` to be running separately
- `cuda_agent` requires `ANTHROPIC_API_KEY` in the environment
- `cuda_agent` reads the bearer token from the same key file as `cuda_exec` (`~/.keys/cuda_exec.key` or `CUDA_EXEC_KEY_PATH`)
- `cuda_agent` loads two plugin MCP servers: `kb` (knowledge search) and `cuda` (toolkit execution)
- the agent (`agent.py`) uses `claude-agent-sdk` to run a single long optimization session
- the agent manages its own iteration loop internally — Claude decides when to compile, evaluate, modify, and converge

### 8. Plugins

- Plugins live in `plugins/` — each is a Claude Code plugin with `.claude-plugin/plugin.json`
- Each plugin can contain: MCP servers (`.mcp.json`), Skills (`skills/`), hooks, agents
- Plugins work in both Claude Code CLI (`--plugin-dir`) and Agent SDK (`mcp_servers={}`)
- MCP servers use the project's `.venv/bin/python` and `PYTHONPATH` set to repo root

## License

MIT — see `LICENSE`.
