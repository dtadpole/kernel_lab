# kernel_lab

A repository for kernel optimization experiments and related tooling.

## Current component

- `cuda_exec/` — FastAPI-based remote CUDA execution service

## Repo-level conventions

### 1. Python environment ownership

- `cuda_exec` manages its own Python dependencies and its own `uv` environment.
- the repo root does **not** define a shared Python environment for all future components.

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

- `original_files: Dict[relative_path, content]`
- `generated_files: Dict[relative_path, content]`

All public request/response file names should use relative paths.

#### Stage model

- `compile` is code-level
- `evaluate` / `profile` are runtime-config-level
- one compile may fan out into many configs
- runtime configs are passed as `configs: Dict[config_slug, ConfigSpec]`
- the same config slug is the stable identity on both request and response

#### Public response boundary

Default public responses should stay small and only expose stage-relevant artifacts/logs.
Internal workflow state is kept for compile/evaluate/profile bookkeeping but is not part of the default public response.
Public request/response file names use relative paths, and public returned files are shaped as relative-path keyed dictionaries.
For evaluate/profile specifically, public responses mirror request shape and use `configs: Dict[config_slug, ...]` instead of result lists.

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

## Owner

- d.t.p

## License

MIT — see `LICENSE`.
