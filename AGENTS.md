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

### 4. `cuda_exec` documentation split

- `cuda_exec/DESIGN.md` is the source of truth for detailed design
- `cuda_exec/README.md` stays short
- this `AGENTS.md` stays at repo-level only

## Owner

- d.t.p

## License

MIT — see `LICENSE`.
