# kernel_lab

A repository for kernel optimization experiments and related tooling.

## Current component

- `cuda_exec/` — FastAPI-based remote CUDA execution service

## Repo layout

```text
kernel_lab/
  cuda_exec/
    __init__.py
    main.py
    models.py
    runner.py
    tasks.py
    scripts/
    requirements.txt
    README.md
  AGENTS.md
  LICENSE
  .gitignore
```

## Permanent project conventions

### 1. `cuda_exec` owns its own Python environment

- `cuda_exec` manages its own Python dependencies and its own `uv` environment.
- the repo root does **not** define a shared `uv` / `venv` environment for all future components.
- future agent-side environment management can live in a separate directory and be managed independently.

### 2. `cuda_exec` request metadata is mandatory

For command-style API requests/responses, `metadata` is required.
Within `metadata`, the required fields are:

- `run_tag`
- `version`
- `direction_id`
- `direction_slug`
- `turn`

### 3. `cuda_exec` does not accept an explicit working directory

`cuda_exec` resolves execution paths by convention from metadata instead of accepting a caller-specified working directory.

The root convention is:

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
tmp/
```

Current execution cwd is:

```text
.../turn_<turn>/workspace/
```

### 4. `cuda_exec` is convention-driven for compile/evaluate/profile

- `compile`, `evaluate`, and `profile` are hardened flows and should not expose free-form command/env configuration in the API.
- `execute` is the only command-style API that currently accepts a caller-provided command and environment variables.
- `profile` is currently hardened to NCU.
- `compile` stages original/generated files into the metadata-derived workspace and compiles exactly one `.cu` file with a fixed `nvcc` convention.

### 5. `cuda_exec` response structure is fixed

Command-style responses return:

- `metadata`
- command status fields (`ok`, `kind`, `command`, `workspace_path`, `returncode`, `duration_seconds`)
- `output.stdout`
- `output.stderr`
- `files[]` for requested returned files

## Owner

- d.t.p

## License

MIT — see `LICENSE`.
