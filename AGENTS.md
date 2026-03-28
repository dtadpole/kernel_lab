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
    DESIGN.md
  AGENTS.md
  LICENSE
  .gitignore
```

## Permanent project conventions

### 1. `cuda_exec` owns its own Python environment

- `cuda_exec` manages its own Python dependencies and its own `uv` environment.
- the repo root does **not** define a shared Python environment for all future components.

### 2. `cuda_exec` request metadata is mandatory

For command-style API requests/responses, `metadata` is required.
Within `metadata`, the required fields are:

- `run_tag`
- `version`
- `direction_id`
- `direction_slug`
- `turn`

### 3. `cuda_exec` uses a fixed turn-root convention

The root convention is:

```text
~/.cuda_exec/<run_tag>/<version>/<direction_id>_<direction_slug>/turn_<turn>/
```

On this machine, that means:

```text
/home/centos/.cuda_exec/<run_tag>/<version>/<direction_id>_<direction_slug>/turn_<turn>/
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

Current execution cwd is:

```text
.../turn_<turn>/workspace/
```

### 4. `cuda_exec` workflow is convention-driven and immutable per turn

- `compile` must run first for a turn.
- `compile` runs exactly once per turn.
- `evaluate` and `profile` are only valid after compile state exists for that turn.
- if new files are uploaded after `compile`, that work must move to a new turn.
- old turns are immutable and should not be modified in place.
- `execute` remains the general-purpose CUDA-tool command endpoint for special cases only.

### 5. `cuda_exec` stage outputs are fixed by convention

The caller does not choose target artifacts or return-file sets in V0.

#### Compile returns

- primary binary in `outputs/<stem>`
- `logs/compile.log`
- `logs/compile.stdout`
- `logs/compile.stderr`
- `state/compile.json`

#### Evaluate returns

- `logs/evaluate.log`
- `logs/evaluate.stdout`
- `logs/evaluate.stderr`
- `state/evaluate.json`

#### Profile returns

- `profiles/<stem>-ncu.ncu-rep`
- `logs/profile.log`
- `logs/profile.stdout`
- `logs/profile.stderr`
- `state/profile.json`

#### Execute returns

- `logs/execute.log`
- `logs/execute.stdout`
- `logs/execute.stderr`

### 6. `cuda_exec` response structure is fixed

Command-style responses return:

- `metadata`
- command status fields (`ok`, `kind`, `command`, `turn_root`, `workspace_path`, `returncode`, `duration_seconds`)
- `artifacts[]` for structured output description
- `output.stdout`
- `output.stderr`
- `files[]` for convention-returned files

## Owner

- d.t.p

## License

MIT — see `LICENSE`.
