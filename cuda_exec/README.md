# cuda_exec

FastAPI-based remote CUDA execution service.

## Purpose

`cuda_exec` is a convention-driven remote execution service for CUDA workflows.
The early version intentionally keeps the agent-facing API small:

- `compile`
- `evaluate`
- `profile`
- `execute`

The service, not the agent, owns most workflow conventions.

## V0 principles

- `compile` must happen first
- `compile` runs only once per turn
- `evaluate` and `profile` depend on compile state from the same turn
- new files require a new turn
- old turns are immutable
- returned files are fixed by convention, not chosen by the caller
- all stage stdout/stderr are persisted locally under the turn root

## Required metadata

All request payloads include:

- `run_tag`
- `version`
- `direction_id`
- `direction_slug`
- `turn`

Example:

```json
{
  "metadata": {
    "run_tag": "agent_a",
    "version": "v1",
    "direction_id": 7,
    "direction_slug": "convention",
    "turn": 3
  }
}
```

## Turn-root convention

```text
~/.cuda_exec/<run_tag>/<version>/<direction_id>_<direction_slug>/turn_<turn>/
```

On this machine:

```text
/home/centos/.cuda_exec/<run_tag>/<version>/<direction_id>_<direction_slug>/turn_<turn>/
```

Fixed subdirectories:

```text
workspace/
outputs/
logs/
profiles/
state/
tmp/
```

Execution cwd is:

```text
.../turn_<turn>/workspace/
```

## Endpoint summary

### `POST /compile`

Request fields:

- `metadata`
- `timeout_seconds`
- `original_files`
- `generated_files`

### `POST /evaluate`

Request fields:

- `metadata`
- `timeout_seconds`

### `POST /profile`

Request fields:

- `metadata`
- `timeout_seconds`

### `POST /execute`

Request fields:

- `metadata`
- `timeout_seconds`
- `command`
- `env`

## Response summary

All command-style endpoints return:

- `metadata`
- `ok`
- `kind`
- `command`
- `turn_root`
- `workspace_path`
- `returncode`
- `duration_seconds`
- `artifacts[]`
- `output.stdout`
- `output.stderr`
- `files[]`

## Stage-return convention

### Compile returns

- primary binary in `outputs/<stem>`
- `logs/compile.log`
- `logs/compile.stdout`
- `logs/compile.stderr`
- `state/compile.json`

### Evaluate returns

- `logs/evaluate.log`
- `logs/evaluate.stdout`
- `logs/evaluate.stderr`
- `state/evaluate.json`

### Profile returns

- `profiles/<stem>-ncu.ncu-rep`
- `logs/profile.log`
- `logs/profile.stdout`
- `logs/profile.stderr`
- `state/profile.json`

### Execute returns

- `logs/execute.log`
- `logs/execute.stdout`
- `logs/execute.stderr`

## Detailed design

See:

- `/home/centos/kernel_lab/cuda_exec/DESIGN.md`
