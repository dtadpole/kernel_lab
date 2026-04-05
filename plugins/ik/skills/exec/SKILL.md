---
name: exec
description: Compile, trial, and profile CUDA kernels
user-invocable: true
argument-hint: <action> [kernel=K] [impl=<slug>] [gpu=N]
---

# CUDA Kernel Execution

Compile, trial, and profile CUDA kernels via Hydra CLI.

## GPU Selection

When the user specifies `gpu=N`, pass `exec.gpu=N` on the command.
If no GPU is specified, check `CLAUDE.md` or `AGENTS.md` for the assigned GPU indices for the current host.

**GPU is session-sticky**: once a GPU index is set by ANY ik skill (ik:exec, ik:bench, ik:optimize), ALL subsequent ik skill invocations in the same session MUST use that same GPU — unless the user explicitly provides a new `gpu=` value to override it.

## Implementation Slugs

Implementations are discovered dynamically. A **slug** has the format `{source}-{name}`:

| Source | Directory | Example |
|--------|-----------|---------|
| `ref` | `data/ref/{kernel}/` | `ref-cublas` |
| `gen` | `data/gen/{arch}/{kernel}/` | `gen-cuda` |

Use `list_impls(kernel, arch)` from `cuda_exec/impls.py` to discover all available slugs.

## Actions

All commands run from the project root:

```bash
cd /home/zhenc/kernel_lab
```

### Compile

```bash
.venv/bin/python -m cuda_exec.exec_cli exec.action=compile exec.kernel=matmul exec.arch=sm90 exec.impl=gen-cuda exec.gpu=4 exec.run_tag=optim_001
```

Returns: `all_ok`, `artifacts` (ptx, sass, resource_usage), `tool_outputs` (nvcc/ptxas stderr).

The compile step automatically:
- Resolves the impl slug to source files via `resolve_impl()`
- Finds the primary `ref-*` impl as the reference baseline
- Includes `.py` gen impls as additional reference files

### Trial

```bash
# Trial ALL configs (same run_tag as compile)
.venv/bin/python -m cuda_exec.exec_cli exec.action=trial exec.kernel=matmul exec.arch=sm90 exec.impl=gen-cuda exec.gpu=4 exec.run_tag=optim_001

# Trial specific configs
.venv/bin/python -m cuda_exec.exec_cli exec.action=trial exec.kernel=matmul exec.arch=sm90 exec.impl=gen-cuda exec.gpu=4 exec.run_tag=optim_001 'exec.configs=[mat-256x256,mat-8192x8192]'
```

Returns: `all_ok`, `configs` with per-config `status`, `correctness`, `performance`.

### Profile

```bash
# Profile the generated side (same run_tag as compile)
.venv/bin/python -m cuda_exec.exec_cli exec.action=profile exec.kernel=matmul exec.arch=sm90 exec.impl=gen-cuda exec.gpu=4 exec.run_tag=optim_001 'exec.configs=[mat-8192x8192]' exec.side=generated

# Profile the reference side
.venv/bin/python -m cuda_exec.exec_cli exec.action=profile exec.kernel=matmul exec.arch=sm90 exec.impl=gen-cuda exec.gpu=4 exec.run_tag=optim_001 'exec.configs=[mat-8192x8192]' exec.side=reference
```

Returns: `all_ok`, `configs` with per-config `status`, `summary` (NCU metrics).

### Turn management

```bash
# Turn 1: first compile
.venv/bin/python -m cuda_exec.exec_cli exec.action=compile exec.kernel=matmul exec.arch=sm90 exec.impl=gen-cuda exec.run_tag=optim_001 exec.turn=1

# After modifying source code, increment turn
.venv/bin/python -m cuda_exec.exec_cli exec.action=compile exec.kernel=matmul exec.arch=sm90 exec.impl=gen-cuda exec.run_tag=optim_001 exec.turn=2
```

## Workflow

1. **Compile** once per turn — resolves impl slug, loads ref + gen files, compiles
2. **Trial** against selected configs — check correctness and latency
3. **Profile** selectively (1-2 configs) — NCU hardware metrics

## Rules

- Compile exactly once per turn before trial or profile
- New source code requires a new turn (increment `exec.turn`)
- Old turns are immutable — never recompile on a previous turn number
- One compile fans out to many trial/profile calls with different configs

## Hydra Config

All settings in `conf/exec/default.yaml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `action` | required | `compile`, `trial`, or `profile` |
| `kernel` | required | Kernel name (matmul, fa4, vecadd) |
| `arch` | required | GPU architecture (sm90, sm120) |
| `impl` | required | Impl slug (e.g. `gen-cuda`, `ref-cublas`) |
| `gpu` | `null` | GPU index (sets CUDA_VISIBLE_DEVICES; null = use env) |
| `turn` | `1` | Turn number (increment for new source code) |
| `run_tag` | required | Workspace isolation tag (must be same across compile→trial→profile) |
| `configs` | `all` | "all" or list of config slugs |
| `side` | `generated` | Profile only: `generated`, `reference`, or `cudnn` |
| `timeout` | `300` | Per-action timeout in seconds |
| `data_root` | `null` | Source data root (null = project data/) |
