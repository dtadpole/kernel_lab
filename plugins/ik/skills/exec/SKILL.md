---
name: exec
description: Compile, trial, and profile CUDA kernels
user-invocable: true
argument-hint: <action> [kernel=K] [impl=<slug>] [gpu=N]
---

# CUDA Kernel Execution

Compile, trial, and profile CUDA kernels via Hydra CLI. For iterative
development — use `/ik:bench` for formal assessment.

## Arguments

| Arg | Required | Default | Description |
|-----|----------|---------|-------------|
| `action` | **yes** | — | `compile`, `trial`, or `profile` |
| `kernel` | **yes** | — | Kernel name: `fa4`, `matmul`, etc. |
| `impl` | **yes** | — | Impl slug: `gen-cuda`, `ref-pytorch`, etc. |
| `arch` | no | auto | GPU arch: `sm90`, `sm120` |
| `gpu` | no | from host config | GPU index (`CUDA_VISIBLE_DEVICES`) |
| `run_tag` | no | auto | Workspace isolation tag (same across compile→trial→profile) |
| `revision` | no | `1` | Revision number (increment for new source code) |
| `configs` | no | `all` | Config slugs to trial/profile, or `all` |
| `side` | no | `generated` | Profile only: `generated` or `reference` |

GPU is **session-sticky**: once set, all subsequent ik invocations reuse it.

Impl slugs are discovered via `list_impls()` — see `artifacts/slug-resolution.md`.

## Usage

Use `exec.arch=auto` to auto-detect the GPU architecture. Do NOT hardcode sm90/sm100.

```bash
cd /home/zhenc/kernel_lab

# Compile
.venv/bin/python -m cuda_exec.exec_cli exec.action=compile \
  exec.kernel=matmul exec.arch=auto exec.impl=gen-cuda exec.gpu=4

# Trial all configs
.venv/bin/python -m cuda_exec.exec_cli exec.action=trial \
  exec.kernel=matmul exec.arch=auto exec.impl=gen-cuda exec.gpu=4

# Trial specific configs
.venv/bin/python -m cuda_exec.exec_cli exec.action=trial \
  exec.kernel=matmul exec.arch=auto exec.impl=gen-cuda exec.gpu=4 \
  'exec.configs=[mat-256x256,mat-8192x8192]'

# Profile (NCU)
.venv/bin/python -m cuda_exec.exec_cli exec.action=profile \
  exec.kernel=matmul exec.arch=auto exec.impl=gen-cuda exec.gpu=4 \
  'exec.configs=[mat-8192x8192]' exec.side=generated
```

## Workflow

```
compile (once per revision)
   ↓
trial (any configs, repeatable)
   ↓
profile (1-2 configs, NCU deep dive)
```

1. **Compile** — resolves impl slug, loads ref + gen files, runs nvcc/ptxas
2. **Trial** — runs kernel on selected configs, returns correctness + latency
3. **Profile** — NCU hardware metrics. See `artifacts/profiling-guide.md` for
   key metrics, bottleneck classification, and assembly examination.

New source code → increment `exec.revision`. Old revisions are immutable.

## Rules

- Compile exactly once per revision before trial or profile
- One compile fans out to many trial/profile calls
- Same `run_tag` across compile → trial → profile
- `exec.revision` must increment when source code changes
