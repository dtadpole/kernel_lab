---
name: env
description: Manage development environment — Python venv, CUDA toolkit, NVIDIA driver
user-invocable: true
argument-hint: <status|test|install|nuke|reinstall> [kit=venv|cuda-toolkit|cuda-driver] [version=X.Y]
---

# Environment Management

Manage the development environment kits on the current host. Auto-detects
the machine from `conf/hosts/default.yaml` and applies host-specific settings.

## Commands

| Command | Destructive | kit required | Description |
|---------|-------------|--------------|-------------|
| `status` | no | no | Show all kits status (fast, no GPU) |
| `test` | no | no | GPU verification (imports, matmul, FA4) |
| `install` | **yes** | **yes** | Install a kit |
| `nuke` | **yes** | **yes** | Delete a kit |
| `reinstall` | **yes** | **yes** | nuke + install |

## Kits

| kit | What it manages | Future |
|-----|-----------------|--------|
| `venv` | Standalone Python + `.venv/` + PyTorch + deps | `venv-runtime` |
| `cuda-toolkit` | CUDA toolkit (nvcc, headers, libs) | — |
| `cuda-driver` | NVIDIA kernel driver + libcuda | — |

## Usage

```bash
# Check environment
python plugins/ik/scripts/env.py status
python plugins/ik/scripts/env.py test

# Python dev environment (most common)
python plugins/ik/scripts/env.py install kit=venv
python plugins/ik/scripts/env.py nuke kit=venv
python plugins/ik/scripts/env.py reinstall kit=venv

# CUDA toolkit
python plugins/ik/scripts/env.py install kit=cuda-toolkit
python plugins/ik/scripts/env.py install kit=cuda-toolkit version=13.0

# CUDA driver (needs root)
python plugins/ik/scripts/env.py install kit=cuda-driver
```

## What `install kit=venv` does

1. Installs standalone Python 3.12 via `uv` (not fbcode — uses system ld.so)
2. Creates `.venv` with the standalone Python
3. Installs PyTorch matching the host's CUDA version (from `conf/hosts/default.yaml`)
4. Installs all dependencies from `plugins/ik/requirements.txt`
5. For hosts without internet, pip runs through `ssh localhost`

## Key: standalone Python avoids fbcode libcuda conflict

Meta devvms ship fbcode Python whose ELF interpreter (`ld.so`) hard-codes
`/usr/local/fbcode/platform010/lib/` as a search path. That directory contains
an old `libcuda.so.550` which shadows the real system driver (580/CUDA 13.0).

The `uv`-installed standalone Python uses the system `/lib64/ld-linux-x86-64.so.2`,
which finds the correct `/usr/lib64/libcuda.so.580.65.06` automatically.
No `LD_PRELOAD` needed.

## Host Config

All settings come from `conf/hosts/default.yaml`:

| Field | Used by | Example |
|-------|---------|---------|
| `env.cuda_home` | CUDA_HOME for compilation | `/usr/local/cuda-13.0` |
| `env.torch_cuda` | PyTorch index suffix | `cu130` |
| `env.torch_cuda_arch` | TORCH_CUDA_ARCH_LIST | `9.0` |
| `network.proxy_bypass_method` | pip via ssh localhost | `ssh_localhost` |
| `hardware.driver_version` | verify libcuda version | `580.65.06` |
