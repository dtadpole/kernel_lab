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
python plugins/ik/scripts/ik_env.py status
python plugins/ik/scripts/ik_env.py test

# Python dev environment (most common)
python plugins/ik/scripts/ik_env.py install kit=venv
python plugins/ik/scripts/ik_env.py nuke kit=venv
python plugins/ik/scripts/ik_env.py reinstall kit=venv

# CUDA toolkit (from NVIDIA official runfile — never use repo packages)
python plugins/ik/scripts/ik_env.py install kit=cuda-toolkit
python plugins/ik/scripts/ik_env.py install kit=cuda-toolkit version=13.2

# CUDA driver (from NVIDIA official runfile — needs root, reboot recommended)
python plugins/ik/scripts/ik_env.py install kit=cuda-driver
python plugins/ik/scripts/ik_env.py install kit=cuda-driver version=595
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

## What `install kit=cuda-toolkit` does

1. Downloads the official CUDA runfile from `developer.download.nvidia.com`
2. Installs toolkit-only (no driver) to `/usr/local/cuda-<major>.<minor>`
3. Verifies nvcc, headers, and libnvrtc after install

**IMPORTANT:** Always use the official NVIDIA runfile installer. Never use
system package managers (`dnf install cuda-*`, `apt install cuda-*`) or
internal repos (`rolling_stable_nvidia`). The runfile gives us a clean,
self-contained toolkit at a known path with no dependency surprises.

Known versions (update `_resolve_cuda_version()` in `ik_env.py` to add more):

| Version | Toolkit | Driver | Runfile |
|---------|---------|--------|---------|
| 12.8 | 12.8.1 | 570.86.15 | `cuda_12.8.1_570.86.15_linux.run` |
| 13.0 | 13.0.1 | 580.76.02 | `cuda_13.0.1_580.76.02_linux.run` |
| 13.2 | 13.2.0 | 595.45.04 | `cuda_13.2.0_595.45.04_linux.run` |

## What `install kit=cuda-driver` does

1. Downloads the official NVIDIA driver runfile from `download.nvidia.com/XFree86/`
2. Installs with `--silent` (requires root / sudo)
3. Verifies the new driver via `nvidia-smi`

**Safety checks:** Refuses to install if GPU compute processes are running.
Accepts driver major version (e.g. `595`), full version (e.g. `595.45.04`),
or CUDA version (e.g. `13.2`). A reboot is strongly recommended after install.

Known drivers (update `_resolve_driver_version()` to add more):

| Driver Major | Full Version | CUDA | URL pattern |
|-------------|-------------|------|-------------|
| 550 | 550.90.07 | 12.4 | `XFree86/Linux-x86_64/550.90.07/` |
| 570 | 570.86.15 | 12.8 | `XFree86/Linux-x86_64/570.86.15/` |
| 580 | 580.76.02 | 13.0 | `XFree86/Linux-x86_64/580.76.02/` |
| 595 | 595.45.04 | 13.2 | `XFree86/Linux-x86_64/595.45.04/` |

## Host Config

All settings come from `conf/hosts/default.yaml`:

| Field | Used by | Example |
|-------|---------|---------|
| `env.cuda_home` | CUDA_HOME for compilation | `/usr/local/cuda-13.0` |
| `env.torch_cuda` | PyTorch index suffix | `cu130` |
| `env.torch_cuda_arch` | TORCH_CUDA_ARCH_LIST | `9.0` |
| `network.proxy_bypass_method` | pip via ssh localhost | `ssh_localhost` |
| `hardware.driver_version` | verify libcuda version | `580.65.06` |
