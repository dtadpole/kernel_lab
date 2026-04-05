---
name: local
description: Manage the local development Python environment (.venv) — setup, test, nuke, activate
user-invocable: true
argument-hint: <info|setup|test|nuke|activate> [--rebuild]
---

# Local Development Environment

Manage the project `.venv` for the current host. Auto-detects the machine
from `conf/hosts/default.yaml` and applies the correct CUDA toolkit, PyTorch
build, driver workarounds, and network constraints.

## Commands

| Command | Purpose |
|---------|---------|
| **info** | Show detected host, driver, CUDA, venv status |
| **setup** | Create `.venv` + install all deps (idempotent) |
| **test** | Verify environment: torch, CUDA, flash-attn, key packages |
| **nuke** | Remove `.venv` completely |
| **activate** | Print shell export commands (`eval` it) |

## Usage

```bash
python plugins/devenv/cli.py info
python plugins/devenv/cli.py setup
python plugins/devenv/cli.py setup --rebuild    # nuke + recreate
python plugins/devenv/cli.py test
python plugins/devenv/cli.py nuke
eval $(python plugins/devenv/cli.py activate)
```

## What `setup` does

1. Creates `.venv` with the host's Python
2. Installs PyTorch from the correct CUDA index (e.g. `cu128` vs `cu130`)
3. Installs core dependencies from `plugins/devenv/requirements.txt`
4. Installs flash-attn (source build if CUDA >= 13.0)
5. Installs FA4 CuTe DSL module (`flash_attn.cute`)

For hosts without direct internet (e.g. `h8_4`), pip runs through
`ssh localhost` to bypass the sandbox proxy.

## What `activate` exports

```bash
export PATH=".venv/bin:$PATH"
export CUDA_HOME="/usr/local/cuda-13.0"
export LD_PRELOAD="/usr/lib64/libcuda.so.580.65.06"   # if needed
export TORCH_CUDA_ARCH_LIST="9.0"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
```

## Host Config

All host-specific settings come from `conf/hosts/default.yaml`:

```yaml
h8_4:
  hardware:
    driver_version: "580.65.06"
    driver_cuda_version: "13.0"
  env:
    cuda_home: /usr/local/cuda-13.0
    torch_cuda: "cu130"
    torch_cuda_arch: "9.0"
    ld_preload: /usr/lib64/libcuda.so.580.65.06
  network:
    internet: false
    proxy_bypass_method: "ssh_localhost"
```
