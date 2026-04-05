# Design: ik:env — Environment Management Skill

**Goal**: Single command to manage Python venv, CUDA toolkit, and NVIDIA driver on any devvm. Auto-detects host, installs standalone Python (not fbcode), picks correct PyTorch for the CUDA version.

## Command Set

```bash
# Non-destructive (no kit required)
/ik:env status                                    # all kits status
/ik:env test                                      # GPU verification

# Destructive (kit required)
/ik:env install kit=venv                          # standalone Python + .venv + deps
/ik:env nuke kit=venv                             # delete .venv
/ik:env reinstall kit=venv                        # nuke + install

/ik:env install kit=cuda-toolkit                  # auto-detect version
/ik:env install kit=cuda-toolkit version=13.0     # explicit version

/ik:env install kit=cuda-driver                   # needs root
```

## Kit Values

| kit | product | future |
|-----|---------|--------|
| `venv` | `.venv/` (local repo) | `venv-runtime` (remote service) |
| `cuda-toolkit` | `/usr/local/cuda-X.Y/` | — |
| `cuda-driver` | kernel module + libcuda | — |

## File Structure

```
scripts/env.py                     # CLI script (~400 lines)
plugins/ik/skills/env/SKILL.md     # skill doc
conf/env/default.yaml              # Hydra config defaults
plugins/deprecated/devenv/         # archived old plugin
```

## Hydra Config

```yaml
# conf/env/default.yaml
kit: null               # venv | cuda-toolkit | cuda-driver
version: auto           # auto-detect from driver, or explicit
python_version: "3.12"
force: false
```

## Kit: venv — Install Flow

```
[1/4] Python — uv standalone
      ssh localhost "uv python install cpython-3.12" (if needed)
      Verify interpreter is /lib64/ld-linux-x86-64.so.2 (NOT fbcode)

[2/4] Create .venv
      Use standalone Python: ~/.local/share/uv/python/.../bin/python3.12 -m venv .venv
      Verify .venv/bin/python interpreter

[3/4] PyTorch
      Read torch_cuda from conf/hosts/default.yaml (e.g. cu130)
      ssh localhost ".venv/bin/pip install torch --index-url=..."

[4/4] Dependencies
      ssh localhost ".venv/bin/pip install -r plugins/ik/requirements.txt"
```

## Kit: venv — Status Checks

```
[✓] Python standalone (not fbcode ld.so)
[✓] libcuda matches driver (580.65.06 → CUDA 13.0)
[✓] .venv exists
[✓] PyTorch 2.11.0+cu130
[✓] cutlass-dsl importable
[✓] flash-attn-4 importable
```

## Kit: venv — Test Checks (GPU)

```
[✓] torch.cuda available
[✓] CUDA matmul (bf16)
[✓] FA4 CuTe DSL smoke test
```

## Host Config Integration

All decisions read from `conf/hosts/default.yaml`:
- `env.cuda_home` → CUDA_HOME for compilation
- `env.torch_cuda` → PyTorch index (cu128/cu130)
- `env.torch_cuda_arch` → TORCH_CUDA_ARCH_LIST
- `network.proxy_bypass_method` → ssh_localhost for pip
- `hardware.driver_version` → verify libcuda matches
