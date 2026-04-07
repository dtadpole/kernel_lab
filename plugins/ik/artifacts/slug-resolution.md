# Implementation Slug Resolution

Implementations are discovered dynamically from the directory structure.
A **slug** has the format `{source}-{name}`:

| Source | Directory | Example slug | Example path |
|--------|-----------|--------------|--------------|
| `ref` | `ref/{kernel}/` | `ref-pytorch` | `ref/matmul/pytorch/pytorch.py` |
| `peak` | `peak/{arch}/{kernel}/` | `peak-cuda` | `peak/sm90/fa4/cuda/cuda.cu` |
| `gen` | `gen/{arch}/{kernel}/` | `gen-cuda` | `gen/sm90/matmul/cuda/cuda.cu` |

**Forward resolution** (slug → files): `resolve_impl(kernel, arch, slug)`
- Tries `{name}.py` first, then `{name}.cu` in the source directory
- Auto-includes helper files (`.py` helpers for `.py` entry points, `.h`/`.cuh` for `.cu`)

**Reverse discovery** (directory → all slugs): `list_impls(kernel, arch)`
- Scans `ref/{kernel}/` — every subdir with `.py` or `.cu` entry point becomes `ref-{name}`
- Scans `peak/{arch}/{kernel}/` → `peak-{name}`
- Scans `gen/{arch}/{kernel}/` → `gen-{name}`

When `impls` is `all` (default), both directions are used: reverse discovery
finds all slugs, then forward resolution loads their files.

Both functions live in `cuda_exec/impls.py` and accept `data_root` to work
against either original `data/` or a snapshot copy.

## Environment Resolution

The benchmark auto-resolves the host environment via `cuda_exec.host_env`
by matching the current hostname against `conf/hosts/default.yaml`:

| What | Source | Override |
|------|--------|----------|
| `arch` | `env.torch_cuda_arch` or nvidia-smi | `bench.arch=sm90` |
| `gpu` | `benchmark.cuda_visible_devices` | `bench.gpu=5` |
| `CUDA_HOME` | `env.cuda_home` | (env var) |
| `LD_PRELOAD` | `env.ld_preload` | (env var) |

No manual environment setup is needed — `formal.py` reads the host config and
applies the right settings automatically.
