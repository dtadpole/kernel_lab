# Project Layout

## Directory Structure

| What | Path |
|------|------|
| Reference impls | `data/ref/{kernel}/` (in kernel_lab) → slugs `ref-{stem}` |
| Generated impls | `~/kernel_lab_kb/runs/run_<host>/gen/{arch}/{kernel}/` → slugs `gen-{stem}` |
| Latest gem (seed) | `~/kernel_lab_kb/runs/run_*/gems/{slug}/v00N/gen/` |
| Configs | `data/configs/{kernel}.json` (in kernel_lab) |
| Results | `~/kernel_lab_kb/runs/run_<ts>/impls/<bench_ts>/` |
| Roofline specs | `docs/roofline/` |
| NVIDIA docs (local) | Use `/ik:docs` to search indexed CUDA Toolkit docs |
| NVIDIA docs (online) | Web search for official NVIDIA docs, PTX ISA, tuning guides |

## Gen Code Resolution

`cuda_exec/impls.py` auto-resolves gen code from:
1. Active KB run's `gen/` folder (`~/kernel_lab_kb/runs/run_<host>/gen/`)
2. If gen/ doesn't exist, seeds it from the latest gem across all KB runs
3. The run tag defaults to `run_<host_slug>` (auto-detected from `conf/hosts/default.yaml`)

## Implementation Slugs

Slugs follow `{source}-{name}` pattern (e.g. `ref-cublas`, `gen-cuda`).
See `cuda_exec/impls.py` for slug resolution logic.

```python
from cuda_exec.impls import list_impls
list_impls("matmul", "sm90")  # → [{"slug": "ref-cublas", ...}, {"slug": "gen-cuda", ...}]
```

## Slug Mapping Examples

```
data/ref/                          kernel_lab_kb gen/ (per-run scratch):
  matmul/                            sm90/matmul/
    cublas.py      → ref=cublas        cuda/cuda.cu  → gen=cuda
  fa4/                               sm90/fa4/
    cudnn.py       → ref=cudnn         cuda/cuda.cu  → gen=cuda
    cutedsl.py     → ref=cutedsl
  vecadd/
    cublas.py      → ref=cublas
```

Default ref discovery: when `ref=` is omitted, all `.py`/`.cu` files in
`data/ref/{kernel}/` are used.

## Seed Strategies

| `seed` | Behavior |
|--------|----------|
| `auto` | **Default.** LLM inspects gems from the current run, picks a promising or under-explored seed — not always the latest. Enables tree-search over optimization branches. |
| `latest` | Always seed from the most recent gem. No LLM decision. |
| `init` | Start from scratch — empty gen/ folder, no seed. For writing a kernel from zero. |
| `v003` | Seed from a specific gem version in the current run. |

### `seed=auto` Decision Process

```python
from cuda_exec.impls import list_gems, reseed_gen

gems = list_gems(kernel, arch, run_tag=run_tag)
# LLM inspects each gem's results + the run's journal:
#   - Which gems were fully explored vs abandoned early?
#   - Which architectural branches haven't been tried?
#   - Is the latest gem at a local optimum (micro-opts exhausted)?
#   - Would backtracking to an earlier fork point open new possibilities?
# Then picks a gem and calls:
reseed_gen(kernel, arch, gem_path=chosen_gem["gen_path"])
```
