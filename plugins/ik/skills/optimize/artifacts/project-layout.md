# Project Layout (Optimize)

For impl slug format and discovery, see `artifacts/slug-resolution.md`.
For KB directory structure and gem rules, see `artifacts/kb-layout.md`.

## Key Paths

| What | Path |
|------|------|
| Reference impls | `data/ref/{kernel}/` |
| Peak impls | `.peak/{arch}/{kernel}/` |
| Gen impls (scratch) | `~/kernel_lab_kb/runs/run_<host>/gen/{arch}/{kernel}/` |
| Configs | `data/configs/{kernel}.json` |
| Roofline specs | `docs/roofline/` |
| NVIDIA docs | Use `/ik:docs` or web search |

## Gen Code Resolution

`cuda_exec/impls.py` resolves gen code from:
1. `IK_RUN_HOME` env var (if set) → `$IK_RUN_HOME/gen/{arch}/{kernel}/`
2. Active KB run: `~/kernel_lab_kb/runs/run_{host_slug}/gen/{arch}/{kernel}/`

## Seed Strategies

| `seed` | Behavior |
|--------|----------|
| `auto` | LLM inspects gems, picks a promising or under-explored seed |
| `latest` | Always seed from the most recent gem |
| `init` | Start from scratch — empty gen/, no seed |
| `vNNN` | Seed from a specific gem version (e.g. `v003`) |

### `seed=auto` Decision Process

```python
from cuda_exec.impls import list_gems, reseed_gen

gems = list_gems(kernel, arch, run_tag=run_tag)
# LLM inspects each gem's results:
#   - Which gems were fully explored vs abandoned early?
#   - Is the latest gem at a local optimum?
#   - Would backtracking open new possibilities?
reseed_gen(kernel, arch, gem_path=chosen_gem["gen_path"])
```
