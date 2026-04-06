# Design: kernel_lab_kb Unified Run Structure

## Overview

Restructure `kernel_lab_kb` so that generated kernel code lives as run output
(not in `kernel_lab/data/gen/`), with per-run iteration history and gems.

## Final Structure

```
kernel_lab_kb/
└── runs/
    ├── run_20260405_2232/
    │   ├── gen/                                # scratch space — solver's working area
    │   │   └── sm90/matmul/cuda/cuda.cu        # cleared + re-seeded each solver init
    │   │
    │   ├── ref/matmul/cublas/cublas.py          # reference snapshot (immutable)
    │   ├── configs/matmul.json                  # config snapshot (immutable)
    │   │
    │   ├── impls/                               # immutable iteration history
    │   │   ├── 20260405_2232/                   # seeded from gem of previous run
    │   │   │   ├── gen/sm90/matmul/cuda/cuda.cu # frozen code snapshot
    │   │   │   ├── compile/                     # ptxas, SASS, resource usage
    │   │   │   └── results.json                 # per-config latency, correctness
    │   │   ├── 20260405_2245/                   # second iteration
    │   │   │   ├── gen/sm90/matmul/cuda/cuda.cu
    │   │   │   ├── compile/
    │   │   │   └── results.json
    │   │   └── ...
    │   │
    │   ├── gems/gen-cuda/                       # best impls from THIS run
    │   │   ├── v001/  → impls/20260405_2232     # first gem
    │   │   └── v002/  → impls/20260405_2245     # beat v001
    │   │
    │   ├── journal/                             # optimization session log
    │   │   ├── transcript.md                    # what the agent tried + reasoning
    │   │   └── decisions.md                     # key decisions and NCU evidence
    │   │
    │   └── command.json                         # run metadata
    │
    ├── run_20260406_0045/
    │   └── ...
    │
    └── run_20260406_0127/
        └── ...
```

## Concepts

### `gen/` — Mutable Scratch Space
- Solver's working directory for the kernel being optimized
- Cleared and re-seeded from a gem each time the solver (re)initializes
- Layout mirrors `kernel_lab/data/gen/`: `<arch>/<kernel>/<impl>/`
- NOT preserved — `impls/` snapshots are the permanent record

### `ref/` — Immutable Reference Snapshot
- Copied from `kernel_lab/data/ref/` at run creation
- Layout: `<kernel>/<impl>/` (arch-agnostic, same as source)
- Never modified during the run

### `configs/` — Immutable Config Snapshot
- Copied from `kernel_lab/data/configs/` at run creation

### `impls/<timestamp>/` — Immutable Iteration History
- Each compile+trial cycle produces one timestamped entry
- Contains a frozen snapshot of `gen/` at that point + compile artifacts + results
- Timestamp format: `YYYYMMDD_HHMM`
- `ls impls/` gives chronological optimization history

### `gems/<impl-slug>/v00N/` — Per-Run Best Implementations
- Created when an impl iteration beats the previous best within this run
- Points back to the specific `impls/<timestamp>/` that achieved it
- Solver re-seeds `gen/` from the latest gem when re-initializing
- Cross-run seeding: new run reads gems from the most recent previous run

### `journal/` — Optimization Session Log
- Replaces the planned `agent_journal/` from SYSTEM_DESIGN.md
- Per-run, not a separate global directory
- Contains reasoning, NCU profiles, what was tried and why

### `command.json` — Run Metadata
```json
{
  "timestamp": "2026-04-06T01:27:34",
  "slug": "tma-store-epilogue",
  "kernel": "matmul",
  "arch": "sm90",
  "device": "NVIDIA H100",
  "gpu_index": 4,
  "git_commit": "5dca463",
  "seed_gem": "runs/run_20260406_0045/gems/gen-cuda/v003"
}
```

## Solver Lifecycle

```
1. Create run:        runs/run_<timestamp>/
2. Snapshot:          ref/, configs/ copied from kernel_lab/data/
3. Seed gen/:         clear gen/, copy from seed gem
4. Modify gen/:       solver edits gen/sm90/matmul/cuda/cuda.cu
5. Compile + trial:   snapshot gen/ → impls/<timestamp>/gen/
                      compile artifacts → impls/<timestamp>/compile/
                      results → impls/<timestamp>/results.json
6. Evaluate:          if improvement → gems/gen-cuda/v00N/ → impls/<timestamp>
7. Loop:              back to step 4
8. Re-init:           clear gen/, re-seed from latest gem → back to step 4
```

## Changes Required

### kernel_lab (tooling repo)

| File | Change |
|------|--------|
| `data/gen/` | Remove entirely. No longer in tooling repo. |
| `cuda_exec/trajectory.py` | Rewrite `prepare_run` and `finalize_run` for new paths |
| `cuda_exec/formal.py` | Update to read gen code from KB run, write impls per-iteration |
| `cuda_exec/impls.py` | `resolve_impl()` reads from `gen/` scratch in active run or latest gem |
| `plugins/ik/skills/bench/SKILL.md` | Update path references |
| `plugins/ik/skills/optimize/SKILL.md` | Update seed-from-gem logic, iteration flow |
| `plugins/ik/skills/exec/SKILL.md` | Update compile/trial paths |
| `docs/SYSTEM_DESIGN.md` | Replace `agent_journal/` with `runs/.../journal/` |
| `AGENTS.md` | Update data directory layout section |

### kernel_lab_kb (knowledge repo)

| Change | Detail |
|--------|--------|
| `ik_bench/` | Remove prefix. `ik_bench/runs/` → `runs/`, `ik_bench/gems/` → inside runs |
| Existing runs | Migrate `ik_bench/runs/<kernel>/<arch>/<ts>/` → `runs/run_<ts>/` |
| Existing gems | Migrate into corresponding run folders |

## Migration Plan

1. Create new `runs/` structure in kernel_lab_kb
2. Migrate existing `ik_bench/runs/` data to new format
3. Update `trajectory.py` to write new format
4. Update `formal.py` and `impls.py` to read new format
5. Update skills (bench, optimize, exec)
6. Remove `data/gen/` from kernel_lab
7. Update SYSTEM_DESIGN.md and AGENTS.md
8. Test: `ik:bench matmul` must produce correct results in new structure
