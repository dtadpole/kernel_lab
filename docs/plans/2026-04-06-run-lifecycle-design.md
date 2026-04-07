# Design: Run Lifecycle — Gen Code Initialization

## Principle

**No gems → no code.** A fresh run starts with an empty gen/ folder.
Code only enters the run through explicit actions, never through
automatic cross-run copying or legacy fallback.

## Run States

```
1. EMPTY        — fresh run, gen/ does not exist
2. SEEDED       — gen/ has code, placed by ik:optimize (seed=auto|latest|init)
3. ACTIVE       — solver is modifying gen/ (compile/trial/iterate)
4. BENCHMARKED  — formal bench created impls/<ts>/ with frozen snapshot
5. GEMMED       — improvement found, gems/<slug>/v00N/ created
```

## How Code Enters a Run

| Entry point | Action |
|-------------|--------|
| `ik:optimize seed=auto` | LLM inspects THIS run's gems, picks one to seed gen/ |
| `ik:optimize seed=latest` | Copies from THIS run's latest gem to gen/ |
| `ik:optimize seed=init` | Creates empty gen/, solver writes code from scratch |
| `ik:optimize seed=v003` | Copies from THIS run's specific gem to gen/ |
| Manual copy | User or supervisor copies code into gen/ directly |

**There is NO automatic seeding.** `_ensure_gen_dir()` does NOT auto-seed.
If gen/ doesn't exist, it stays empty. The solver (ik:optimize) is responsible
for populating it.

## What Happens When gen/ is Empty

When `ik:optimize` starts and finds gen/ empty (no gems in this run):

1. **First time (no gems at all):** The solver generates initial kernel code.
   This is the `seed=init` path. The LLM writes a kernel from scratch using:
   - NVIDIA documentation (PTX ISA, CUDA Programming Guide)
   - Reference implementations (ref-pytorch, ref-cutedsl) for API contract
   - Architecture specs (SM90 WGMMA, TMA, etc.)
   - Previous results files for institutional knowledge

2. **Has gems:** The solver picks a gem to seed from (`seed=auto` or `seed=latest`).

## _ensure_gen_dir Behavior

```python
def _ensure_gen_dir(kernel, arch, run_tag, kb_repo):
    gen_path = run_dir / "gen" / arch / kernel
    if gen_path.exists():
        return gen_path       # Already seeded — use as-is
    # NOT seeded — return path but don't create or auto-populate
    return gen_path           # Caller must check .exists()
```

`_ensure_gen_dir` does NOT create directories or copy files.
It only returns the path. The caller (ik:optimize, ik:bench) decides
what to do if the path doesn't exist.

## _gen_dir Behavior

```python
def _gen_dir(kernel, arch, data_root=None):
    if data_root:
        return data_root / "gen" / arch / kernel  # bench snapshot
    return _ensure_gen_dir(kernel, arch)           # may not exist
```

If `_gen_dir` returns a path that doesn't exist, `list_impls` and
`resolve_impl` will simply not find gen-* implementations. This is
correct — a run with no code has no gen implementations.

## _find_latest_gem Behavior

```python
def _find_latest_gem(kernel, arch, run_tag):
    # Search ONLY within run_tag's gems/
    # No cross-run. No legacy fallback.
    # Returns None if no gems exist in this run.
```

## reseed_gen Behavior

```python
def reseed_gen(kernel, arch, run_tag, gem_path=None):
    # Clear gen/ scratch
    # If gem_path provided: copy from that gem
    # If gem_path is None: find latest gem in THIS run
    # If no gems in this run: leave gen/ empty (caller handles)
```

## ik:optimize Startup

```
1. Resolve run_tag (Hydra: explicit > ENV > host_slug)
2. Check gen/ exists for this kernel+arch
3. If exists → use current code (resume optimization)
4. If NOT exists:
   a. Check gems in THIS run → seed=auto picks one
   b. No gems → seed=init → LLM generates initial code
5. Enter optimization loop (Phase 1-7)
```

## ik:bench Startup

```
1. Resolve run_tag (same as optimize)
2. Check gen/ exists for this kernel+arch
3. If exists → snapshot to impls/, compile, trial
4. If NOT exists → ERROR: "No gen code found. Run ik:optimize first."
```

## Cross-Run Data Flow

Runs are isolated. The ONLY way data flows between runs is:
- **User copies** code from one run's gem to another run's gen/
- **Supervisor** explicitly sets run_tag and seeds gen/ before starting solver

There is no automatic cross-run discovery, seeding, or fallback.
