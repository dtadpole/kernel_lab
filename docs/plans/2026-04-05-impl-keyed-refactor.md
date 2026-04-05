# Plan: Impl-Keyed Refactor

**Goal**: Replace the hardcoded 3-slot model (reference/generated/cudnn) with an impl-keyed design where N implementations are first-class citizens, each independently benchmarked.

**Architecture**: 
- Data directory: each impl gets its own subdirectory (`data/{ref,gen}/.../{impl_name}/`)
- Request/Response: `Dict[slug, files]` replaces 3 fixed fields
- Trial: each impl runs independently via the appropriate harness (`.py` → Python, `.cu` → C)
- Backward compat: `trial.py` CLI keeps working (it receives workspace path with staged impl dirs)

**Tech Stack**: Python, Pydantic, CUDA/PTX (unchanged kernel code)

---

## Phase 1: Directory restructure

Move files into per-impl subdirectories. Update `impls.py` to scan subdirs.

### Step 1: Move data files to per-impl subdirs

**Files**: `data/ref/`, `data/gen/`

Move:
```
data/ref/matmul/cublas.py          → data/ref/matmul/cublas/cublas.py
data/ref/fa4/cudnn.py              → data/ref/fa4/cudnn/cudnn.py
data/ref/fa4/cutedsl.py            → data/ref/fa4/cutedsl/cutedsl.py
data/ref/vecadd/cublas.py          → data/ref/vecadd/cublas/cublas.py

data/gen/sm90/matmul/cutedsl.py    → data/gen/sm90/matmul/cutedsl/cutedsl.py
data/gen/sm90/matmul/cute_gemm_sm90.py → data/gen/sm90/matmul/cutedsl/cute_gemm_sm90.py
data/gen/sm90/matmul/cuda.cu       → data/gen/sm90/matmul/cuda/cuda.cu
data/gen/sm90/fa4/cuda.cu          → data/gen/sm90/fa4/cuda/cuda.cu
data/gen/sm90/vecadd/cuda.cu       → data/gen/sm90/vecadd/cuda/cuda.cu

(same pattern for sm120)
```

Verify: `find data/ref data/gen -type f | sort` shows new layout.

### Step 2: Update impls.py to scan subdirs

**File**: `cuda_exec/impls.py`

Change `list_impls()` and `resolve_impl()`:
- Old: scan files in `data/ref/{kernel}/` → each `.py`/`.cu` file = one impl
- New: scan **subdirectories** in `data/ref/{kernel}/` → each subdir = one impl
  - Entry point: `{subdir}/{subdir_name}.py` or `{subdir}/{subdir_name}.cu`
  - Helpers: all other files in the subdir

Verify: `python -c "from cuda_exec.impls import list_impls; print(list_impls('matmul', 'sm90'))"`

### Step 3: Test Phase 1 — bench matmul

Run: `.venv/bin/python -m cuda_exec.formal bench.kernel=matmul bench.arch=sm90 bench.gpu=7`

All 6 configs must produce results. This validates impls.py finds files in new layout.

---

## Phase 2: Request/Response refactor

### Step 4: Update models.py — CompileRequest

**File**: `cuda_exec/models.py`

Replace:
```python
class CompileRequest(RequestBase):
    reference_files: Dict[str, str]
    generated_files: Dict[str, str]
    cudnn_files: Dict[str, str]
```

With:
```python
class CompileRequest(RequestBase):
    impls: Dict[str, Dict[str, str]]
    # slug → {filename: content}
```

### Step 5: Update models.py — TrialConfigOutput

**File**: `cuda_exec/models.py`

Replace:
```python
class TrialConfigOutput(BaseModel):
    reference: Dict[str, Any]
    generated: Dict[str, Any]
    cudnn: Dict[str, Any]
```

With:
```python
class ImplTrialResult(BaseModel):
    performance: PerformanceSummary
    correctness: CorrectnessSummary | None = None  # only for gen impls vs ref baseline

class TrialConfigOutput(BaseModel):
    status: ...
    impls: Dict[str, ImplTrialResult]  # slug → result
    baseline_slug: str  # which ref-* impl was used as correctness baseline
```

### Step 6: Update tasks.py — compile staging

**File**: `cuda_exec/tasks.py`

Change `run_compile_task()`:
- Old: write to `inputs/reference/`, `inputs/generated/`, `inputs/cudnn/`
- New: write to `inputs/{slug}/` for each impl
  - `.cu` impls: also compile via `compile.sh` and produce binary

### Step 7: Update tasks.py — trial routing

**File**: `cuda_exec/tasks.py`

Change `run_trial_task()`:
- Old: run trial.py once (measures reference + generated + cudnn)
- New: for each impl in workspace:
  - `.py` impl: run `measure_reference()` in-process
  - `.cu` impl: run compiled binary via subprocess
- Correctness: compare each gen-* output against the first ref-* baseline

### Step 8: Update trial.py — impl-keyed output

**File**: `cuda_exec/scripts/trial.py`

Change `main()`:
- Old: loads reference, cudnn, generated from fixed paths
- New: scans `inputs/*/` for all impl dirs, runs each one
- Output JSON: `{"impls": {"ref-cublas": {...}, "gen-cutedsl": {...}, "gen-cuda": {...}}}`

### Step 9: Update eval_support.py — rename functions

**File**: `cuda_exec/scripts/eval_support.py`

- `load_reference_entry()` → `find_py_entry_point(impl_dir)`
- `load_cudnn_entry()` → removed (subsumed by generic find)
- `measure_reference()` → `measure_py_impl(module, config, device, ...)`
- No functional change, just naming to avoid "reference"/"cudnn" confusion

### Step 10: Update formal.py — impl-keyed flow

**File**: `cuda_exec/formal.py`

Change `formal_benchmark()`:
- Old: builds CompileRequest with 3 fixed slots, treats .py gens as "measured on reference side"
- New: builds CompileRequest with all impls, each independently benchmarked
- Iterates over ALL impls (not just .cu gens)

### Step 11: Update exec_cli.py

**File**: `cuda_exec/exec_cli.py`

Update to use new CompileRequest format.

### Step 12: Update AGENTS.md and skill docs

**Files**: `AGENTS.md`, `plugins/ik/skills/{bench,exec,optimize}/SKILL.md`

Update path examples and contract descriptions.

---

## Phase 3: End-to-end verification

### Step 13: Bench matmul — all 3 impls must have data

Run: `.venv/bin/python -m cuda_exec.formal bench.kernel=matmul bench.arch=sm90 bench.gpu=7`

Verify output JSON has `impls: {ref-cublas: {...}, gen-cutedsl: {...}, gen-cuda: {...}}` with median latency > 0 for all three.

### Step 14: Bench fa4 — all 3 impls must have data

Run: `.venv/bin/python -m cuda_exec.formal bench.kernel=fa4 bench.arch=sm90 bench.gpu=7`

### Step 15: Commit, push, pull

```bash
git add -A
git commit -m "refactor: impl-keyed design — N impls as first-class citizens"
git push
```

---

## Task Dependencies

| Group | Steps | Can Parallelize | Files Touched |
|-------|-------|-----------------|---------------|
| 1 | Step 1 | No | data/ref/, data/gen/ |
| 2 | Step 2 | No (depends on 1) | cuda_exec/impls.py |
| 3 | Step 3 | No (depends on 2) | (test only) |
| 4 | Steps 4-5 | Yes | cuda_exec/models.py |
| 5 | Steps 6-7 | No (depends on 4) | cuda_exec/tasks.py |
| 6 | Steps 8-9 | Yes (depends on 5) | trial.py, eval_support.py |
| 7 | Steps 10-11 | Yes (depends on 5) | formal.py, exec_cli.py |
| 8 | Step 12 | No (depends on 6-7) | AGENTS.md, SKILL.md |
| 9 | Steps 13-15 | No (depends on all) | (test + commit) |
