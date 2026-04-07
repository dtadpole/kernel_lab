# Plan: Implement kernel_lab_kb Redesign

**Goal**: Migrate from `ik_bench/runs/{kernel}/{arch}/{ts}` + `ik_bench/gems/` + `data/gen/` to unified `runs/run_{ts}/` with per-run gen scratch, impls history, and gems.
**Architecture**: See `docs/plans/2026-04-06-kb-redesign.md` for full design.
**Tech Stack**: Python (cuda_exec), Markdown (skills), bash (migration)

## Step 1: Update trajectory.py path constants

**File**: `cuda_exec/trajectory.py`

### 1a. Change global path constants (lines 21-23)

```python
# OLD:
RUNS_DIR = KB_REPO / "ik_bench" / "runs"
GEMS_DIR = KB_REPO / "ik_bench" / "gems"

# NEW:
RUNS_DIR = KB_REPO / "runs"
# GEMS_DIR removed — gems are per-run now
```

### 1b. Verify no other code references GEMS_DIR directly
```bash
grep -n "GEMS_DIR" cuda_exec/trajectory.py
```

### 1c. Commit
```bash
git commit -m "refactor: trajectory.py path constants — drop ik_bench/ prefix"
```

## Step 2: Rewrite prepare_run() for new structure

**File**: `cuda_exec/trajectory.py` (function at line 392)

### 2a. Change run directory naming

```python
# OLD (line 412):
run_dir = runs_dir / kernel / arch / ts_str

# NEW:
run_dir = runs_dir / f"run_{ts_str}"
```

### 2b. Change snapshot paths

```python
# OLD (lines 430-446):
ref_dst = run_dir / "data" / "ref" / kernel
gen_dst = run_dir / "data" / "gen" / arch / kernel
cfg_dst = run_dir / "data" / "configs" / f"{kernel}.json"

# NEW:
ref_dst = run_dir / "ref" / kernel
gen_dst = run_dir / "gen" / arch / kernel
cfg_dst = run_dir / "configs" / f"{kernel}.json"
```

Drop the `data/` subdirectory — `gen/`, `ref/`, `configs/` are directly under run.

### 2c. Add kernel/arch to command.json

```python
command = {
    "timestamp": ts_str,
    "kernel": kernel,
    "arch": arch,
    # ... existing fields ...
}
```

### 2d. Verify
```bash
python3 -c "from cuda_exec.trajectory import prepare_run; print(prepare_run('matmul', 'sm90', ['gen-cuda','ref-pytorch'], 300))"
ls -la ~/kernel_lab_kb/runs/run_*/
```

### 2e. Commit
```bash
git commit -m "refactor: prepare_run() — run_<ts> naming, flat gen/ref/configs"
```

## Step 3: Rewrite finalize_run() for per-run gems

**File**: `cuda_exec/trajectory.py` (function at line 455)

### 3a. Change impls path

```python
# OLD (line 476):
impl_dir = run_dir / "impls" / impl_slug

# NEW — impls/<timestamp>/ per formal bench iteration:
bench_ts = time.strftime("%Y%m%d_%H%M%S")
impl_dir = run_dir / "impls" / bench_ts
```

### 3b. Snapshot gen code into impls

```python
# After creating impl_dir, copy current gen/ snapshot:
gen_snapshot_src = run_dir / "gen"
gen_snapshot_dst = impl_dir / "gen"
if gen_snapshot_src.exists():
    shutil.copytree(gen_snapshot_src, gen_snapshot_dst)
```

### 3c. Change gems path to per-run

```python
# OLD (line 507):
gem_base = gems_dir / kernel / arch / impl_slug

# NEW:
gem_base = run_dir / "gems" / impl_slug
```

### 3d. Remove gem data snapshot (code already in impls/)

```python
# OLD (lines 518-527): copy run_dir/"data" to gem_dir/"data"
# NEW: gem just has results.json, report.md, and source_run.json
gem_results["source_impl"] = str(impl_dir.relative_to(run_dir))
(gem_dir / "results.json").write_text(json.dumps(gem_results, indent=2) + "\n")
(gem_dir / "report.md").write_text(report)
# Also copy the actual kernel code for easy access:
gen_src = impl_dir / "gen"
if gen_src.exists():
    shutil.copytree(gen_src, gem_dir / "gen")
```

### 3e. Verify
```bash
# Run a bench and check the output structure
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m cuda_exec.formal bench.kernel=matmul
ls -R ~/kernel_lab_kb/runs/run_*/impls/
ls -R ~/kernel_lab_kb/runs/run_*/gems/
```

### 3f. Commit
```bash
git commit -m "refactor: finalize_run() — impls/<ts>, per-run gems, gen snapshot"
```

## Step 4: Update formal.py to use new snapshot paths

**File**: `cuda_exec/formal.py`

### 4a. Change snapshot_data reference

```python
# OLD (line 303):
snapshot_data = run_dir / "data"

# NEW:
snapshot_data = run_dir
```

Since gen/, ref/, configs/ are now directly under run_dir (not under data/).

### 4b. Verify
```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m cuda_exec.formal bench.kernel=matmul
# Check: all 6 configs pass, results written to new paths
```

### 4c. Commit
```bash
git commit -m "refactor: formal.py — snapshot_data is run_dir directly"
```

## Step 5: Update impls.py for KB-aware gen resolution

**File**: `cuda_exec/impls.py`

### 5a. Keep existing data_root logic working

The current `data_root` parameter already allows pointing to any directory.
When called from formal.py with `data_root=run_dir`, it resolves:
- `run_dir / "ref" / kernel` (for ref-*)
- `run_dir / "gen" / arch / kernel` (for gen-*)

This already works with the Step 2 changes. No code change needed in impls.py
as long as formal.py passes the correct data_root.

### 5b. Verify
```bash
python3 -c "
from cuda_exec.impls import list_impls
# Test with default data_root (kernel_lab/data/)
print(list_impls('matmul', 'sm90'))
"
```

### 5c. Commit (if any changes needed)

## Step 6: Update ik:bench skill doc

**File**: `plugins/ik/skills/bench/SKILL.md`

### 6a. Update output structure section (lines 93-120)

```markdown
## Output Structure

### kernel_lab_kb (git repo, text only)

\```
runs/
  run_<YYYYMMDD_HHMMSS>/
    command.json                    # invocation parameters
    gen/<arch>/<kernel>/            # solver scratch (mutable)
    ref/<kernel>/                   # reference snapshot (immutable)
    configs/<kernel>.json           # config snapshot (immutable)
    impls/<YYYYMMDD_HHMMSS>/        # formal bench output (immutable)
      gen/<arch>/<kernel>/          # frozen code snapshot
      compile/                     # ptxas + resource usage
      results.json                 # structured results
      report.md                    # human-readable table
    gems/<impl_slug>/
      v001/                        # first gem = first improvement
      v002/                        # only created when beating v001
    journal/                       # optimization session log
\```
```

### 6b. Update slug resolution table (lines 21-25)

Keep the `data/ref/` path (stays in kernel_lab).
Update `data/gen/` note to explain it's in kernel_lab_kb.

### 6c. Commit
```bash
git commit -m "docs: update bench skill for new KB run structure"
```

## Step 7: Update ik:optimize skill doc

**File**: `plugins/ik/skills/optimize/SKILL.md`

### 7a. Update project layout table (lines 86-97)

```markdown
| What | Path |
|------|------|
| Reference impls | `data/ref/{kernel}/` (in kernel_lab) |
| Generated impls | `~/kernel_lab_kb/runs/run_<latest>/gen/{arch}/{kernel}/` |
| Configs | `data/configs/{kernel}.json` (in kernel_lab) |
```

### 7b. Update baseline snapshot (lines 277-282)

```markdown
# Gen code is in the active run's gen/ scratch:
# ~/kernel_lab_kb/runs/run_<active>/gen/{arch}/{kernel}/{name}.cu
# Baseline is the gem that seeded this run
```

### 7c. Update seed discovery (lines 234-246)

```markdown
# Find latest gem:
ls ~/kernel_lab_kb/runs/run_*/gems/gen-cuda/ | tail -1
```

### 7d. Commit
```bash
git commit -m "docs: update optimize skill for new KB run structure"
```

## Step 8: Update AGENTS.md data directory layout

**File**: `AGENTS.md`

### 8a. Update section "11. Data directory layout" (lines 428-476)

Remove `gen/` from the data/ tree. Add note about kernel_lab_kb.

```markdown
data/
├── configs/            # Benchmark configs — arch-agnostic, one per kernel
├── ref/                # Reference implementations — arch-agnostic library code
└── nvidia-docs/        # Cached NVIDIA documentation

# Generated implementations live in kernel_lab_kb:
# ~/kernel_lab_kb/runs/run_<ts>/gen/<arch>/<kernel>/<impl>/
```

### 8b. Commit
```bash
git commit -m "docs: AGENTS.md — gen/ moved to kernel_lab_kb"
```

## Step 9: Update SYSTEM_DESIGN.md

**File**: `docs/SYSTEM_DESIGN.md`

### 9a. Replace agent_journal references with runs/.../journal/

Search for `agent_journal` and update all references to point to
`runs/run_<ts>/journal/`.

### 9b. Commit
```bash
git commit -m "docs: SYSTEM_DESIGN.md — agent_journal → runs/.../journal/"
```

## Step 10: Migrate existing kernel_lab_kb data

### 10a. Migrate ik_bench/runs/ to runs/

```bash
cd ~/kernel_lab_kb
mkdir -p runs

# Move each existing run
for dir in ik_bench/runs/*/*/20*; do
    ts=$(basename "$dir")
    # Restructure: move data/ contents up one level
    new_dir="runs/run_${ts}"
    mkdir -p "$new_dir"
    # Move data/gen/ → gen/, data/ref/ → ref/, data/configs/ → configs/
    if [ -d "$dir/data/gen" ]; then cp -r "$dir/data/gen" "$new_dir/gen"; fi
    if [ -d "$dir/data/ref" ]; then cp -r "$dir/data/ref" "$new_dir/ref"; fi
    if [ -d "$dir/data/configs" ]; then cp -r "$dir/data/configs" "$new_dir/configs"; fi
    # Move impls/ and command.json as-is
    if [ -d "$dir/impls" ]; then cp -r "$dir/impls" "$new_dir/impls"; fi
    if [ -f "$dir/command.json" ]; then cp "$dir/command.json" "$new_dir/command.json"; fi
done
```

### 10b. Migrate ik_bench/gems/ into corresponding runs

```bash
# For each gem, find the run it came from and move it there
# This is approximate — gems reference timestamps that match runs
```

### 10c. Remove old ik_bench/ directory

```bash
rm -rf ik_bench/
git add -A && git commit -m "migrate: ik_bench/ → runs/run_<ts>/ with flat structure"
```

## Step 11: Remove data/gen/ from kernel_lab

### 11a. Move current gen code to latest KB run as seed

```bash
# Create a seed run with current best code
cd ~/kernel_lab_kb
mkdir -p runs/run_20260406_seed/gen/sm90/matmul/cuda
cp ~/kernel_lab/data/gen/sm90/matmul/cuda/cuda.cu runs/run_20260406_seed/gen/sm90/matmul/cuda/
# Same for fa4 if exists
```

### 11b. Remove data/gen/ from kernel_lab

```bash
cd ~/kernel_lab
rm -rf data/gen/
git add -A && git commit -m "refactor: remove data/gen/ — gen code lives in kernel_lab_kb"
```

### 11c. Update .gitignore if needed

## Step 12: End-to-end verification

### 12a. Run ik:bench matmul

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m cuda_exec.formal bench.kernel=matmul
```

Verify:
- [ ] Bench creates `~/kernel_lab_kb/runs/run_<new_ts>/`
- [ ] `gen/sm90/matmul/cuda/cuda.cu` exists in the run
- [ ] `ref/matmul/cublas/cublas.py` exists in the run
- [ ] `impls/<bench_ts>/` exists with results.json + report.md
- [ ] `impls/<bench_ts>/gen/` has frozen code snapshot
- [ ] If improvement: `gems/gen-cuda/v00N/` exists
- [ ] All 6 configs pass correctness
- [ ] No references to `ik_bench/` anywhere

### 12b. Commit and push
```bash
git add -A && git commit -m "verified: ik:bench works with new KB structure"
git push
```

## Task Dependencies

| Group | Steps | Can Parallelize |
|-------|-------|-----------------|
| 1 | Steps 1-2 | No (2 depends on 1) |
| 2 | Step 3 | No (depends on 2) |
| 3 | Step 4-5 | Yes (independent) |
| 4 | Steps 6-9 | Yes (all docs, independent) |
| 5 | Step 10 | No (depends on 1-4) |
| 6 | Step 11 | No (depends on 10) |
| 7 | Step 12 | No (final verification) |
