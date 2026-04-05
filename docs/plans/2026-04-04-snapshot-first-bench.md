# Plan: Snapshot-First ik:bench Architecture

**Goal**: Every ik:bench run snapshots source files before compile/trial, and all compile/trial runs against the snapshot — never the original files.
**Architecture**: Two-phase trajectory (prepare → finalize), `resolve_impls` reads from snapshot dir, run directory restructured to per-invocation layout.
**Tech Stack**: Python, cuda_exec, kernel_lab_kb repo

## Current State

- `formal.py`: calls `resolve_impls()` → reads from `data/ref/`, `data/gen/` → passes file contents to `CompileRequest`
- `trajectory.py`: single-phase `write_trajectory()` called after bench completes
- `impls.py`: `resolve_impl()` hardcodes `_PROJECT_ROOT / "data" / "ref"` and `_PROJECT_ROOT / "data" / "gen"`

## New Run Directory Structure

```
runs/<kernel>/<arch>/<YYYYMMDD_HHMMSS>/
  command.json              # invocation parameters
  data/
    ref/<kernel>/           # snapshot of data/ref/<kernel>/
    gen/<arch>/<kernel>/    # snapshot of data/gen/<arch>/<kernel>/
    configs/<kernel>.json   # snapshot of data/configs/<kernel>.json
  impls/
    gen-cuda/
      compile/
      results.json
      report.md
    gen-cutedsl/
      results.json
      report.md
```

Gems use identical structure under `gems/` with versioned dir names.

## Dependency Table

| Group | Steps | Can Parallelize |
|-------|-------|-----------------|
| 1 | Step 1 (restructure trajectory.py) | No |
| 2 | Step 2 (add ref_dir/gen_dir to impls.py) | No |
| 3 | Step 3 (rewrite formal.py) | No (depends on 1+2) |
| 4 | Step 4 (end-to-end test) | No (depends on 3) |

---

## Step 1: Restructure trajectory.py — prepare + finalize

**File**: `cuda_exec/trajectory.py`

### 1a. Add `prepare_run()` function

Creates run directory, writes `command.json`, snapshots source files and configs.

```python
def prepare_run(
    kernel: str,
    arch: str,
    impls: str | list[str],
    timeout_seconds: int,
) -> Path:
    """Create run dir, snapshot sources + configs. Returns run_dir path."""
    if not KB_REPO.exists():
        raise FileNotFoundError(f"kernel_lab_kb not found at {KB_REPO}")

    ts = datetime.now()
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / kernel / arch / ts_str
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write command.json
    command = {
        "kernel": kernel,
        "arch": arch,
        "impls": impls,
        "timeout_seconds": timeout_seconds,
        "timestamp": ts.isoformat(timespec="seconds"),
        "git_commit": _git_commit_hash(),
        "git_branch": _git_branch(),
        "device": _device_name(),
        "gpu_index": _gpu_index(),
    }
    (run_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")

    # 2. Snapshot data/ref/<kernel>/ → run_dir/data/ref/<kernel>/
    ref_src = PROJECT_ROOT / "data" / "ref" / kernel
    ref_dst = run_dir / "data" / "ref" / kernel
    if ref_src.exists():
        shutil.copytree(ref_src, ref_dst)

    # 3. Snapshot data/gen/<arch>/<kernel>/ → run_dir/data/gen/<arch>/<kernel>/
    gen_src = PROJECT_ROOT / "data" / "gen" / arch / kernel
    gen_dst = run_dir / "data" / "gen" / arch / kernel
    if gen_src.exists():
        shutil.copytree(gen_src, gen_dst)

    # 4. Snapshot data/configs/<kernel>.json → run_dir/data/configs/<kernel>.json
    cfg_src = PROJECT_ROOT / "data" / "configs" / f"{kernel}.json"
    cfg_dst = run_dir / "data" / "configs" / f"{kernel}.json"
    cfg_dst.parent.mkdir(parents=True, exist_ok=True)
    if cfg_src.exists():
        shutil.copy2(cfg_src, cfg_dst)

    return run_dir
```

### 1b. Rewrite `finalize_run()` — per-impl results under `impls/`

Replaces current `write_trajectory()`. Takes the run_dir + bench_result, writes per-impl results.

```python
def finalize_run(run_dir: Path, bench_result: dict) -> None:
    """Write per-impl results + check gems after bench completes."""
    kernel = bench_result["kernel"]
    arch = bench_result["arch"]
    ts_str = run_dir.name  # e.g., "20260404_220000"

    for impl_slug, impl_data in bench_result.get("results", {}).items():
        if impl_data.get("compile_ok") is None:
            continue

        impl_dir = run_dir / "impls" / impl_slug
        impl_dir.mkdir(parents=True, exist_ok=True)

        # Write compile info
        compile_result = impl_data.get("compile_result")
        if compile_result:
            _copy_compile_text(compile_result, impl_dir / "compile")

        # Build + write results.json
        compile_info = _extract_compile_info(compile_result) if compile_result else {"ok": False}
        trial_result = impl_data.get("trial_result", {})
        config_results = _extract_config_results(trial_result) if trial_result else {}

        # Load command.json for metadata
        cmd = json.loads((run_dir / "command.json").read_text())

        results = {
            "kernel": kernel,
            "arch": arch,
            "impl": impl_slug,
            "timestamp": cmd["timestamp"],
            "git_commit": cmd["git_commit"],
            "git_branch": cmd["git_branch"],
            "device": cmd["device"],
            "gpu_index": cmd["gpu_index"],
            "compile": compile_info,
            "configs": config_results,
        }
        (impl_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")

        # Generate report.md
        report = _generate_report(results)
        (impl_dir / "report.md").write_text(report)

        # Check gem
        gem_base = GEMS_DIR / kernel / arch / impl_slug
        gem_info = _check_gem(config_results, gem_base)
        if gem_info:
            ver = _next_gem_version(gem_base)
            gem_dir_name = f"v{ver:03d}_{ts_str}"
            gem_dir = gem_base / gem_dir_name
            # Copy entire run_dir to gem (or just the impl portion + sources)
            _write_gem(gem_dir, run_dir, impl_slug, results, gem_info)

    _auto_commit(kernel, arch, ts_str)
```

### 1c. Remove old `write_trajectory()` and `_write_single_run()`

These are replaced by `prepare_run()` + `finalize_run()`.

### 1d. Remove `_copy_sources()` 

No longer needed — `prepare_run()` handles snapshotting via `shutil.copytree()`.

### 1e. Update `_generate_report()` signature

Remove `previous` parameter — comparison is now done against latest gem, not latest run.

---

## Step 2: Add snapshot-aware resolution to impls.py

**File**: `cuda_exec/impls.py`

### 2a. Add `data_root` parameter to resolution functions

```python
def resolve_impl(kernel, arch, impl_slug, *, data_root=None):
    root = Path(data_root) if data_root else _PROJECT_ROOT / "data"
    ref_base = root / "ref" / kernel
    gen_base = root / "gen" / arch / kernel
    # ... rest unchanged, just use ref_base/gen_base instead of _ref_dir/_gen_dir
```

Same for `list_impls()`, `resolve_impls()`, `load_configs()`.

### 2b. Verify backward compatibility

When `data_root=None` (default), behavior is identical to current code.

---

## Step 3: Rewrite formal.py to use snapshot-first flow

**File**: `cuda_exec/formal.py`

### 3a. New flow

```python
def formal_benchmark(kernel, arch, *, impls="all", timeout_seconds=300):
    # 1. Prepare: snapshot to kernel_lab_kb
    from cuda_exec.trajectory import prepare_run, finalize_run
    run_dir = prepare_run(kernel, arch, impls, timeout_seconds)
    snapshot_data = run_dir / "data"

    # 2. Resolve impls from SNAPSHOT (not original)
    configs = load_configs(kernel, data_root=snapshot_data)
    resolved = resolve_impls(kernel, arch, impls, data_root=snapshot_data)

    # 3. Compile + trial as before (using snapshot file contents)
    ...

    # 4. Finalize: write results + check gems
    finalize_run(run_dir, bench_result)

    return bench_result
```

### 3b. Fallback if KB repo not found

If `~/kernel_lab_kb` doesn't exist, fall back to reading from original `data/` (current behavior). Log a warning.

---

## Step 4: End-to-end verification

### 4a. Clean KB repo and run bench

```bash
cd ~/kernel_lab_kb && rm -rf ik_bench/runs/* ik_bench/gems/*
cd ~/kernel_lab && CUDA_VISIBLE_DEVICES=4 LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH \
  .venv/bin/python -m cuda_exec.formal bench.kernel=vecadd bench.arch=sm90
```

### 4b. Verify snapshot exists before results

```bash
# Should see: command.json, data/ref/, data/gen/, data/configs/ 
# AND impls/gen-cuda/results.json, report.md
find ~/kernel_lab_kb/ik_bench/runs/vecadd/sm90/ -type f | sort
```

### 4c. Verify gem created with v001

```bash
find ~/kernel_lab_kb/ik_bench/gems/ -type d -name "v*" | sort
```

### 4d. Run bench again — verify no new gem (noise filtered)

```bash
CUDA_VISIBLE_DEVICES=4 LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH \
  .venv/bin/python -m cuda_exec.formal bench.kernel=vecadd bench.arch=sm90
# Should have 2 runs, still 1 gem
```

---

## Potential Conflicts

1. **Gem directory structure change** — gems currently mirror old `runs/<kernel>/<arch>/<impl>/<ts>/` layout. New layout puts impls inside the run dir. For gems, we keep per-impl dirs since each gem tracks one impl's progression: `gems/<kernel>/<arch>/<impl_slug>/v001_<ts>/` with sources + compile + results.json + report.md.

2. **`_load_latest_results()` path change** — currently scans timestamp dirs for results.json. For runs, results.json moves to `impls/<impl>/results.json`. For gems, stays at top level. Need to update the scan function.

3. **Existing KB data** — old runs in kernel_lab_kb use the old layout. Clean them out before deploying.

4. **formal.py fallback** — if KB repo missing, need to gracefully fall back to original data/ paths so bench still works without kernel_lab_kb.
