"""Write benchmark runs and gems to kernel_lab_kb.

Two-phase flow:
1. prepare_run() — called BEFORE bench: snapshot sources + configs, write command.json
2. finalize_run() — called AFTER bench: write per-impl results, check gems

All compile/trial must use the snapshot in run_dir/data/, never the original files.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


KB_REPO = Path.home() / "kernel_lab_kb"
RUNS_DIR = KB_REPO / "ik_bench" / "runs"
GEMS_DIR = KB_REPO / "ik_bench" / "gems"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Gem thresholds — both must be exceeded for a config to count as improved
GEM_ABS_THRESHOLD_MS = 0.002   # minimum absolute improvement in milliseconds
GEM_REL_THRESHOLD = 0.002      # minimum relative improvement (0.002 = 0.2%)


# ---------------------------------------------------------------------------
# Git / device helpers
# ---------------------------------------------------------------------------

def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _device_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "--id=0"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return "unknown"


def _gpu_index() -> int:
    import os
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        try:
            return int(cvd.split(",")[0])
        except ValueError:
            pass
    return -1


# ---------------------------------------------------------------------------
# Compile info parsing
# ---------------------------------------------------------------------------

def _parse_ptxas(stderr: str) -> dict:
    info: dict[str, Any] = {}
    m = re.search(r"Used (\d+) registers", stderr)
    if m:
        info["registers"] = int(m.group(1))
    m = re.search(r"(\d+) bytes spill stores", stderr)
    if m:
        info["spill_stores"] = int(m.group(1))
    m = re.search(r"(\d+) bytes spill loads", stderr)
    if m:
        info["spill_loads"] = int(m.group(1))
    m = re.search(r"used (\d+) barriers", stderr)
    if m:
        info["barriers"] = int(m.group(1))
    m = re.search(r"(\d+) bytes gmem", stderr)
    if m:
        info["gmem"] = int(m.group(1))
    return info


def _parse_resource_usage(stdout: str) -> dict:
    info: dict[str, Any] = {}
    m = re.search(r"REG:(\d+)", stdout)
    if m:
        info["registers"] = int(m.group(1))
    m = re.search(r"SHARED:(\d+)", stdout)
    if m:
        info["shared_mem"] = int(m.group(1))
    m = re.search(r"STACK:(\d+)", stdout)
    if m:
        info["stack"] = int(m.group(1))
    return info


def _extract_compile_info(compile_result: dict) -> dict:
    info: dict[str, Any] = {"ok": compile_result.get("all_ok", False)}
    tool_outputs = compile_result.get("tool_outputs", {})

    ptxas = tool_outputs.get("ptxas", {})
    ptxas_stderr = ""
    if isinstance(ptxas, dict):
        stderr_obj = ptxas.get("stderr", {})
        if isinstance(stderr_obj, dict):
            ptxas_stderr = stderr_obj.get("content", "") or ""
    info.update(_parse_ptxas(ptxas_stderr))

    res_usage = tool_outputs.get("resource_usage", {})
    if isinstance(res_usage, dict):
        stdout_obj = res_usage.get("stdout", {})
        if isinstance(stdout_obj, dict):
            res_stdout = stdout_obj.get("content", "") or ""
            parsed = _parse_resource_usage(res_stdout)
            for k, v in parsed.items():
                if k not in info:
                    info[k] = v

    return info


def _copy_compile_logs(runtime_root: Path | None, compile_result: dict, dest: Path) -> None:
    """Copy all compile log files to dest directory.

    Strategy: if runtime_root is available, copy log files directly from the
    runtime logs/ directory (complete). Otherwise fall back to extracting from
    the Pydantic tool_outputs (partial — only what the response carries).
    """
    dest.mkdir(parents=True, exist_ok=True)

    # Try direct copy from runtime logs/
    if runtime_root and runtime_root.exists():
        # Find the compile logs directory (nested under run_tag/v1/.../turn_N/logs/)
        log_files = list(runtime_root.rglob("logs/compile.attempt_001.*"))
        for f in log_files:
            if f.is_file() and f.stat().st_size > 0:
                dest_name = f.name.replace("compile.attempt_001.", "")
                shutil.copy2(f, dest / dest_name)
        if log_files:
            return

    # Fallback: extract from tool_outputs in the Pydantic response
    tool_outputs = compile_result.get("tool_outputs", {})
    for key in ("nvcc_ptx", "ptxas", "resource_usage", "nvdisasm"):
        entry = tool_outputs.get(key, {})
        if not isinstance(entry, dict):
            continue
        for stream in ("stdout", "stderr"):
            obj = entry.get(stream, {})
            if isinstance(obj, dict):
                content = obj.get("content", "") or ""
                if content.strip():
                    (dest / f"{key}.{stream}.txt").write_text(content)


# ---------------------------------------------------------------------------
# Config results extraction
# ---------------------------------------------------------------------------

def _extract_config_results(trial_result: dict) -> dict:
    configs_out: dict[str, dict] = {}
    for slug, cfg in trial_result.get("configs", {}).items():
        ref_perf = cfg.get("reference", {}).get("performance", {})
        gen_perf = cfg.get("generated", {}).get("performance", {})
        cudnn_perf = cfg.get("cudnn", {}).get("performance", {})
        gen_correct = cfg.get("generated", {}).get("correctness", {})

        ref_median = ref_perf.get("latency_ms", {}).get("median")
        gen_median = gen_perf.get("latency_ms", {}).get("median")
        cudnn_median = cudnn_perf.get("latency_ms", {}).get("median")

        speedup = None
        if ref_median and gen_median and gen_median > 0:
            speedup = round(ref_median / gen_median, 3)

        entry: dict[str, Any] = {
            "correctness": gen_correct.get("passed", False),
            "ref_median_ms": ref_median,
            "gen_median_ms": gen_median,
            "speedup": speedup,
            "ref_latency": ref_perf.get("latency_ms", {}),
            "gen_latency": gen_perf.get("latency_ms", {}),
        }
        if cudnn_median is not None:
            entry["cudnn_median_ms"] = cudnn_median
            entry["cudnn_latency"] = cudnn_perf.get("latency_ms", {})
        if gen_correct.get("max_abs_error") is not None:
            entry["max_abs_error"] = gen_correct["max_abs_error"]

        configs_out[slug] = entry

    return configs_out


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_report(results: dict, gem_info: dict | None = None) -> str:
    kernel = results["kernel"]
    arch = results["arch"]
    impl = results["impl"]
    ts = results["timestamp"]
    device = results["device"]
    gpu = results["gpu_index"]
    commit = results["git_commit"]
    compile_info = results.get("compile", {})

    lines = [
        f"# {kernel}/{arch}/{impl} — {ts}",
        "",
        f"**Device:** {device} (GPU {gpu})",
        f"**Commit:** {commit}",
    ]

    compile_parts = []
    if "registers" in compile_info:
        compile_parts.append(f"{compile_info['registers']} regs")
    if compile_info.get("spill_stores", 0) > 0 or compile_info.get("spill_loads", 0) > 0:
        compile_parts.append(f"{compile_info.get('spill_stores', 0)}/{compile_info.get('spill_loads', 0)} spills")
    else:
        compile_parts.append("0 spills")
    if "shared_mem" in compile_info:
        compile_parts.append(f"{compile_info['shared_mem']}B shared")
    if compile_parts:
        lines.append(f"**Compile:** {', '.join(compile_parts)}")

    if gem_info:
        improved = gem_info.get("improved_configs", [])
        lines.append(f"**Gem:** {len(improved)} config(s) improved — {', '.join(improved)}")

    lines.append("")

    # Check if any config has cudnn data
    configs = results.get("configs", {})
    has_cudnn = any(cfg.get("cudnn_median_ms") is not None for cfg in configs.values())

    header = "| Config | Correct | Ref (ms) | Gen (ms) |"
    sep = "|--------|---------|----------|----------|"
    if has_cudnn:
        header += " cuDNN (ms) |"
        sep += "------------|"
    header += " Speedup |"
    sep += "---------|"

    lines.append(header)
    lines.append(sep)

    for slug, cfg in configs.items():
        correct = "\u2713" if cfg.get("correctness") else "\u2717"
        ref_ms = cfg.get("ref_median_ms")
        gen_ms = cfg.get("gen_median_ms")
        cudnn_ms = cfg.get("cudnn_median_ms")
        speedup = cfg.get("speedup")

        ref_str = f"{ref_ms:.4f}" if ref_ms is not None else "N/A"
        gen_str = f"{gen_ms:.4f}" if gen_ms is not None else "N/A"
        spd_str = f"{speedup:.2f}\u00d7" if speedup is not None else "N/A"

        row = f"| {slug} | {correct} | {ref_str} | {gen_str} |"
        if has_cudnn:
            cudnn_str = f"{cudnn_ms:.4f}" if cudnn_ms is not None else "N/A"
            row += f" {cudnn_str} |"
        row += f" {spd_str} |"

        lines.append(row)

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gem logic
# ---------------------------------------------------------------------------

def _next_gem_version(gem_base: Path) -> int:
    if not gem_base.exists():
        return 1
    max_ver = 0
    for d in gem_base.iterdir():
        if d.is_dir() and d.name.startswith("v"):
            try:
                ver = int(d.name.split("_", 1)[0][1:])
                max_ver = max(max_ver, ver)
            except (ValueError, IndexError):
                pass
    return max_ver + 1


def _load_latest_gem_results(gem_base: Path) -> dict | None:
    """Load results.json from the most recent gem."""
    if not gem_base.exists():
        return None
    versions = sorted(
        [d.name for d in gem_base.iterdir() if d.is_dir() and d.name.startswith("v")],
        reverse=True,
    )
    for ver_dir in versions:
        results_file = gem_base / ver_dir / "results.json"
        if results_file.exists():
            try:
                return json.loads(results_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
    return None


def _check_gem(current_configs: dict, gem_base: Path) -> dict | None:
    """Check if any config beats the latest gem. Returns gem_info or None."""
    previous = _load_latest_gem_results(gem_base)

    if previous is None:
        return {
            "previous_run": None,
            "improved_configs": list(current_configs.keys()),
            "summary": f"first run — {len(current_configs)} configs",
        }

    prev_configs = previous.get("configs", {})
    improved: list[str] = []

    for slug, cfg in current_configs.items():
        if not cfg.get("correctness", False):
            continue
        gen_ms = cfg.get("gen_median_ms")
        if gen_ms is None:
            continue

        prev = prev_configs.get(slug, {})
        prev_ms = prev.get("gen_median_ms")

        if prev_ms is None:
            improved.append(slug)
        elif prev_ms > 0:
            abs_diff = prev_ms - gen_ms
            rel_diff = abs_diff / prev_ms
            if abs_diff > GEM_ABS_THRESHOLD_MS and rel_diff > GEM_REL_THRESHOLD:
                improved.append(slug)

    if not improved:
        return None

    prev_ts = previous.get("timestamp", "unknown")
    return {
        "previous_run": prev_ts,
        "improved_configs": improved,
        "summary": f"{len(improved)}/{len(current_configs)} configs improved vs {prev_ts}",
    }


# ---------------------------------------------------------------------------
# Phase 1: prepare_run() — BEFORE benchmark
# ---------------------------------------------------------------------------

def prepare_run(
    kernel: str,
    arch: str,
    impls: str | list[str],
    timeout_seconds: int,
    *,
    kb_repo: Path | None = None,
) -> Path:
    """Create run dir, snapshot sources + configs. Returns run_dir path.

    Call this BEFORE compile/trial. The snapshot in run_dir/data/ is the
    canonical source for all subsequent resolve_impls/load_configs calls.
    """
    repo = kb_repo or KB_REPO
    if not repo.exists():
        raise FileNotFoundError(f"kernel_lab_kb not found at {repo}")

    runs_dir = repo / "ik_bench" / "runs"
    ts = datetime.now()
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / kernel / arch / ts_str
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

    # 2. Snapshot data/ref/<kernel>/
    ref_src = PROJECT_ROOT / "data" / "ref" / kernel
    ref_dst = run_dir / "data" / "ref" / kernel
    if ref_src.exists():
        shutil.copytree(ref_src, ref_dst)

    # 3. Snapshot data/gen/<arch>/<kernel>/
    gen_src = PROJECT_ROOT / "data" / "gen" / arch / kernel
    gen_dst = run_dir / "data" / "gen" / arch / kernel
    if gen_src.exists():
        shutil.copytree(gen_src, gen_dst)

    # 4. Snapshot data/configs/<kernel>.json
    cfg_src = PROJECT_ROOT / "data" / "configs" / f"{kernel}.json"
    cfg_dst = run_dir / "data" / "configs" / f"{kernel}.json"
    cfg_dst.parent.mkdir(parents=True, exist_ok=True)
    if cfg_src.exists():
        shutil.copy2(cfg_src, cfg_dst)

    return run_dir


# ---------------------------------------------------------------------------
# Phase 2: finalize_run() — AFTER benchmark
# ---------------------------------------------------------------------------

def finalize_run(run_dir: Path, bench_result: dict, *, kb_repo: Path | None = None, runtime_root: Path | None = None) -> dict:
    """Write per-impl results + check gems after bench completes.

    Returns a dict of impl_slug -> gem_info for impls that set a new gem,
    or an empty dict if no improvements were found.
    """
    repo = kb_repo or KB_REPO
    gems_dir = repo / "ik_bench" / "gems"
    kernel = bench_result["kernel"]
    arch = bench_result["arch"]
    ts_str = run_dir.name
    new_gems: dict[str, dict] = {}

    # Load command.json for metadata
    cmd = json.loads((run_dir / "command.json").read_text())

    for impl_slug, impl_data in bench_result.get("results", {}).items():
        if impl_data.get("compile_ok") is None:
            continue

        impl_dir = run_dir / "impls" / impl_slug
        impl_dir.mkdir(parents=True, exist_ok=True)

        # Write compile logs
        compile_result = impl_data.get("compile_result")
        if compile_result:
            _copy_compile_logs(runtime_root, compile_result, impl_dir / "compile")

        # Build results.json
        compile_info = _extract_compile_info(compile_result) if compile_result else {"ok": False}
        trial_result = impl_data.get("trial_result", {})
        config_results = _extract_config_results(trial_result) if trial_result else {}

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
        gem_base = gems_dir / kernel / arch / impl_slug
        gem_info = _check_gem(config_results, gem_base)
        if gem_info:
            ver = _next_gem_version(gem_base)
            gem_dir_name = f"v{ver:03d}_{ts_str}"
            gem_info["version"] = ver
            new_gems[impl_slug] = gem_info
            gem_dir = gem_base / gem_dir_name
            gem_dir.mkdir(parents=True, exist_ok=True)

            # Copy snapshot sources to gem
            snapshot_data = run_dir / "data"
            for subdir in ("ref", "gen", "configs"):
                src = snapshot_data / subdir
                dst = gem_dir / "data" / subdir
                if src.exists():
                    if src.is_dir():
                        shutil.copytree(src, dst)
                    else:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)

            # Copy compile info to gem
            if compile_result:
                _copy_compile_logs(runtime_root, compile_result, gem_dir / "compile")

            # Write gem results.json with gem metadata
            gem_results = {**results, "gem": gem_info}
            (gem_dir / "results.json").write_text(json.dumps(gem_results, indent=2) + "\n")

            # Write gem report.md
            gem_report = _generate_report(gem_results, gem_info)
            (gem_dir / "report.md").write_text(gem_report)

    _auto_commit(repo, kernel, arch, ts_str)

    return new_gems


def _auto_commit(repo: Path, kernel: str, arch: str, ts_str: str) -> None:
    try:
        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(repo), capture_output=True, timeout=10,
        )
        subprocess.run(
            ["git", "commit", "-m", f"bench: {kernel}/{arch} {ts_str}"],
            cwd=str(repo), capture_output=True, timeout=10,
        )
    except Exception:
        pass
