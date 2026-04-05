"""Write benchmark runs and gems to kernel_lab_kb.

Called by formal.py after each benchmark run to persist:
- runs/: every benchmark result (sources, compile info, results.json, report.md)
- gems/: only runs where at least one config beats the previous best
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


def _device_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
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


def _extract_config_results(trial_result: dict) -> dict:
    configs_out: dict[str, dict] = {}
    for slug, cfg in trial_result.get("configs", {}).items():
        ref_perf = cfg.get("reference", {}).get("performance", {})
        gen_perf = cfg.get("generated", {}).get("performance", {})
        gen_correct = cfg.get("generated", {}).get("correctness", {})

        ref_median = ref_perf.get("latency_ms", {}).get("median")
        gen_median = gen_perf.get("latency_ms", {}).get("median")

        speedup = None
        if ref_median and gen_median and gen_median > 0:
            speedup = round(ref_median / gen_median, 3)

        configs_out[slug] = {
            "correctness": gen_correct.get("passed", False),
            "ref_median_ms": ref_median,
            "gen_median_ms": gen_median,
            "speedup": speedup,
            "ref_latency": ref_perf.get("latency_ms", {}),
            "gen_latency": gen_perf.get("latency_ms", {}),
        }
        if gen_correct.get("max_abs_error") is not None:
            configs_out[slug]["max_abs_error"] = gen_correct["max_abs_error"]

    return configs_out


def _load_latest_results(base_dir: Path) -> dict | None:
    """Load results.json from the most recent timestamped subdirectory."""
    if not base_dir.exists():
        return None
    timestamps = sorted(
        [d.name for d in base_dir.iterdir() if d.is_dir()],
        reverse=True,
    )
    for ts in timestamps:
        results_file = base_dir / ts / "results.json"
        if results_file.exists():
            try:
                return json.loads(results_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
    return None


def _generate_report(results: dict, previous: dict | None = None, gem_info: dict | None = None) -> str:
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

    prev_configs = previous.get("configs", {}) if previous else {}
    has_previous = bool(prev_configs)

    header = "| Config | Correct | Ref (ms) | Gen (ms) | Speedup |"
    sep = "|--------|---------|----------|----------|---------|"
    if has_previous:
        header += " Prev (ms) | Delta |"
        sep += "-----------|-------|"

    lines.append(header)
    lines.append(sep)

    configs = results.get("configs", {})
    for slug, cfg in configs.items():
        correct = "\u2713" if cfg.get("correctness") else "\u2717"
        ref_ms = cfg.get("ref_median_ms")
        gen_ms = cfg.get("gen_median_ms")
        speedup = cfg.get("speedup")

        ref_str = f"{ref_ms:.4f}" if ref_ms is not None else "N/A"
        gen_str = f"{gen_ms:.4f}" if gen_ms is not None else "N/A"
        spd_str = f"{speedup:.2f}\u00d7" if speedup is not None else "N/A"

        row = f"| {slug} | {correct} | {ref_str} | {gen_str} | {spd_str} |"

        if has_previous:
            prev = prev_configs.get(slug, {})
            prev_ms = prev.get("gen_median_ms")
            if prev_ms is not None and gen_ms is not None and prev_ms > 0:
                delta_pct = (prev_ms - gen_ms) / prev_ms * 100
                sign = "+" if delta_pct >= 0 else ""
                row += f" {prev_ms:.4f} | {sign}{delta_pct:.1f}% |"
            else:
                row += " N/A | N/A |"

        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def _copy_sources(kernel: str, arch: str, impl_slug: str, dest: Path) -> None:
    ref_dir = PROJECT_ROOT / "data" / "ref" / kernel
    gen_dir = PROJECT_ROOT / "data" / "gen" / arch / kernel

    ref_dest = dest / "reference"
    if ref_dir.exists():
        ref_dest.mkdir(parents=True, exist_ok=True)
        for f in ref_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, ref_dest / f.name)

    gen_dest = dest / "generated"
    if gen_dir.exists():
        gen_dest.mkdir(parents=True, exist_ok=True)
        for f in gen_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, gen_dest / f.name)


def _copy_compile_text(compile_result: dict, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    tool_outputs = compile_result.get("tool_outputs", {})

    for key in ("ptxas", "resource_usage"):
        entry = tool_outputs.get(key, {})
        if not isinstance(entry, dict):
            continue
        for stream in ("stdout", "stderr"):
            obj = entry.get(stream, {})
            if isinstance(obj, dict):
                content = obj.get("content", "") or ""
                if content.strip():
                    (dest / f"{key}.{stream}.txt").write_text(content)


def _next_gem_version(gem_base: Path) -> int:
    """Return the next gem version number (1-indexed) by scanning existing directories."""
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


def _check_gem(current_configs: dict, gem_base: Path) -> dict | None:
    """Check if any config in this run beats the latest gem.

    Returns gem_info dict if this is a new gem, None otherwise.
    First run for a kernel/arch/impl is always a gem.
    """
    previous = _load_latest_results(gem_base)

    # First run is always a gem
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


def _write_single_run(
    kernel: str, arch: str, impl_slug: str, impl_data: dict,
    ts: datetime, ts_str: str, base_dir: Path,
    gem_info: dict | None = None,
    dir_name: str | None = None,
) -> tuple[Path, dict]:
    """Write one run or gem directory. Returns (path, results_dict)."""
    run_dir = base_dir / kernel / arch / impl_slug / (dir_name or ts_str)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy source files
    _copy_sources(kernel, arch, impl_slug, run_dir / "sources")

    # 2. Copy compile text files
    compile_result = impl_data.get("compile_result")
    if compile_result:
        _copy_compile_text(compile_result, run_dir / "compile")

    # 3. Build results.json
    compile_info = _extract_compile_info(compile_result) if compile_result else {"ok": False}
    trial_result = impl_data.get("trial_result", {})
    config_results = _extract_config_results(trial_result) if trial_result else {}

    results = {
        "kernel": kernel,
        "arch": arch,
        "impl": impl_slug,
        "timestamp": ts.isoformat(timespec="seconds"),
        "git_commit": _git_commit_hash(),
        "git_branch": _git_branch(),
        "device": _device_name(),
        "gpu_index": _gpu_index(),
        "compile": compile_info,
        "configs": config_results,
    }
    if gem_info:
        results["gem"] = gem_info

    (run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    # 4. Generate report.md
    run_base = RUNS_DIR / kernel / arch / impl_slug
    previous = _load_latest_results(run_base)
    report = _generate_report(results, previous, gem_info)
    (run_dir / "report.md").write_text(report)

    return run_dir, results


def write_trajectory(
    bench_result: dict,
    *,
    timestamp: datetime | None = None,
) -> Path | None:
    """Write benchmark run + gem (if applicable) to kernel_lab_kb.

    Returns:
        Path to the run directory, or None if KB repo not found.
    """
    if not KB_REPO.exists():
        return None

    ts = timestamp or datetime.now()
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    kernel = bench_result["kernel"]
    arch = bench_result["arch"]

    written_paths: list[Path] = []

    for impl_slug, impl_data in bench_result.get("results", {}).items():
        if impl_data.get("compile_ok") is None:
            continue

        # Write run (always)
        run_path, results = _write_single_run(
            kernel, arch, impl_slug, impl_data, ts, ts_str, RUNS_DIR,
        )
        written_paths.append(run_path)

        # Check if this is a gem
        gem_base = GEMS_DIR / kernel / arch / impl_slug
        gem_info = _check_gem(results.get("configs", {}), gem_base)

        if gem_info:
            ver = _next_gem_version(gem_base)
            gem_dir_name = f"v{ver:03d}_{ts_str}"
            gem_info["version"] = ver
            _write_single_run(
                kernel, arch, impl_slug, impl_data, ts, ts_str, GEMS_DIR,
                gem_info=gem_info, dir_name=gem_dir_name,
            )

    # Auto-commit
    if written_paths:
        _auto_commit(kernel, arch, ts_str)

    return written_paths[0] if written_paths else None


def _auto_commit(kernel: str, arch: str, ts_str: str) -> None:
    try:
        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(KB_REPO), capture_output=True, timeout=10,
        )
        subprocess.run(
            ["git", "commit", "-m", f"bench: {kernel}/{arch} {ts_str}"],
            cwd=str(KB_REPO), capture_output=True, timeout=10,
        )
    except Exception:
        pass
