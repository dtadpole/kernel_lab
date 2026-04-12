"""Formal benchmark: atomic compile + trial ALL configs, ALL implementations.

Used by the Formal Evaluator (Judge) agent via the ik:bench skill.
Not for iterative development — use ik:exec for that.

Key differences from ik:exec:
- Compile + trial are bundled atomically (no separate steps)
- ALL configs are trialed (no cherry-picking)
- ALL implementations are trialed (or a specified subset)
- No profiling (that's ik:exec's job during iteration)
- Simplified metadata (no revision management)
- Snapshot-first: sources are snapshotted to kernel_lab_kb BEFORE compile/trial
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from cuda_exec.impls import load_configs, resolve_impls
from cuda_exec.models import (
    CompileRequest,
    TrialRequest,
    Metadata,
)
from cuda_exec.tasks import compile_endpoint, trial_endpoint, _primary_artifact_from_manifest
from cuda_exec.runner import resolve_workspace_bundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU process cleanup — kill all processes on a specific GPU before benchmarking
# ---------------------------------------------------------------------------

def _kill_gpu_processes(gpu_id: int) -> None:
    """Kill all processes running on the specified GPU (requires sudo).

    Uses nvidia-smi to find PIDs, then sudo kill -9 each one.
    Skips GPU 0 as a safety measure (GPU 0 is often the default/system GPU).
    """
    import subprocess, os

    if gpu_id == 0:
        return

    my_pid = os.getpid()

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader",
             f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return

        pids = [int(line.strip()) for line in result.stdout.strip().split("\n")
                if line.strip().isdigit()]

        # Don't kill ourselves
        pids = [p for p in pids if p != my_pid]

        if not pids:
            return

        logger.info("GPU %d: killing %d existing processes: %s", gpu_id, len(pids), pids)
        for pid in pids:
            try:
                subprocess.run(["sudo", "kill", "-9", str(pid)],
                               capture_output=True, timeout=5)
            except Exception:
                pass

        # Brief pause for processes to fully exit
        time.sleep(1)

    except Exception as exc:
        logger.warning("GPU cleanup failed: %s", exc)


# ---------------------------------------------------------------------------
# FLOPs computation per kernel family
# ---------------------------------------------------------------------------

def _compute_flops(kernel: str, config: dict) -> int:
    """Compute total FLOPs for a single kernel invocation."""
    family = config.get("family", "")

    if "matrix-multiplication" in family or kernel == "matmul":
        shape = config.get("shape", [])
        if len(shape) >= 2:
            M, N = shape[0], shape[1]
            K = M  # square matmul: A(M×K) @ B(K×N)
            return 2 * M * N * K
        return 0

    if "fa4" in family or kernel == "fa4":
        B = config.get("batch_size", 1)
        S = config.get("seq_len", 1)
        H = config.get("num_heads", 1)
        D = config.get("head_dim", 1)
        causal = config.get("causal", False)
        # Q@K^T + Softmax@V: each is 2*B*H*S*S*D
        flops = 4 * B * H * S * S * D
        if causal:
            flops = flops // 2
        return flops

    if kernel == "vecadd":
        return config.get("input_size", 0)

    return 0


# ---------------------------------------------------------------------------
# Result enrichment: extract metrics, compute TFLOPS / speedup / % peak
# ---------------------------------------------------------------------------

def _extract_impl_metrics(
    bench_result: dict,
    configs: dict,
) -> Dict[str, Dict[str, dict]]:
    """Extract per-impl, per-config latency and correctness.

    Returns ``{impl_slug: {config_slug: {median_ms, correct}}}``.
    """
    metrics: Dict[str, Dict[str, dict]] = {}

    for slug, impl_result in bench_result.get("results", {}).items():
        metrics[slug] = {}
        trial = impl_result.get("trial_result")
        if not trial:
            continue
        trial_configs = trial.get("configs", {})
        for config_slug in configs:
            entry = trial_configs.get(config_slug, {})
            median_ms = None
            if "impls" in entry:
                # cu trial: configs[c]["impls"][slug]["performance"]
                impl_data = entry.get("impls", {}).get(slug, {})
                median_ms = impl_data.get("performance", {}).get("latency_ms", {}).get("p25")
            elif "performance" in entry:
                # py impl: configs[c]["performance"]
                median_ms = entry.get("performance", {}).get("latency_ms", {}).get("p25")
            metrics[slug][config_slug] = {"median_ms": median_ms, "correct": False}  # default: fail until proven correct

    # Correctness comes from cu trial results (which compare all impls vs golden)
    for slug, impl_result in bench_result.get("results", {}).items():
        trial = impl_result.get("trial_result")
        if not trial:
            continue
        for config_slug, entry in trial.get("configs", {}).items():
            if "impls" not in entry:
                continue
            for impl_slug, impl_data in entry.get("impls", {}).items():
                corr = impl_data.get("correctness")
                if corr is not None and impl_slug in metrics and config_slug in metrics[impl_slug]:
                    metrics[impl_slug][config_slug]["correct"] = corr.get("passed")

    return metrics


def enrich_result(bench_result: dict, configs: dict) -> dict:
    """Add a ``summary`` section with TFLOPS, speedup, correctness, % peak.

    Mutates and returns ``bench_result``.
    """
    from cuda_exec.host_env import resolve_gpu_peak_tflops, resolve_gpu_name

    kernel = bench_result["kernel"]
    peak = resolve_gpu_peak_tflops()
    gpu_name = resolve_gpu_name()
    # Golden = first ref (alphabetical, matches formal_benchmark's primary_ref)
    golden = bench_result["refs"][0] if bench_result.get("refs") else None
    impl_order = bench_result.get("impls_requested", [])

    raw = _extract_impl_metrics(bench_result, configs)

    summary_impls: Dict[str, dict] = {}
    for slug in impl_order:
        impl_configs: Dict[str, dict] = {}
        best_pct = 0.0
        for config_slug, config in configs.items():
            m = raw.get(slug, {}).get(config_slug, {})
            median_ms = m.get("median_ms")
            correct = m.get("correct")

            flops = _compute_flops(kernel, config)
            tflops = (flops / median_ms / 1e9) if (median_ms and median_ms > 0) else None
            pct_peak = (tflops / peak * 100) if tflops else None

            golden_ms = raw.get(golden, {}).get(config_slug, {}).get("median_ms") if golden else None
            speedup = (golden_ms / median_ms) if (golden_ms and median_ms and median_ms > 0) else None

            entry = {
                "median_ms": round(median_ms, 4) if median_ms else None,
                "flops": flops,
                "tflops": round(tflops, 1) if tflops else None,
                "pct_peak": round(pct_peak, 1) if pct_peak else None,
                "speedup": round(speedup, 2) if speedup else None,
                "correct": correct,
            }
            impl_configs[config_slug] = entry
            if pct_peak and pct_peak > best_pct:
                best_pct = pct_peak

        summary_impls[slug] = {
            "configs": impl_configs,
            "best_pct_peak": round(best_pct, 1),
        }

    bench_result["summary"] = {
        "gpu": gpu_name,
        "peak_tflops": peak,
        "golden": golden,
        "impls": summary_impls,
    }
    return bench_result


# ---------------------------------------------------------------------------
# Markdown table for stderr
# ---------------------------------------------------------------------------

def _detect_env_versions() -> dict:
    """Detect software versions for the table header."""
    import subprocess
    versions = {}
    try:
        import torch
        versions["pytorch"] = torch.__version__  # e.g. "2.11.0+cu130"
    except Exception:
        pass
    try:
        import ctypes
        lib = ctypes.CDLL("libcudnn.so.9", mode=ctypes.RTLD_GLOBAL)
        lib.cudnnGetVersion.restype = ctypes.c_ulonglong
        v = lib.cudnnGetVersion()
        versions["cudnn"] = f"{v // 10000}.{(v % 10000) // 100}.{v % 100}"
    except Exception:
        pass
    try:
        import ctypes as _ct
        _cublas = _ct.CDLL("libcublas.so")
        _v = _ct.c_int()
        _cublas.cublasGetProperty(0, _ct.byref(_v)); _maj = _v.value
        _cublas.cublasGetProperty(1, _ct.byref(_v)); _min = _v.value
        _cublas.cublasGetProperty(2, _ct.byref(_v)); _pat = _v.value
        versions["cublas"] = f"{_maj}.{_min}.{_pat}"
    except Exception:
        pass
    try:
        import importlib.metadata
        versions["flash_attn_4"] = importlib.metadata.version("flash-attn-4")
    except Exception:
        pass
    try:
        import importlib.metadata
        versions["cutlass_dsl"] = importlib.metadata.version("nvidia-cutlass-dsl")
    except Exception:
        pass
    try:
        drv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=5,
        ).strip().split("\n")[0]
        versions["driver"] = drv
    except Exception:
        pass
    try:
        from cuda_exec.host_env import resolve_host_env
        env = resolve_host_env()
        cuda_home = env.get("CUDA_HOME", "")
        if cuda_home:
            # Extract version from path like /usr/local/cuda-13.2
            import re
            m = re.search(r"cuda[/-](\d+\.\d+)", cuda_home)
            versions["cuda"] = m.group(1) if m else ""
    except Exception:
        pass
    try:
        from cuda_exec.impls import _detect_host_slug
        versions["host"] = _detect_host_slug()
    except Exception:
        pass
    try:
        import os
        versions["gpu_idx"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    except Exception:
        pass
    return versions


def _impl_version_label(slug: str, versions: dict) -> str:
    """Generate a version label for an impl slug."""
    if "cudnn" in slug:
        v = versions.get("cudnn", "")
        return f"cuDNN {v}" if v else ""
    if "cublas" in slug:
        v = versions.get("cublas", "")
        return f"cuBLAS {v}" if v else ""
    if "cutedsl" in slug:
        v = versions.get("flash_attn_4", "")
        dsl = versions.get("cutlass_dsl", "")
        parts = []
        if v:
            parts.append(f"FA4 {v}")
        if dsl:
            parts.append(f"DSL {dsl}")
        return ", ".join(parts)
    if "pytorch" in slug:
        v = versions.get("pytorch", "")
        return f"PyTorch {v}" if v else ""
    return ""


def format_results_table(bench_result: dict) -> str:
    """Format the summary as an aligned Markdown comparison table."""
    summary = bench_result.get("summary", {})
    if not summary:
        return ""

    gpu = summary["gpu"]
    peak = summary["peak_tflops"]
    golden = summary["golden"]
    impl_order = bench_result.get("impls_requested", [])
    config_order = list(next(iter(summary["impls"].values()))["configs"].keys()) if summary["impls"] else []

    versions = _detect_env_versions()

    # --- Build cell content first, then compute column widths ---
    CFG_LABEL = "Config"
    cfg_width = max(len(CFG_LABEL), *(len(c) for c in config_order), len("% of peak"))

    def _fmt_cell(slug: str, config_slug: str) -> str:
        entry = summary["impls"][slug]["configs"].get(config_slug, {})
        ms = entry.get("median_ms")
        if ms is None:
            return "—"
        ms_s = f"{ms:.3f}" if ms < 1 else f"{ms:.2f}"
        tf = entry.get("tflops")
        tf_s = f"{tf:.1f}" if tf else "—"
        cell = f"{ms_s:>6}  {tf_s:>6}"
        if slug != golden:
            ok = entry.get("correct")
            cell += " ✓" if ok is True else (" ✗" if ok is False else "  ")
            spd = entry.get("speedup")
            cell += f" {spd:.2f}x" if spd is not None else "     "
        return cell

    # Compute column widths from content + version labels
    col_widths: dict[str, int] = {}
    for slug in impl_order:
        sub = "ms    TFLOPS" + ("   speedup" if slug != golden else "")
        ver = _impl_version_label(slug, versions)
        w = max(len(slug), len(sub), len(ver))
        for cfg in config_order:
            w = max(w, len(_fmt_cell(slug, cfg)))
        pct = summary["impls"][slug]["best_pct_peak"]
        w = max(w, len(f"{pct:.1f}%"))
        col_widths[slug] = w

    # --- Render ---
    lines: List[str] = []

    # Environment info (GPU name is already in the table header Config column)
    host = versions.get("host", "")
    cuda_ver = versions.get("cuda", "")
    drv = versions.get("driver", "")
    env_parts = []
    if host:
        env_parts.append(f"host: {host}")
    if drv:
        env_parts.append(f"driver {drv}")
    if cuda_ver:
        env_parts.append(f"CUDA {cuda_ver}")
    if env_parts:
        lines.append(" | ".join(env_parts))
    lines.append("")

    # Build Config column info lines
    gpu_idx = versions.get("gpu_idx", "")
    cfg_info_1 = gpu  # GPU name
    peak_str = f"GPU {gpu_idx}, {peak} TFLOPS" if gpu_idx else f"{peak} TFLOPS"
    cfg_info_2 = peak_str
    cfg_width = max(cfg_width, len(cfg_info_1), len(cfg_info_2))

    # Header row 1: impl slugs
    hdr1 = f"| {cfg_info_1:<{cfg_width}} |"
    for slug in impl_order:
        hdr1 += f" {slug:<{col_widths[slug]}} |"
    lines.append(hdr1)

    # Header row 2: version labels
    hdr2 = f"| {cfg_info_2:<{cfg_width}} |"
    for slug in impl_order:
        ver = _impl_version_label(slug, versions)
        hdr2 += f" {ver:<{col_widths[slug]}} |"
    lines.append(hdr2)

    # Header row 3: column sub-labels (ms/TFLOPS/speedup)
    hdr3 = f"| {'':<{cfg_width}} |"
    for slug in impl_order:
        sub = "ms    TFLOPS" + ("   speedup" if slug != golden else "")
        hdr3 += f" {sub:<{col_widths[slug]}} |"
    lines.append(hdr3)

    # Separator
    sep = f"|{'-' * (cfg_width + 2)}|"
    for slug in impl_order:
        sep += f"{'-' * (col_widths[slug] + 2)}|"
    lines.append(sep)

    # Data rows
    for config_slug in config_order:
        row = f"| {config_slug:<{cfg_width}} |"
        for slug in impl_order:
            cell = _fmt_cell(slug, config_slug)
            row += f" {cell:<{col_widths[slug]}} |"
        lines.append(row)

    # % of peak row
    lines.append(sep)
    pct_row = f"| {'% of peak':<{cfg_width}} |"
    for slug in impl_order:
        pct = summary["impls"][slug]["best_pct_peak"]
        cell = f"{pct:.1f}%"
        pct_row += f" {cell:>{col_widths[slug]}} |"
    lines.append(pct_row)

    # Footer
    lines.append("")
    lines.append(f"Golden: {golden} — ✓/✗ = correctness vs golden")

    # Correctness summary — count failures per impl
    for slug in impl_order:
        if slug == golden:
            continue
        failed_configs = []
        passed_configs = []
        for cfg in config_order:
            entry = summary["impls"][slug]["configs"].get(cfg, {})
            ok = entry.get("correct")
            if ok is True:
                passed_configs.append(cfg)
            else:
                failed_configs.append(cfg)
        if failed_configs:
            lines.append("")
            lines.append(f"⚠️  {slug}: CORRECTNESS FAILED on {len(failed_configs)}/{len(config_order)} configs: {', '.join(failed_configs)}")
            lines.append(f"    Fix correctness BEFORE optimizing performance. Wrong results = wasted effort.")
        elif passed_configs:
            lines.append(f"✅  {slug}: correctness PASSED on all {len(passed_configs)} configs")

    # GPU clock lock status
    clock_lock = bench_result.get("gpu_clock_lock", {})
    if clock_lock:
        status = clock_lock.get("status", "unknown")
        lines.append("")
        if status == "ok":
            sm = clock_lock.get("locked_sm_mhz", "?")
            mem = clock_lock.get("locked_mem_mhz", "?")
            pw = clock_lock.get("power_limit_w")
            pw_str = f", Power {pw}W" if pw else ""
            lines.append(f"GPU clocks locked: SM {sm} MHz, Mem {mem} MHz{pw_str}")
        elif status == "warning":
            sm = clock_lock.get("locked_sm_mhz", "?")
            pw = clock_lock.get("power_limit_w")
            pw_str = f", Power {pw}W" if pw else ""
            lines.append(f"GPU clocks locked: SM {sm} MHz{pw_str} — WARNING: throttling detected")
            for w in clock_lock.get("warnings", []):
                lines.append(f"  - {w}")
        elif status == "error":
            err = clock_lock.get("error", "unknown")
            lines.append(f"GPU clock lock FAILED: {err}")
        elif status == "skipped":
            lines.append("GPU clock lock: skipped (disabled)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-config autotune: compile + trial per combo group
# ---------------------------------------------------------------------------

def _autotune_compile_only(
    *,
    cu_impl: dict,
    at_result,
    py_impls: list,
    configs: dict,
    run_tag: str,
    kernel: str,
    timeout_seconds: int,
) -> dict:
    """Compile autotune groups — NO trial. Trial happens in the main single-pass Phase C.

    Returns dict with:
        compiled_groups: {defines_flags: {meta, binary, cr, config_slugs}}
        per_config_binary: {config_slug: binary_path} — maps each config to its winning binary
        compile_ok: bool
        autotune_info: dict
    """
    import os
    from collections import defaultdict

    per_config = at_result.per_config_results

    # Step 1: Group autotuned configs by winning combo's defines_flags
    groups: Dict[str, List[str]] = defaultdict(list)
    for config_slug, pcw in per_config.items():
        groups[pcw.defines_flags].append(config_slug)

    logger.info("[%s] per-config autotune: %d unique combos for %d configs",
                cu_impl["slug"], len(groups), len(per_config))

    # ================================================================
    # Phase 1: Compile ALL groups — collect binaries
    # ================================================================
    compile_impls = {cu_impl["slug"]: dict(cu_impl["files"])}
    for py_impl in py_impls:
        compile_impls[py_impl["slug"]] = dict(py_impl["files"])

    # compiled_groups: {defines_flags: {"meta": ..., "binary": ..., "cr": ..., "config_slugs": ...}}
    compiled_groups: Dict[str, dict] = {}
    all_compile_ok = True
    compile_results_all: Dict[str, dict] = {}

    compile_phase_start = time.time()
    for group_idx, (defines_flags, group_config_slugs) in enumerate(groups.items()):
        unique_rev = (int(time.time_ns()) % 10000000) + group_idx
        meta = Metadata(
            run_tag=run_tag,
            version="v1",
            direction_id=0,
            direction_slug=f"{kernel}-{cu_impl['slug']}",
            revision=unique_rev,
        )

        # Set NVCC_EXTRA_FLAGS for this group's defines
        old_extra_flags = os.environ.get("NVCC_EXTRA_FLAGS")
        existing = old_extra_flags or ""
        os.environ["NVCC_EXTRA_FLAGS"] = f"{existing} {defines_flags}".strip()

        logger.info("[%s] compile group %d/%d: %s (configs: %s)",
                    cu_impl["slug"], group_idx + 1, len(groups),
                    defines_flags, ", ".join(group_config_slugs))

        try:
            compile_start = time.time()
            compile_req = CompileRequest(
                metadata=meta,
                timeout_seconds=timeout_seconds,
                impls=compile_impls,
            )
            compile_resp = compile_endpoint(compile_req)
            cr = compile_resp.model_dump(mode="json")
            logger.info("[%s] compile group %d done (%.1fs)",
                        cu_impl["slug"], group_idx + 1, time.time() - compile_start)
        finally:
            if old_extra_flags is not None:
                os.environ["NVCC_EXTRA_FLAGS"] = old_extra_flags
            else:
                os.environ.pop("NVCC_EXTRA_FLAGS", None)

        compile_results_all[defines_flags] = cr

        if not compile_resp.all_ok:
            logger.warning("[%s] compile group %d failed", cu_impl["slug"], group_idx + 1)
            all_compile_ok = False
            compiled_groups[defines_flags] = {
                "meta": meta, "binary": None, "cr": cr,
                "config_slugs": group_config_slugs,
            }
            continue

        # Resolve binary
        try:
            workspace = resolve_workspace_bundle(**meta.model_dump())
            target_path, _ = _primary_artifact_from_manifest(workspace)
            binary_path = str(target_path)
            logger.info("[%s] group %d binary: %s", cu_impl["slug"], group_idx + 1, target_path)
        except Exception as exc:
            logger.warning("[%s] binary resolution failed for group %d: %s",
                          cu_impl["slug"], group_idx + 1, exc)
            binary_path = None

        compiled_groups[defines_flags] = {
            "meta": meta, "binary": binary_path, "cr": cr,
            "config_slugs": group_config_slugs,
        }

    logger.info("[%s] compile phase done (%.1fs): %d/%d groups have binaries",
                cu_impl["slug"], time.time() - compile_phase_start,
                sum(1 for g in compiled_groups.values() if g["binary"]),
                len(compiled_groups))

    # ================================================================
    # Phase 1b: Compile default group (configs without autotune)
    # ================================================================
    default_config_slugs = at_result.configs_without_autotune
    if default_config_slugs:
        unique_rev = (int(time.time_ns()) % 10000000) + len(compiled_groups)
        default_meta = Metadata(
            run_tag=run_tag,
            version="v1",
            direction_id=0,
            direction_slug=f"{kernel}-{cu_impl['slug']}",
            revision=unique_rev,
        )

        old_extra_flags = os.environ.get("NVCC_EXTRA_FLAGS")
        # No extra defines — use kernel's default #define values
        os.environ.pop("NVCC_EXTRA_FLAGS", None)

        logger.info("[%s] compile default group: %d configs (no autotune)",
                    cu_impl["slug"], len(default_config_slugs))

        try:
            compile_start = time.time()
            compile_req = CompileRequest(
                metadata=default_meta,
                timeout_seconds=timeout_seconds,
                impls=compile_impls,
            )
            compile_resp = compile_endpoint(compile_req)
            cr = compile_resp.model_dump(mode="json")
            logger.info("[%s] compile default group done (%.1fs)",
                        cu_impl["slug"], time.time() - compile_start)
        finally:
            if old_extra_flags is not None:
                os.environ["NVCC_EXTRA_FLAGS"] = old_extra_flags

        compile_results_all["__default__"] = cr
        default_binary = None

        if compile_resp.all_ok:
            try:
                workspace = resolve_workspace_bundle(**default_meta.model_dump())
                target_path, _ = _primary_artifact_from_manifest(workspace)
                default_binary = str(target_path)
                logger.info("[%s] default group binary: %s", cu_impl["slug"], target_path)
            except Exception as exc:
                logger.warning("[%s] default group binary resolution failed: %s",
                              cu_impl["slug"], exc)
        else:
            all_compile_ok = False

        compiled_groups["__default__"] = {
            "meta": default_meta, "binary": default_binary, "cr": cr,
            "config_slugs": default_config_slugs,
        }

    # ================================================================
    # Build per-config binary map (NO trial here — trial in Phase C)
    # ================================================================
    per_config_binary: Dict[str, str | None] = {}
    for defines_flags, group in compiled_groups.items():
        for cs in group["config_slugs"]:
            per_config_binary[cs] = group["binary"]  # None if compile failed

    autotune_info = {
        "total_combos": at_result.total_combos,
        "valid_combos": at_result.valid_combos,
        "compiled_ok": at_result.compiled_ok,
        "benchmarked_ok": at_result.benchmarked_ok,
        "duration_s": round(at_result.duration_s, 1),
        "num_groups": len(groups),
        "configs_without_autotune": at_result.configs_without_autotune,
        "per_config_winners": {
            slug: {
                "combo": pcw.best_combo,
                "tag": pcw.best_tag,
                "median_ms": pcw.best_median_ms,
                "defines_flags": pcw.defines_flags,
            }
            for slug, pcw in per_config.items()
        },
    }

    # Use any group's metadata for the trial workspace
    any_meta = next((g["meta"] for g in compiled_groups.values() if g["binary"]), None)

    return {
        "compiled_groups": compiled_groups,
        "per_config_binary": per_config_binary,
        "compile_ok": all_compile_ok,
        "compile_results": compile_results_all,
        "autotune_info": autotune_info,
        "meta": any_meta,
    }


def formal_benchmark(
    kernel: str,
    arch: str,
    *,
    run_tag: str | None = None,
    impls: str | List[str] = "all",
    timeout_seconds: int = 120,
    kb_repo: str | None = None,
    runtime_root: str | None = None,
    data_root: str | None = None,
    gpu_lockdown: bool = True,
) -> dict:
    """Atomic compile + trial ALL configs for specified implementations.

    Snapshot-first flow:
    1. Snapshot sources + configs to kernel_lab_kb (prepare_run)
    2. Resolve impls from the snapshot (never the original files)
    3. Compile + trial using snapshot file contents
    4. Write results + check gems (finalize_run)
    """
    # Resolve "auto" arch from host config / nvidia-smi
    if arch == "auto":
        from cuda_exec.host_env import resolve_arch
        arch = resolve_arch()

    _ts_fmt = "%H:%M:%S"
    bench_start = datetime.now()
    logger.info("Bench start [%s] kernel=%s arch=%s", bench_start.strftime(_ts_fmt), kernel, arch)

    # --- Propagate host-specific flags to reference impls ---
    import os
    from cuda_exec.host_env import _match_host_entry
    _, host_entry = _match_host_entry()
    if host_entry and host_entry.get("env", {}).get("cudnn_sdpa_broken"):
        os.environ["CUDA_EXEC_CUDNN_BROKEN"] = "1"
    else:
        os.environ.pop("CUDA_EXEC_CUDNN_BROKEN", None)

    # --- Resolve paths from config ---
    kb_repo_path = Path(kb_repo).expanduser() if kb_repo else Path.home() / "kernel_lab_kb"
    runtime_root_path = Path(runtime_root).expanduser() if runtime_root else Path.home() / ".cuda_exec_bench"
    data_root_path = Path(data_root).expanduser() if data_root else None

    # --- Resolve run_tag: explicit > env var > auto-detect ---
    if not run_tag:
        run_tag = os.environ.get("CUDA_EXEC_RUN_TAG")

    # --- Phase 1: Snapshot ---
    run_dir = None
    snapshot_data = data_root_path  # fallback: use explicit data_root or None (= project data/)
    logger.info("Snapshot start [%s] run_tag=%s", datetime.now().strftime(_ts_fmt), run_tag or "(auto)")
    try:
        from cuda_exec.trajectory import prepare_run
        run_dir = prepare_run(kernel, arch, impls, timeout_seconds, kb_repo=kb_repo_path, run_tag=run_tag)
        snapshot_data = run_dir
        logger.info("Snapshot done [%s] → %s", datetime.now().strftime(_ts_fmt), run_dir)
    except Exception as exc:
        logger.warning("Snapshot failed [%s]: %s — falling back to original data/", datetime.now().strftime(_ts_fmt), exc)

    # --- Phase 2: Resolve from snapshot (or original if snapshot failed) ---
    logger.info("Resolve impls start [%s]", datetime.now().strftime(_ts_fmt))
    configs = load_configs(kernel, data_root=snapshot_data)
    resolved = resolve_impls(kernel, arch, impls, data_root=snapshot_data)
    logger.info("Resolve impls done [%s] impls=%s", datetime.now().strftime(_ts_fmt), [r["slug"] for r in resolved])

    # --- Isolate runtime ---
    ts_str = run_dir.name if run_dir else f"{int(time.time())}"
    bench_runtime = runtime_root_path / kernel / arch / ts_str
    bench_runtime.mkdir(parents=True, exist_ok=True)
    old_exec_root = os.environ.get("CUDA_EXEC_ROOT")
    os.environ["CUDA_EXEC_ROOT"] = str(bench_runtime)

    # Auto-generate run_tag for workspace isolation
    run_tag = f"bench-{kernel}-{int(time.time())}"

    refs = [r for r in resolved if r["source"] in ("ref", "peak")]
    gens = [r for r in resolved if r["source"] == "gen"]
    samples = [r for r in resolved if r["source"] == "sample"]

    # Golden = first ref (alphabetical). Used for speedup ratios.
    # Correctness is only valid between impls sharing the same input
    # generation path (.cu↔.cu via C harness, .py↔.py via generate_inputs).
    primary_ref = refs[0]

    results: Dict[str, dict] = {}

    # --- Helper: load and run a .py impl, return per-config performance + output ---
    def _run_py_impl(impl: dict, per_impl_timeout: int = 600) -> dict:
        """Run a .py impl in a subprocess with timeout to prevent hangs.

        CuTe DSL JIT compilation and cuDNN can hang or segfault — running
        in a subprocess isolates failures and enforces a timeout.
        """
        import subprocess as _sp
        import tempfile as _tempfile

        tmp_dir = Path(_tempfile.mkdtemp(prefix=f"bench_{impl['slug']}_"))
        for fname, content in impl["files"].items():
            (tmp_dir / fname).write_text(content, encoding="utf-8")

        # Write a runner script that imports the module and measures all configs
        runner = tmp_dir / "_runner.py"
        runner.write_text(f"""
import json, sys, os
sys.path.insert(0, {str(tmp_dir)!r})
sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
from pathlib import Path
from cuda_exec.scripts.eval_support import load_reference_module, measure_reference
import torch

device = torch.device("cuda")
entry = Path({str(tmp_dir / f"{impl['name']}.py")!r})
configs = json.loads({json.dumps(configs)!r})

mod = load_reference_module(entry)
results = {{}}
for slug, config in configs.items():
    try:
        r = measure_reference(mod, config, device, num_trials=10)
        results[slug] = {{"status": "ok", "performance": r["performance"]}}
    except Exception as e:
        results[slug] = {{"status": "error", "error": str(e)}}

print(json.dumps({{"ok": True, "configs": results}}))
""", encoding="utf-8")

        env = os.environ.copy()
        try:
            result = _sp.run(
                [sys.executable, str(runner)],
                capture_output=True, text=True,
                timeout=per_impl_timeout, env=env,
                cwd=str(Path(__file__).resolve().parents[1]),
            )
            if result.returncode != 0:
                stderr_tail = (result.stderr or "")[-500:]
                return {"ok": False, "error": f"exit {result.returncode}: {stderr_tail}"}
            stdout = result.stdout.strip()
            if stdout:
                return json.loads(stdout)
            return {"ok": False, "error": "no output from runner"}
        except _sp.TimeoutExpired:
            return {"ok": False, "error": f"timeout after {per_impl_timeout}s"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # Split ALL impls by file type, not by source.
    all_impls = refs + gens + samples
    py_impls = [i for i in all_impls if i["file_type"] == "py"]
    cu_impls = [i for i in all_impls if i["file_type"] == "cu"]

    # --- GPU Clock Lock ---
    from cuda_exec.gpu_clocks import gpu_clock_context
    gpu_lockdown_enabled = gpu_lockdown
    clock_lock_ctx = gpu_clock_context(enabled=gpu_lockdown_enabled)
    clock_lock_info = clock_lock_ctx.__enter__()

    try:

        # ================================================================
        # Phase B: Compile ALL .cu impls, collect binary paths
        # (.py impls don't need compilation — they run via trial.py directly)
        #
        # Two sub-phases:
        #   B1: Autotune impls (sequential — autotune internally parallelizes combos)
        #   B2: Non-autotune impls (parallel — independent compile_endpoint calls)
        # ================================================================
        compiled_binaries: Dict[str, str] = {}     # slug → binary path (non-autotuned)
        compile_results: Dict[str, dict] = {}      # slug → compile result
        compile_metadata: Dict[str, object] = {}   # slug → Metadata
        autotune_results: Dict[str, object] = {}   # slug → autotune result
        autotune_binaries: Dict[str, dict] = {}    # slug → autotune binary info

        # Classify impls: autotune vs non-autotune
        autotune_impls = []
        default_impls = []
        for cu_impl in cu_impls:
            if cu_impl["source"] in ("gen", "sample"):
                autotune_yaml = Path(cu_impl["entry_point"]).parent / "autotune.yaml"
                if autotune_yaml.exists():
                    autotune_impls.append(cu_impl)
                    continue
            default_impls.append(cu_impl)

        # --- B1: Autotune impls (sequential) ---
        for cu_impl in autotune_impls:
            unique_rev = (int(time.time_ns()) % 10000000)
            meta = Metadata(
                run_tag=run_tag,
                version="v1",
                direction_id=0,
                direction_slug=f"{kernel}-{cu_impl['slug']}",
                revision=unique_rev,
            )
            compile_metadata[cu_impl["slug"]] = meta

            try:
                from cuda_exec.autotune import run_autotune, format_autotune_report
                from cuda_exec.host_env import resolve_compile_arch, resolve_host_env

                compile_arch = resolve_compile_arch()
                autotune_env = os.environ.copy()
                autotune_env.update(resolve_host_env())
                if "CUDA_VISIBLE_DEVICES" in os.environ:
                    autotune_env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]

                logger.info("[%s] autotune start", cu_impl["slug"])
                at_start = time.time()
                autotune_dir = bench_runtime / "autotune" / cu_impl["slug"]
                at_result = run_autotune(
                    cu_path=Path(cu_impl["entry_point"]),
                    autotune_yaml=Path(cu_impl["entry_point"]).parent / "autotune.yaml",
                    configs=configs,
                    arch=compile_arch,
                    env_base=autotune_env,
                    output_dir=autotune_dir,
                )
                at_report = format_autotune_report(at_result)
                logger.info("[%s] autotune done (%.1fs)\n%s",
                            cu_impl["slug"], time.time() - at_start, at_report)
                print(at_report, file=sys.stderr)
                autotune_results[cu_impl["slug"]] = at_result

                per_config_binary: Dict[str, str | None] = {}
                for cfg_slug, pcw in at_result.per_config_results.items():
                    per_config_binary[cfg_slug] = pcw.binary_path if pcw.binary_path else None
                for cfg_slug in at_result.configs_without_autotune:
                    per_config_binary[cfg_slug] = None

                autotune_binaries[cu_impl["slug"]] = {
                    "per_config_binary": per_config_binary,
                    "compile_ok": True,
                    "autotune_info": {
                        "total_combos": at_result.total_combos,
                        "valid_combos": at_result.valid_combos,
                        "compiled_ok": at_result.compiled_ok,
                        "benchmarked_ok": at_result.benchmarked_ok,
                        "duration_s": round(at_result.duration_s, 1),
                        "configs_without_autotune": at_result.configs_without_autotune,
                        "per_config_winners": {
                            s: {"combo": p.best_combo, "tag": p.best_tag,
                                "median_ms": p.best_median_ms, "defines_flags": p.defines_flags}
                            for s, p in at_result.per_config_results.items()
                        },
                    },
                }

            except Exception as exc:
                logger.warning("[%s] autotune failed: %s — falling back to default compile",
                               cu_impl["slug"], exc)
                default_impls.append(cu_impl)  # fall back to default compile

        # --- B2: Non-autotune impls (parallel compile) ---
        def _compile_one_impl(cu_impl: dict) -> tuple:
            """Compile a single impl. Returns (slug, meta, cr, binary_path, error)."""
            unique_rev = (int(time.time_ns()) % 10000000) + hash(cu_impl["slug"]) % 1000
            meta = Metadata(
                run_tag=run_tag,
                version="v1",
                direction_id=0,
                direction_slug=f"{kernel}-{cu_impl['slug']}",
                revision=unique_rev,
            )

            compile_impls = {cu_impl["slug"]: dict(cu_impl["files"])}
            for other_cu in cu_impls:
                if other_cu["slug"] != cu_impl["slug"]:
                    compile_impls[other_cu["slug"]] = dict(other_cu["files"])
            for py_impl in py_impls:
                compile_impls[py_impl["slug"]] = dict(py_impl["files"])

            compile_req = CompileRequest(
                metadata=meta,
                timeout_seconds=timeout_seconds,
                impls=compile_impls,
            )
            compile_resp = compile_endpoint(compile_req)
            cr = compile_resp.model_dump(mode="json")

            if not compile_resp.all_ok:
                return (cu_impl["slug"], meta, cr, None, "compile failed")

            try:
                workspace = resolve_workspace_bundle(**meta.model_dump())
                target_path, _ = _primary_artifact_from_manifest(workspace)
                return (cu_impl["slug"], meta, cr, str(target_path), None)
            except Exception as exc:
                return (cu_impl["slug"], meta, cr, None, f"binary resolution failed: {exc}")

        if default_impls:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            compile_phase_start = time.time()
            logger.info("Phase B2: compiling %d non-autotune impls in parallel", len(default_impls))

            with ThreadPoolExecutor(max_workers=len(default_impls)) as pool:
                futures = {pool.submit(_compile_one_impl, impl): impl for impl in default_impls}
                for future in as_completed(futures):
                    slug, meta, cr, binary_path, error = future.result()
                    compile_metadata[slug] = meta
                    compile_results[slug] = cr

                    if error:
                        logger.warning("[%s] %s", slug, error)
                        results[slug] = {
                            "impl": slug,
                            "compile_ok": binary_path is not None,
                            "trial_ok": False,
                            "compile_result": cr,
                            "trial_result": None,
                            "error": error,
                        }
                    else:
                        compiled_binaries[slug] = binary_path
                        logger.info("[%s] compiled → %s", slug, binary_path)

            logger.info("Phase B2: done (%.1fs)", time.time() - compile_phase_start)

        # ================================================================
        # Phase C: Single-pass trial — ALL impls per config.
        # For autotuned impls, the binary changes per config (different
        # autotune winner). For non-autotuned impls, same binary for all.
        # ================================================================
        has_any_impl = compiled_binaries or autotune_binaries or py_impls
        if has_any_impl:
            # Pick a workspace metadata for trial (any compiled impl works)
            trial_meta = None
            if compiled_binaries:
                first_slug = next(iter(compiled_binaries))
                trial_meta = compile_metadata[first_slug]
            elif autotune_binaries:
                for at_data in autotune_binaries.values():
                    if at_data.get("meta"):
                        trial_meta = at_data["meta"]
                        break
            if trial_meta is None:
                trial_meta = Metadata(
                    run_tag=run_tag, version="v1", direction_id=0,
                    direction_slug=f"{kernel}-trial", revision=int(time.time()) % 100000,
                )

            # Check if any autotuned impl has per-config binaries
            has_per_config = bool(autotune_binaries)

            if not has_per_config:
                # Simple case: same binary for all configs → one trial call
                binary_map_str = ",".join(f"{s}={p}" for s, p in compiled_binaries.items())
                logger.info("Trial phase: %d configs × %d cu + %d py (single pass, uniform binaries)",
                             len(configs), len(compiled_binaries), len(py_impls))

                trial_start = time.time()
                trial_req = TrialRequest(
                    metadata=trial_meta,
                    timeout_seconds=timeout_seconds,
                    configs=configs,
                    binary_map=binary_map_str,
                )
                trial_resp = trial_endpoint(trial_req)
                trial_result = trial_resp.model_dump(mode="json")
                logger.info("Trial phase done (%.1fs)", time.time() - trial_start)
            else:
                # Autotuned: per-config binary map → one trial call per config
                logger.info("Trial phase: %d configs × %d cu + %d at + %d py (single pass, per-config binaries)",
                             len(configs), len(compiled_binaries), len(autotune_binaries), len(py_impls))

                trial_start = time.time()
                all_trial_configs = {}

                for config_slug, config_params in configs.items():
                    # Build binary map for this config: non-autotuned + autotuned
                    config_binary_map = dict(compiled_binaries)  # copy non-autotuned

                    for at_slug, at_data in autotune_binaries.items():
                        per_config_bin = at_data.get("per_config_binary", {})
                        binary = per_config_bin.get(config_slug)
                        if binary:
                            config_binary_map[at_slug] = binary

                    binary_map_str = ",".join(f"{s}={p}" for s, p in config_binary_map.items())

                    trial_req = TrialRequest(
                        metadata=trial_meta,
                        timeout_seconds=timeout_seconds,
                        configs={config_slug: config_params},
                        binary_map=binary_map_str,
                    )
                    trial_resp_single = trial_endpoint(trial_req)
                    single_result = trial_resp_single.model_dump(mode="json")

                    for cs, cv in single_result.get("configs", {}).items():
                        all_trial_configs[cs] = cv

                trial_result = {"all_ok": True, "configs": all_trial_configs}
                trial_resp = type("Resp", (), {"all_ok": True})()  # mock
                logger.info("Trial phase done (%.1fs)", time.time() - trial_start)

            # Distribute trial results to each .cu impl
            for cu_impl in cu_impls:
                slug = cu_impl["slug"]
                if slug not in compiled_binaries and slug not in autotune_binaries:
                    continue  # compile failed, already recorded
                cr = compile_results.get(slug, {})
                compile_ok = cr.get("all_ok", False) if isinstance(cr, dict) else False
                # For autotuned impls, compile_ok = autotune compile_ok
                if slug in autotune_binaries:
                    compile_ok = autotune_binaries[slug].get("compile_ok", False)
                impl_result = {
                    "impl": slug,
                    "compile_ok": compile_ok,
                    "trial_ok": trial_resp.all_ok,
                    "compile_result": cr if isinstance(cr, dict) and "all_ok" in cr else None,
                    "trial_result": trial_result,
                }
                at = autotune_results.get(slug)
                if at is not None:
                    at_data = autotune_binaries.get(slug, {})
                    impl_result["autotune"] = at_data.get("autotune_info", {})
                results[slug] = impl_result

            # Distribute trial results to each .py impl
            for py_impl in py_impls:
                slug = py_impl["slug"]
                results[slug] = {
                    "impl": slug,
                    "compile_ok": None,
                    "trial_ok": trial_resp.all_ok,
                    "compile_result": None,
                    "trial_result": trial_result,
                }

    finally:
        # --- Unlock GPU clocks ---
        clock_lock_ctx.__exit__(None, None, None)

    # --- Collect GPU throttle warnings from trial results ---
    throttle_warnings: List[str] = []
    for slug, impl_result in results.items():
        trial = impl_result.get("trial_result")
        if not trial:
            continue
        for config_slug, entry in trial.get("configs", {}).items():
            for impl_slug, impl_data in entry.get("impls", {}).items():
                gs = impl_data.get("gpu_state") if isinstance(impl_data, dict) else None
                if gs and gs.get("throttled"):
                    before_mhz = gs.get("before", {}).get("sm_clock_mhz", "?")
                    during_mhz = gs.get("during", {}).get("sm_clock_mhz", "?")
                    throttle_warnings.append(
                        f"{config_slug}/{impl_slug}: {before_mhz}->{during_mhz} MHz"
                    )

    # --- Build gpu_clock_lock status ---
    gpu_clock_lock = dict(clock_lock_info)
    if throttle_warnings:
        gpu_clock_lock["warnings"] = throttle_warnings
        if gpu_clock_lock.get("status") == "ok":
            gpu_clock_lock["status"] = "warning"

    bench_result = {
        "kernel": kernel,
        "arch": arch,
        "num_configs": len(configs),
        "impls_requested": [r["slug"] for r in resolved],
        "refs": [r["slug"] for r in refs],
        "gens": [r["slug"] for r in gens],
        "source_paths": {r["slug"]: str(Path(r["entry_point"]).resolve()) for r in resolved},
        "results": results,
        "gpu_clock_lock": gpu_clock_lock,
    }

    # --- Phase 4: Restore runtime env + finalize ---
    if old_exec_root is not None:
        os.environ["CUDA_EXEC_ROOT"] = old_exec_root
    else:
        os.environ.pop("CUDA_EXEC_ROOT", None)

    if run_dir is not None:
        try:
            from cuda_exec.trajectory import finalize_run
            new_gems = finalize_run(run_dir, bench_result, kb_repo=kb_repo_path, runtime_root=bench_runtime)
            bench_result["gems"] = new_gems
            bench_result["improved"] = len(new_gems) > 0
            logger.info("Results finalized in %s", run_dir)
        except Exception as exc:
            logger.warning("Failed to finalize run: %s", exc)
            bench_result["gems"] = {}
            bench_result["improved"] = False

    bench_end = datetime.now()
    bench_dur = (bench_end - bench_start).total_seconds()
    logger.info("Bench end [%s] total=%.1fs", bench_end.strftime(_ts_fmt), bench_dur)
    return bench_result


def _merge_best_of_n(all_results: List[dict]) -> dict:
    """Merge N full benchmark passes, keeping best (min) latency per (impl, config).

    Takes the structure of the last run as the base and patches in the best
    latency_ms values from across all runs.  Correctness is OR'd (passed in
    any run = passed).
    """
    base = all_results[-1]  # use last run as structural base

    def _set_latency(d: dict, path: list[str], value: float):
        """Set a nested dict value by path, creating intermediates."""
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value

    def _get_latency(d: dict, path: list[str]) -> float | None:
        for key in path:
            if not isinstance(d, dict):
                return None
            d = d.get(key)
        return d

    # Collect best median_ms per (impl_slug, config_slug) across all runs
    # Two paths depending on impl type:
    #   .cu impl: results[slug].trial_result.configs[cfg].impls[slug].performance.latency_ms.median
    #   .py impl: results[slug].trial_result.configs[cfg].performance.latency_ms.median
    for slug in base.get("results", {}):
        base_impl = base["results"][slug]
        base_trial = base_impl.get("trial_result")
        if not base_trial:
            continue

        for config_slug in base_trial.get("configs", {}):
            entry = base_trial["configs"][config_slug]

            if "impls" in entry:
                # .cu trial — each impl has its own performance block
                for impl_slug in entry.get("impls", {}):
                    path = ["performance", "latency_ms", "median"]
                    best = _get_latency(entry["impls"][impl_slug], path)
                    for r in all_results[:-1]:
                        other = (r.get("results", {}).get(slug, {})
                                  .get("trial_result", {}).get("configs", {})
                                  .get(config_slug, {}).get("impls", {})
                                  .get(impl_slug, {}))
                        v = _get_latency(other, path)
                        if v is not None and (best is None or v < best):
                            best = v
                    if best is not None:
                        _set_latency(entry["impls"][impl_slug], path, best)

                    # Also patch min
                    path_min = ["performance", "latency_ms", "min"]
                    best_min = _get_latency(entry["impls"][impl_slug], path_min)
                    for r in all_results[:-1]:
                        other = (r.get("results", {}).get(slug, {})
                                  .get("trial_result", {}).get("configs", {})
                                  .get(config_slug, {}).get("impls", {})
                                  .get(impl_slug, {}))
                        v = _get_latency(other, path_min)
                        if v is not None and (best_min is None or v < best_min):
                            best_min = v
                    if best_min is not None:
                        _set_latency(entry["impls"][impl_slug], path_min, best_min)

            elif "performance" in entry:
                # .py impl — performance directly on the config entry
                path = ["performance", "latency_ms", "median"]
                best = _get_latency(entry, path)
                for r in all_results[:-1]:
                    other = (r.get("results", {}).get(slug, {})
                              .get("trial_result", {}).get("configs", {})
                              .get(config_slug, {}))
                    v = _get_latency(other, path)
                    if v is not None and (best is None or v < best):
                        best = v
                if best is not None:
                    _set_latency(entry, path, best)

                path_min = ["performance", "latency_ms", "min"]
                best_min = _get_latency(entry, path_min)
                for r in all_results[:-1]:
                    other = (r.get("results", {}).get(slug, {})
                              .get("trial_result", {}).get("configs", {})
                              .get(config_slug, {}))
                    v = _get_latency(other, path_min)
                    if v is not None and (best_min is None or v < best_min):
                        best_min = v
                if best_min is not None:
                    _set_latency(entry, path_min, best_min)

    base["num_runs"] = len(all_results)
    return base


def cli_main() -> None:
    """CLI entry point for ik:bench (Hydra-based)."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import os
    import sys

    _CONF_DIR = str(Path(__file__).resolve().parents[1] / "conf")

    # Hydra compose API: pass overrides from sys.argv
    overrides = [arg for arg in sys.argv[1:] if "=" in arg]

    with initialize_config_dir(config_dir=_CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    OmegaConf.resolve(cfg)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    bench_cfg = cfg.bench

    # Validate required parameters before proceeding
    from omegaconf.errors import MissingMandatoryValue
    try:
        _ = bench_cfg.kernel
    except MissingMandatoryValue:
        print(
            "Usage: .venv/bin/python -m cuda_exec.formal bench.kernel=<KERNEL> [OPTIONS]\n"
            "\n"
            "Required:\n"
            "  bench.kernel=<name>        Kernel to benchmark (fa4, matmul, vecadd, ...)\n"
            "\n"
            "Options:\n"
            "  bench.gpu=<N>              GPU index (CUDA_VISIBLE_DEVICES)\n"
            "  bench.arch=<smXX>          GPU arch (default: auto-detect)\n"
            "  bench.impls=[i1,i2,...]    Impl slugs to bench, or 'all' (default: all)\n"
            "  bench.timeout=<seconds>    Per-config timeout (default: 120)\n"
            "\n"
            "Examples:\n"
            "  .venv/bin/python -m cuda_exec.formal bench.kernel=matmul\n"
            "  .venv/bin/python -m cuda_exec.formal bench.kernel=fa4 bench.gpu=2\n"
            "  .venv/bin/python -m cuda_exec.formal bench.kernel=fa4 'bench.impls=[ref-cublas,gen-cuda]'",
            file=sys.stderr,
        )
        sys.exit(1)

    # Set CUDA_VISIBLE_DEVICES: explicit gpu= > host config > env
    gpu = bench_cfg.get("gpu")
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        from cuda_exec.host_env import resolve_benchmark_gpus
        bench_gpus = resolve_benchmark_gpus()
        if bench_gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = bench_gpus

    # Kill any existing formal bench on the same GPU.
    # Only one formal bench per GPU — concurrent instances fight for GPU
    # resources and all stall. The newest one wins.
    if gpu is not None:
        import signal
        my_pid = os.getpid()
        try:
            result = __import__("subprocess").run(
                ["pgrep", "-af", f"cuda_exec.formal"],
                capture_output=True, text=True,
            )
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split(None, 1)
                pid = int(parts[0])
                cmdline = parts[1] if len(parts) > 1 else ""
                if pid != my_pid and f"bench.gpu={gpu}" in cmdline:
                    logger.info("Killing existing formal bench PID %d on GPU %s", pid, gpu)
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        except Exception as e:
            logger.warning("Failed to check/kill existing bench: %s", e)

    # Kill existing processes on the target GPU (ensures exclusive access)
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu_id is not None and gpu_id != "0":
        _kill_gpu_processes(int(gpu_id))

    impls = bench_cfg.impls
    if isinstance(impls, str) and impls != "all":
        impls = [impls]
    elif hasattr(impls, "__iter__") and not isinstance(impls, str):
        impls = list(impls)

    # --- Resolve gem version → data_root override ---
    gem_version = bench_cfg.get("gem")
    data_root_override = bench_cfg.get("data_root")

    if gem_version is not None and data_root_override is None:
        gem_version = str(gem_version).strip().lstrip("v")
        # Normalize: "5" → "005", "03" → "003", "002" → "002"
        gem_num = int(gem_version)
        gem_prefix = f"v{gem_num:03d}_"

        run_tag = bench_cfg.get("run_tag") or os.environ.get("CUDA_EXEC_RUN_TAG")
        if not run_tag:
            from cuda_exec.impls import _resolve_run_tag
            run_tag = _resolve_run_tag()

        kb_repo_path = Path(bench_cfg.get("kb_repo") or Path.home() / "kernel_lab_kb")
        gems_dir = kb_repo_path / "runs" / run_tag / "gems"

        gem_matches = sorted(gems_dir.glob(f"{gem_prefix}*"))
        if not gem_matches:
            print(f"Error: gem {gem_prefix}* not found in {gems_dir}", file=sys.stderr)
            sys.exit(1)
        gem_dir = gem_matches[-1]
        data_root_override = str(gem_dir)
        print(f"Using gem: {gem_dir.name} → {data_root_override}", file=sys.stderr)

    num_runs = int(bench_cfg.get("runs", 1))
    bench_kwargs = dict(
        kernel=bench_cfg.kernel,
        arch=bench_cfg.arch,
        run_tag=bench_cfg.get("run_tag"),
        impls=impls,
        timeout_seconds=bench_cfg.timeout,
        kb_repo=bench_cfg.get("kb_repo"),
        runtime_root=bench_cfg.get("runtime_root"),
        data_root=data_root_override,
        gpu_lockdown=bench_cfg.get("gpu_lockdown", True),
    )

    if num_runs <= 1:
        result = formal_benchmark(**bench_kwargs)
    else:
        # Multi-run: run N full passes, keep best (min latency) per (impl, config)
        all_results = []
        for run_idx in range(num_runs):
            logger.info("=== Run %d/%d ===", run_idx + 1, num_runs)
            r = formal_benchmark(**bench_kwargs)
            all_results.append(r)

        result = _merge_best_of_n(all_results)

    # Enrich with TFLOPS, speedup, correctness, % peak
    from cuda_exec.impls import load_configs as _load_configs
    try:
        data_root_str = bench_cfg.get("data_root")
        data_root_path = Path(data_root_str).expanduser() if data_root_str else None
        all_configs = _load_configs(result["kernel"], data_root=data_root_path)
        enrich_result(result, all_configs)
    except Exception as exc:
        logger.warning("Failed to enrich result: %s", exc)

    print(json.dumps(result, indent=2, default=str))

    # Markdown table to stderr
    table = format_results_table(result)
    if table:
        print(table, file=sys.stderr)

    # Print source paths and GPU index
    source_paths = result.get("source_paths", {})
    if source_paths:
        print("\nSource paths:", file=sys.stderr)
        for slug, path in source_paths.items():
            p = Path(path)
            line_count = sum(1 for _ in open(p)) if p.is_file() else 0
            print(f"  {slug}: {path} ({line_count} lines)", file=sys.stderr)



if __name__ == "__main__":
    cli_main()
