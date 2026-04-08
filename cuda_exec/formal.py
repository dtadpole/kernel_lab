"""Formal benchmark: atomic compile + trial ALL configs, ALL implementations.

Used by the Formal Evaluator (Judge) agent via the ik:bench skill.
Not for iterative development — use ik:exec for that.

Key differences from ik:exec:
- Compile + trial are bundled atomically (no separate steps)
- ALL configs are trialed (no cherry-picking)
- ALL implementations are trialed (or a specified subset)
- No profiling (that's ik:exec's job during iteration)
- Simplified metadata (no turn management)
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
from cuda_exec.tasks import compile_endpoint, trial_endpoint

logger = logging.getLogger(__name__)


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
                median_ms = impl_data.get("performance", {}).get("latency_ms", {}).get("median")
            elif "performance" in entry:
                # py impl: configs[c]["performance"]
                median_ms = entry.get("performance", {}).get("latency_ms", {}).get("median")
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

    return "\n".join(lines)


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

    # Golden = first ref (alphabetical). Used for speedup ratios.
    # Correctness is only valid between impls sharing the same input
    # generation path (.cu↔.cu via C harness, .py↔.py via generate_inputs).
    primary_ref = refs[0]

    results: Dict[str, dict] = {}

    # --- Helper: load and run a .py impl, return per-config performance + output ---
    def _run_py_impl(impl: dict) -> dict:
        import torch
        from cuda_exec.scripts.eval_support import (
            measure_reference, load_reference_module, set_seed, generate_inputs,
            DEFAULT_SEED,
        )
        device = torch.device("cuda")
        import tempfile as _tempfile, sys as _sys
        tmp_dir = Path(_tempfile.mkdtemp(prefix=f"bench_{impl['slug']}_"))
        for fname, content in impl["files"].items():
            (tmp_dir / fname).write_text(content, encoding="utf-8")
        entry_py = tmp_dir / f"{impl['name']}.py"
        old_path = list(_sys.path)
        _sys.path.insert(0, str(tmp_dir))
        try:
            mod = load_reference_module(entry_py)
            config_results = {}
            for config_slug, config in configs.items():
                try:
                    result = measure_reference(mod, config, device, num_trials=10)
                    config_results[config_slug] = {
                        "status": "ok",
                        "performance": result["performance"],
                        "output_tensor": result.get("output_tensor"),
                    }
                except Exception as exc:
                    config_results[config_slug] = {
                        "status": "error",
                        "error": str(exc),
                    }
            return {"ok": True, "configs": config_results}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            _sys.path[:] = old_path

    # Split ALL impls by file type, not by source.
    # .py impls (ref, peak, or gen) → measured via _run_py_impl
    # .cu impls (ref, peak, or gen) → compiled + trialed via cuda_exec
    all_impls = refs + gens
    py_impls = [i for i in all_impls if i["file_type"] == "py"]
    cu_impls = [i for i in all_impls if i["file_type"] == "cu"]

    # --- Benchmark each .py impl (any source) ---
    for impl in py_impls:
        logger.info("[%s] measure_py start (%d configs)", impl["slug"], len(configs))
        impl_start = time.time()
        r = _run_py_impl(impl)
        if r["ok"]:
            for cs, cr in r["configs"].items():
                cr.pop("output_tensor", None)
            results[impl["slug"]] = {
                "impl": impl["slug"],
                "compile_ok": None, "trial_ok": True,
                "compile_result": None,
                "trial_result": {"all_ok": True, "configs": r["configs"]},
            }
        else:
            results[impl["slug"]] = {
                "impl": impl["slug"], "compile_ok": None, "trial_ok": False,
                "error": r.get("error", ""),
            }
        logger.info("[%s] measure_py done (%.1fs)", impl["slug"], time.time() - impl_start)

    # --- Compile + trial each .cu impl (any source) ---
    for gen in cu_impls:
        unique_turn = int(time.time()) % 100000

        metadata = Metadata(
            run_tag=run_tag,
            version="v1",
            direction_id=0,
            direction_slug=f"{kernel}-{gen['slug']}",
            turn=unique_turn,
        )

        if gen["file_type"] == "cu":
            # .cu needs compile — build impl-keyed request
            compile_impls = {}
            # Include .py impls as references for correctness comparison
            for py_impl in py_impls:
                compile_impls[py_impl["slug"]] = dict(py_impl["files"])
            # Include this .cu gen impl
            compile_impls[gen["slug"]] = dict(gen["files"])

            logger.info("[%s] compile start", gen["slug"])
            compile_start = time.time()
            compile_req = CompileRequest(
                metadata=metadata,
                timeout_seconds=timeout_seconds,
                impls=compile_impls,
            )
            compile_resp = compile_endpoint(compile_req)
            compile_result = compile_resp.model_dump(mode="json")
            logger.info("[%s] compile done (%.1fs)", gen["slug"], time.time() - compile_start)

            if not compile_resp.all_ok:
                results[gen["slug"]] = {
                    "impl": gen["slug"],
                    "compile_ok": False,
                    "trial_ok": False,
                    "compile_result": compile_result,
                    "trial_result": None,
                }
                continue
        else:
            # .py gen impl: already benchmarked above in the ref+py loop
            continue

        # Trial ALL configs
        logger.info("[%s] trial start (%d configs)", gen["slug"], len(configs))
        trial_start = time.time()
        trial_req = TrialRequest(
            metadata=metadata,
            timeout_seconds=timeout_seconds,
            configs=configs,
        )
        trial_resp = trial_endpoint(trial_req)
        trial_result = trial_resp.model_dump(mode="json")
        logger.info("[%s] trial done (%.1fs)", gen["slug"], time.time() - trial_start)

        results[gen["slug"]] = {
            "impl": gen["slug"],
            "compile_ok": compile_result.get("all_ok", False),
            "trial_ok": trial_resp.all_ok,
            "compile_result": compile_result,
            "trial_result": trial_result,
        }

    bench_result = {
        "kernel": kernel,
        "arch": arch,
        "num_configs": len(configs),
        "impls_requested": [r["slug"] for r in resolved],
        "refs": [r["slug"] for r in refs],
        "gens": [r["slug"] for r in gens],
        "source_paths": {r["slug"]: str(Path(r["entry_point"]).resolve()) for r in resolved},
        "results": results,
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

    impls = bench_cfg.impls
    if isinstance(impls, str) and impls != "all":
        impls = [impls]
    elif hasattr(impls, "__iter__") and not isinstance(impls, str):
        impls = list(impls)

    result = formal_benchmark(
        kernel=bench_cfg.kernel,
        arch=bench_cfg.arch,
        run_tag=bench_cfg.get("run_tag"),
        impls=impls,
        timeout_seconds=bench_cfg.timeout,
        kb_repo=bench_cfg.get("kb_repo"),
        runtime_root=bench_cfg.get("runtime_root"),
        data_root=bench_cfg.get("data_root"),
    )

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
