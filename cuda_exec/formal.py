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
            metrics[slug][config_slug] = {"median_ms": median_ms, "correct": None}

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

def format_results_table(bench_result: dict) -> str:
    """Format the summary as a Markdown comparison table."""
    summary = bench_result.get("summary", {})
    if not summary:
        return ""

    gpu = summary["gpu"]
    peak = summary["peak_tflops"]
    golden = summary["golden"]
    impl_order = bench_result.get("impls_requested", [])
    config_order = list(next(iter(summary["impls"].values()))["configs"].keys()) if summary["impls"] else []

    # Header
    lines: List[str] = []
    hdr1 = f"| {'Config':<24} |"
    hdr2 = f"| {'':<24} |"
    sep = f"|{'-' * 26}|"
    for slug in impl_order:
        label = slug
        sub = "ms    TFLOPS"
        if slug != golden:
            sub += "   ×"
        hdr1 += f" {label:<20} |"
        hdr2 += f" {sub:<20} |"
        sep += f"{'-' * 22}|"
    lines.append(f"**{gpu}** — peak {peak} TFLOPS (BF16 TC)")
    lines.append("")
    lines.append(hdr1)
    lines.append(hdr2)
    lines.append(sep)

    # Data rows
    for config_slug in config_order:
        row = f"| {config_slug:<24} |"
        for slug in impl_order:
            entry = summary["impls"][slug]["configs"].get(config_slug, {})
            ms = entry.get("median_ms")
            tf = entry.get("tflops")
            spd = entry.get("speedup")
            ok = entry.get("correct")

            if ms is None:
                cell = "—"
            else:
                ms_s = f"{ms:.3f}" if ms < 1 else f"{ms:.2f}"
                tf_s = f"{tf:>6.1f}" if tf else "    —"
                cell = f"{ms_s:>6}  {tf_s}"
                if slug == golden:
                    pass  # no marker for golden
                elif ok is True:
                    cell += " ✓"
                elif ok is False:
                    cell += " ✗"
                else:
                    cell += "  "
                if slug != golden and spd is not None:
                    cell += f" {spd:.1f}×"
            row += f" {cell:<20} |"
        lines.append(row)

    # % of peak row
    pct_row = f"| {'% of peak':<24} |"
    for slug in impl_order:
        pct = summary["impls"][slug]["best_pct_peak"]
        cell = f"{pct:>5.1f}%"
        pct_row += f" {cell:<20} |"
    lines.append(sep)
    lines.append(pct_row)

    # Footer
    lines.append("")
    lines.append(f"Golden: {golden} — ✓/✗ = correctness vs golden, ×  = speedup vs golden")

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

    # --- Phase 1: Snapshot ---
    run_dir = None
    snapshot_data = data_root_path  # fallback: use explicit data_root or None (= project data/)
    logger.info("Snapshot start [%s]", datetime.now().strftime(_ts_fmt))
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

    refs = [r for r in resolved if r["source"] == "ref"]
    gens = [r for r in resolved if r["source"] == "gen"]

    # Use the first reference as the primary reference for correctness comparison
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

    # --- Benchmark each ref-* and .py gen-* impl independently ---
    # Correctness is NOT computed here — trial.py handles it against the golden.
    for impl in refs + [g for g in gens if g["file_type"] == "py"]:
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

    # --- Trial each gen-* implementation ---
    for gen in gens:
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
            # Include all ref impls
            for ref in refs:
                compile_impls[ref["slug"]] = dict(ref["files"])
            # Include all .py gen impls
            for pg in gens:
                if pg["file_type"] == "py":
                    compile_impls[pg["slug"]] = dict(pg["files"])
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


if __name__ == "__main__":
    cli_main()
