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


def formal_benchmark(
    kernel: str,
    arch: str,
    *,
    impls: str | List[str] = "all",
    timeout_seconds: int = 300,
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
    # --- Resolve paths from config ---
    import os
    kb_repo_path = Path(kb_repo).expanduser() if kb_repo else Path.home() / "kernel_lab_kb"
    runtime_root_path = Path(runtime_root).expanduser() if runtime_root else Path.home() / ".cuda_exec_bench"
    data_root_path = Path(data_root).expanduser() if data_root else None

    # --- Phase 1: Snapshot ---
    run_dir = None
    snapshot_data = data_root_path  # fallback: use explicit data_root or None (= project data/)
    try:
        from cuda_exec.trajectory import prepare_run
        run_dir = prepare_run(kernel, arch, impls, timeout_seconds, kb_repo=kb_repo_path)
        snapshot_data = run_dir / "data"
        logger.info("Snapshot written to %s", run_dir)
    except Exception as exc:
        logger.warning("Failed to prepare snapshot: %s — falling back to original data/", exc)

    # --- Phase 2: Resolve from snapshot (or original if snapshot failed) ---
    configs = load_configs(kernel, data_root=snapshot_data)
    resolved = resolve_impls(kernel, arch, impls, data_root=snapshot_data)

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

    # Trial each gen-* implementation
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
            # .cu needs compile
            ref_files = dict(primary_ref["files"])
            # Include .py gen impls as additional reference files
            py_gens = [g for g in gens if g["file_type"] == "py"]
            for pg in py_gens:
                ref_files.update(pg["files"])
            # cudnn = second ref (vendor baseline) if available
            cudnn = refs[1]["files"] if len(refs) > 1 else {}

            compile_req = CompileRequest(
                metadata=metadata,
                timeout_seconds=timeout_seconds,
                reference_files=ref_files,
                generated_files=gen["files"],
                cudnn_files=cudnn,
            )
            compile_resp = compile_endpoint(compile_req)
            compile_result = compile_resp.model_dump(mode="json")

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
            results[gen["slug"]] = {
                "impl": gen["slug"],
                "compile_ok": None,
                "trial_ok": None,
                "compile_result": None,
                "trial_result": None,
                "note": "Python impl — measured as reference/cudnn side when .cu impls are trialed",
            }
            continue

        # Trial ALL configs
        trial_req = TrialRequest(
            metadata=metadata,
            timeout_seconds=timeout_seconds,
            configs=configs,
        )
        trial_resp = trial_endpoint(trial_req)
        trial_result = trial_resp.model_dump(mode="json")

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
            finalize_run(run_dir, bench_result, kb_repo=kb_repo_path, runtime_root=bench_runtime)
            logger.info("Results finalized in %s", run_dir)
        except Exception as exc:
            logger.warning("Failed to finalize run: %s", exc)

    return bench_result


def cli_main() -> None:
    """CLI entry point for ik:bench (Hydra-based)."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
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
    impls = bench_cfg.impls
    if isinstance(impls, str) and impls != "all":
        impls = [impls]
    elif hasattr(impls, "__iter__") and not isinstance(impls, str):
        impls = list(impls)

    result = formal_benchmark(
        kernel=bench_cfg.kernel,
        arch=bench_cfg.arch,
        impls=impls,
        timeout_seconds=bench_cfg.timeout,
        kb_repo=bench_cfg.get("kb_repo"),
        runtime_root=bench_cfg.get("runtime_root"),
        data_root=bench_cfg.get("data_root"),
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    cli_main()
