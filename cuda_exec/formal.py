"""Formal benchmark: atomic compile + trial ALL configs, ALL implementations.

Used by the Formal Evaluator (Judge) agent via the ik:bench skill.
Not for iterative development — use ik:exec for that.

Key differences from ik:exec:
- Compile + trial are bundled atomically (no separate steps)
- ALL configs are trialed (no cherry-picking)
- ALL implementations are trialed (or a specified subset)
- No profiling (that's ik:exec's job during iteration)
- Simplified metadata (no turn management)
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
from cuda_exec.main import compile_endpoint, trial_endpoint

logger = logging.getLogger(__name__)


def formal_benchmark(
    kernel: str,
    arch: str,
    *,
    impls: str | List[str] = "all",
    run_tag: str = "bench",
    version: str = "v1",
    direction_id: int = 0,
    timeout_seconds: int = 300,
) -> dict:
    """Atomic compile + trial ALL configs for specified implementations.

    Args:
        kernel: Kernel name (e.g., "fa4", "matmul", "vecadd")
        arch: GPU architecture (e.g., "sm90")
        impls: "all" or list of impl slugs (e.g., ["ref-cublas", "gen-cuda"])
        run_tag: Namespace tag for workspace isolation
        version: Version tag
        direction_id: Direction ID for workspace isolation
        timeout_seconds: Timeout per config

    Returns:
        Dict with per-implementation compile + trial results
    """
    configs = load_configs(kernel)
    resolved = resolve_impls(kernel, arch, impls)

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
            version=version,
            direction_id=direction_id,
            direction_slug=f"{kernel}-{gen['slug']}",
            turn=unique_turn,
        )

        if gen["file_type"] == "cu":
            # .cu needs compile
            # reference = primary_ref .py files + any .py gen impl files (for measurement)
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
            # .py gen implementations: compile with primary_ref as reference,
            # but pass gen's .py files as the reference (it becomes the "reference"
            # in the trial), and use a no-op .cu stub.
            # For now, skip .py gens in the compile→trial flow — they'll be
            # measured as part of the reference/cudnn side when another .cu is trialed.
            results[gen["slug"]] = {
                "impl": gen["slug"],
                "compile_ok": None,  # N/A — .py impl, no compile needed
                "trial_ok": None,    # measured as reference side in .cu trials
                "compile_result": None,
                "trial_result": None,
                "note": "Python impl — measured as reference/cudnn side when .cu impls are trialed",
            }
            continue

        # Trial ALL configs (only for .cu impls that were compiled)
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

    return {
        "kernel": kernel,
        "arch": arch,
        "num_configs": len(configs),
        "impls_requested": [r["slug"] for r in resolved],
        "refs": [r["slug"] for r in refs],
        "gens": [r["slug"] for r in gens],
        "results": results,
    }


def cli_main() -> None:
    """CLI entry point for ik:bench."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="formal_benchmark",
        description="Formal benchmark: atomic compile + trial ALL configs",
    )
    parser.add_argument("kernel", help="Kernel name (e.g., fa4, matmul, vecadd)")
    parser.add_argument("arch", help="GPU architecture (e.g., sm90)")
    parser.add_argument(
        "--impls", nargs="*", default=None,
        help="Implementation slugs (default: all). E.g., ref-cublas gen-cuda",
    )
    parser.add_argument("--run-tag", default="bench", help="Workspace namespace (default: bench)")
    parser.add_argument("--version", default="v1", help="Version tag (default: v1)")
    parser.add_argument("--direction-id", type=int, default=0, help="Direction ID (default: 0)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per config in seconds (default: 300)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    impl_arg = args.impls if args.impls else "all"

    result = formal_benchmark(
        kernel=args.kernel,
        arch=args.arch,
        impls=impl_arg,
        run_tag=args.run_tag,
        version=args.version,
        direction_id=args.direction_id,
        timeout_seconds=args.timeout,
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    cli_main()
