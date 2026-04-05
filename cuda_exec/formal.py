"""Formal benchmark: atomic compile + trial ALL configs.

Used by the Formal Evaluator (Judge) agent via the ik:bench skill.
Not for iterative development — use ik:exec for that.

Key differences from ik:exec:
- Compile + trial are bundled atomically (no separate steps)
- ALL configs are trialed (loaded from fixture, no cherry-picking)
- No profiling (that's ik:exec's job during iteration)
- Simplified metadata (no turn management)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from cuda_exec.models import (
    CompileRequest,
    TrialRequest,
    Metadata,
)
from cuda_exec.main import compile_endpoint, trial_endpoint

logger = logging.getLogger(__name__)

# Project root for resolving fixture paths
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_all_configs(kernel: str, arch: str) -> Dict[str, Dict[str, Any]]:
    """Load ALL configs from the fixture. No selection, no filtering."""
    configs_path = _PROJECT_ROOT / "data" / "fixtures" / arch / kernel / "configs.json"
    if not configs_path.exists():
        raise FileNotFoundError(
            f"Configs not found: {configs_path}. "
            f"Available fixtures: {list((_PROJECT_ROOT / 'data' / 'fixtures' / arch).glob('*/configs.json'))}"
        )
    with open(configs_path, encoding="utf-8") as f:
        configs = json.load(f)
    if not configs:
        raise ValueError(f"Empty configs file: {configs_path}")
    return configs


def _load_source_files(kernel: str, arch: str) -> tuple[Dict[str, str], Dict[str, str], Dict[str, str] | None]:
    """Load reference, generated, and optional cuDNN source files from fixture/generated paths."""
    fixture_dir = _PROJECT_ROOT / "data" / "fixtures" / arch / kernel
    generated_dir = _PROJECT_ROOT / "data" / "generated" / arch / kernel

    # Reference files
    reference_files: Dict[str, str] = {}
    if fixture_dir.exists():
        for f in fixture_dir.iterdir():
            if f.is_file() and f.suffix in (".py", ".cu", ".h", ".cuh"):
                if f.name == "configs.json":
                    continue
                reference_files[f.name] = f.read_text(encoding="utf-8")
    if "cutedsl.py" not in reference_files:
        raise FileNotFoundError(f"cutedsl.py not found in {fixture_dir}")

    # Generated files
    generated_files: Dict[str, str] = {}
    generated_cu = generated_dir / "generated.cu"
    if not generated_cu.exists():
        raise FileNotFoundError(f"generated.cu not found at {generated_cu}")
    generated_files["generated.cu"] = generated_cu.read_text(encoding="utf-8")
    # Include any headers in the generated dir
    for f in generated_dir.iterdir():
        if f.is_file() and f.suffix in (".h", ".cuh") and f.name != "generated.cu":
            generated_files[f.name] = f.read_text(encoding="utf-8")

    # Optional cuDNN files
    cudnn_files: Dict[str, str] | None = None
    cudnn_py = fixture_dir / "cudnn.py"
    if cudnn_py.exists():
        cudnn_files = {"cudnn.py": cudnn_py.read_text(encoding="utf-8")}

    return reference_files, generated_files, cudnn_files


def formal_benchmark(
    kernel: str,
    arch: str,
    *,
    run_tag: str = "bench",
    version: str = "v1",
    direction_id: int = 0,
    timeout_seconds: int = 300,
) -> dict:
    """Atomic compile + trial ALL configs. Returns combined result.

    Args:
        kernel: Kernel name (e.g., "fa4", "matmul", "vecadd")
        arch: GPU architecture (e.g., "sm90")
        run_tag: Namespace tag for workspace isolation
        version: Version tag
        direction_id: Direction ID for workspace isolation
        timeout_seconds: Timeout per config

    Returns:
        Dict with compile_ok, trial_ok, compile_result, trial_result
    """
    # Load everything from fixtures
    configs = _load_all_configs(kernel, arch)
    reference_files, generated_files, cudnn_files = _load_source_files(kernel, arch)

    metadata = Metadata(
        run_tag=run_tag,
        version=version,
        direction_id=direction_id,
        direction_slug=kernel,
        turn=0,  # bench uses turn=0 (single-shot, no iteration)
    )

    # Step 1: Compile
    compile_req = CompileRequest(
        metadata=metadata,
        timeout_seconds=timeout_seconds,
        reference_files=reference_files,
        generated_files=generated_files,
        cudnn_files=cudnn_files or {},
    )
    compile_resp = compile_endpoint(compile_req)
    compile_result = compile_resp.model_dump(mode="json")

    if not compile_resp.all_ok:
        return {
            "compile_ok": False,
            "trial_ok": False,
            "kernel": kernel,
            "arch": arch,
            "num_configs": len(configs),
            "compile_result": compile_result,
            "trial_result": None,
        }

    # Step 2: Trial ALL configs
    trial_req = TrialRequest(
        metadata=metadata,
        timeout_seconds=timeout_seconds,
        configs=configs,
    )
    trial_resp = trial_endpoint(trial_req)
    trial_result = trial_resp.model_dump(mode="json")

    return {
        "compile_ok": True,
        "trial_ok": trial_resp.all_ok,
        "kernel": kernel,
        "arch": arch,
        "num_configs": len(configs),
        "compile_result": compile_result,
        "trial_result": trial_result,
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

    result = formal_benchmark(
        kernel=args.kernel,
        arch=args.arch,
        run_tag=args.run_tag,
        version=args.version,
        direction_id=args.direction_id,
        timeout_seconds=args.timeout,
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    cli_main()
