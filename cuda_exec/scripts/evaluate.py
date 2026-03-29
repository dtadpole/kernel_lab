#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch.nn as nn

from _cli_common import add_metadata_args, ensure_repo_root_on_path

ensure_repo_root_on_path()

from cuda_exec.models import Metadata  # noqa: E402
from cuda_exec.runner import resolve_workspace_bundle  # noqa: E402
from cuda_exec.tasks import _config_env, _primary_artifact_from_manifest, _slugify  # noqa: E402


def _load_reference_entry(reference_root: Path) -> Path:
    candidates = sorted(reference_root.rglob("reference.py"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"reference execution requires exactly one reference.py under {reference_root}; found {len(candidates)}"
        )
    return candidates[0]


def _load_reference_module(reference_path: Path):
    spec = importlib.util.spec_from_file_location("cuda_exec_reference_module", reference_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load reference module from {reference_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_reference_contract(module: Any) -> tuple[Any, Any, Any]:
    model_cls = getattr(module, "Model", None)
    get_inputs = getattr(module, "get_inputs", None)
    get_init_inputs = getattr(module, "get_init_inputs", None)
    if model_cls is None or not callable(get_inputs) or not callable(get_init_inputs):
        raise RuntimeError(
            "reference module contract requires Model, get_inputs(config), and get_init_inputs()"
        )
    if not isinstance(model_cls, type) or not issubclass(model_cls, nn.Module):
        raise RuntimeError("reference module Model must be a subclass of torch.nn.Module")
    return model_cls, get_inputs, get_init_inputs


def _extract_config_payload(env_json: str) -> dict[str, Any]:
    payload = json.loads(env_json)
    if not isinstance(payload, dict):
        raise RuntimeError("CUDA_EXEC_CONFIG_JSON must decode to a JSON object")
    params = payload.get("params")
    if not isinstance(params, dict):
        raise RuntimeError("CUDA_EXEC_CONFIG_JSON must include object field 'params'")
    return params


def _tensor_to_jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_tensor_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _tensor_to_jsonable(v) for k, v in value.items()}
    return value


def _flatten_numeric(value: Any) -> list[float]:
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_numeric(item))
        return out
    if isinstance(value, (int, float)):
        return [float(value)]
    raise RuntimeError(f"reference output contains non-numeric value {value!r}")


def _measure_reference(module: Any, config: dict[str, Any], runs: int = 5) -> dict[str, Any]:
    model_cls, get_inputs, get_init_inputs = _normalize_reference_contract(module)
    model = model_cls(*list(get_init_inputs()))
    latencies_ms: list[float] = []
    last_output: Any = None
    for _ in range(runs):
        inputs = list(get_inputs(config))
        started = time.perf_counter()
        last_output = model(*inputs)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
    if last_output is None:
        raise RuntimeError("reference contract did not produce an output")
    output_json = _tensor_to_jsonable(last_output)
    return {
        "output": output_json,
        "performance": {
            "metadata": {},
            "latency_ms": {
                "min": min(latencies_ms),
                "median": statistics.median(latencies_ms),
                "max": max(latencies_ms),
                "mean": statistics.fmean(latencies_ms),
            },
            "runs": len(latencies_ms),
        },
    }


def _run_generated(target_path: Path, env: dict[str, str], workspace_path: str, timeout_seconds: int) -> dict[str, Any]:
    completed = subprocess.run(
        [str(target_path)],
        cwd=workspace_path,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError("generated execution produced empty stdout")
    payload = json.loads(stdout)
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "payload": payload,
    }


def _comparison_payload(reference_output: Any, generated_payload: dict[str, Any]) -> dict[str, Any]:
    ref_values = _flatten_numeric(reference_output)
    gen_values = _flatten_numeric(generated_payload.get("output", {}).get("result", []))
    if len(ref_values) != len(gen_values):
        correctness = {
            "passed": False,
            "reason": f"length mismatch: reference={len(ref_values)} generated={len(gen_values)}",
        }
    else:
        abs_errors = [abs(a - b) for a, b in zip(ref_values, gen_values)]
        max_abs_error = max(abs_errors) if abs_errors else 0.0
        mean_abs_error = statistics.fmean(abs_errors) if abs_errors else 0.0
        correctness = {
            "passed": max_abs_error <= 1e-5,
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
        }

    ref_perf = generated_payload.get("reference_performance", {})
    gen_perf = generated_payload.get("performance", {}).get("latency_ms", {})
    ref_median = float(ref_perf.get("median", 0.0) or 0.0)
    gen_median = float(gen_perf.get("median", 0.0) or 0.0)
    speedup = (ref_median / gen_median) if gen_median > 0 else None
    return {
        "correctness": correctness,
        "performance": {
            "reference_median_ms": ref_median,
            "generated_median_ms": gen_median,
            "delta_ms": gen_median - ref_median,
            "speedup": speedup,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Comparison runner for cuda_exec evaluate. Runs reference and generated for one config.",
    )
    add_metadata_args(parser)
    parser.add_argument("--config-slug", required=True, help="Stable runtime config slug")
    parser.add_argument("--config-json", default="{}", help="Kernel-specific config payload as JSON object")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    args = parser.parse_args()

    metadata = Metadata(
        run_tag=args.run_tag,
        version=args.version,
        direction_id=args.direction_id,
        direction_slug=args.direction_slug,
        turn=args.turn,
    )
    config = json.loads(args.config_json)
    if not isinstance(config, dict):
        raise SystemExit("--config-json must decode to a JSON object")

    workspace = resolve_workspace_bundle(**metadata.model_dump())
    workspace_path = workspace["workspace_path"]
    target_path, _ = _primary_artifact_from_manifest(workspace)
    reference_root = Path(workspace_path) / "inputs" / "reference"
    reference_path = _load_reference_entry(reference_root)
    config_rel = f"state/evaluate.inline.{_slugify(args.config_slug)}.json"
    env = _config_env(workspace, "evaluate", 1, args.config_slug, config, config_rel)

    reference_module = _load_reference_module(reference_path)
    reference_config = _extract_config_payload(env["CUDA_EXEC_CONFIG_JSON"])
    reference_result = _measure_reference(reference_module, reference_config)
    generated_run = _run_generated(target_path, env, workspace_path, args.timeout)
    generated_payload = generated_run["payload"]

    result = {
        "metadata": metadata.model_dump(),
        "config_slug": args.config_slug,
        "status": "ok",
        "reference": {
            "output": {
                "result": reference_result["output"],
                "metadata": reference_config,
            },
            "correctness": {
                "metadata": reference_config,
                "passed": True,
            },
            "performance": {
                **reference_result["performance"],
                "metadata": reference_config,
            },
        },
        "generated": {
            "output": generated_payload.get("output", {}),
            "correctness": {
                **generated_payload.get("correctness", {}),
                "metadata": {
                    **reference_config,
                    **generated_payload.get("correctness", {}).get("metadata", {}),
                },
            },
            "performance": {
                **generated_payload.get("performance", {}),
                "metadata": {
                    **reference_config,
                    **generated_payload.get("performance", {}).get("metadata", {}),
                },
            },
            "logs": {
                "stdout": generated_run["stdout"],
                "stderr": generated_run["stderr"],
            },
        },
        "comparison": _comparison_payload(reference_result["output"], generated_payload),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
