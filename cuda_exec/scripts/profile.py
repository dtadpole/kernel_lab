#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from _cli_common import add_metadata_args, ensure_repo_root_on_path

ensure_repo_root_on_path()

from cuda_exec.models import Metadata  # noqa: E402
from cuda_exec.runner import resolve_workspace_bundle  # noqa: E402
from cuda_exec.tasks import _config_env, _primary_artifact_from_manifest, _slugify  # noqa: E402
from cuda_exec.scripts.eval_support import (  # noqa: E402
    acquire_device_lock,
    extract_config_payload,
    gpu_cleanup,
    load_reference_entry,
    load_reference_module,
    measure_reference,
    release_device_lock,
    watchdog_handler,
)


VALID_MODES = {"reference_only", "generated_only", "dual"}


def _run_generated(target_path: Path, env: dict[str, str], workspace_path: str, timeout_seconds: int) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        [str(target_path)],
        cwd=workspace_path,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    duration_ms = (time.perf_counter() - started) * 1000.0
    stdout = completed.stdout.strip()
    payload = json.loads(stdout) if stdout else {}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "payload": payload,
        "duration_ms": duration_ms,
    }


def _generated_summary(generated_payload: dict[str, Any], duration_ms: float, config: dict[str, Any]) -> dict[str, Any]:
    performance = generated_payload.get("performance", {}) if isinstance(generated_payload, dict) else {}
    latency_ms = performance.get("latency_ms", {}) if isinstance(performance, dict) else {}
    if latency_ms:
        return {
            "metadata": {**config, **performance.get("metadata", {})},
            "latency_ms": latency_ms,
            "runs": performance.get("runs"),
        }
    return {
        "metadata": {"source": "process_duration_fallback", **config},
        "latency_ms": {
            "min": duration_ms,
            "median": duration_ms,
            "max": duration_ms,
            "mean": duration_ms,
        },
        "runs": 1,
    }


def _dual_summary(reference_summary: dict[str, Any] | None, generated_summary: dict[str, Any] | None, config: dict[str, Any]) -> dict[str, Any]:
    ref_median = None
    gen_median = None
    if reference_summary:
        ref_median = reference_summary.get("latency_ms", {}).get("median")
    if generated_summary:
        gen_median = generated_summary.get("latency_ms", {}).get("median")
    speedup = None
    if ref_median not in {None, 0} and gen_median not in {None, 0}:
        speedup = float(ref_median) / float(gen_median)
    return {
        "metadata": config,
        "latency_ms": {
            "min": gen_median,
            "median": gen_median,
            "max": gen_median,
            "mean": gen_median,
        },
        "runs": generated_summary.get("runs") if generated_summary else None,
        "comparison": {
            "reference_median_ms": ref_median,
            "generated_median_ms": gen_median,
            "speedup": speedup,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile runner for cuda_exec profile.")
    add_metadata_args(parser)
    parser.add_argument("--config-slug", required=True, help="Stable runtime config slug")
    parser.add_argument("--config-json", default="{}", help="Kernel-specific config payload as JSON object")
    parser.add_argument("--mode", default="generated_only", choices=sorted(VALID_MODES))
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

    device = torch.device("cuda")
    lock_fd: int | None = None

    old_handler = signal.signal(signal.SIGALRM, watchdog_handler)
    signal.alarm(args.timeout)

    try:
        lock_fd = acquire_device_lock(device)

        workspace = resolve_workspace_bundle(**metadata.model_dump())
        workspace_path = workspace["workspace_path"]
        target_path, _ = _primary_artifact_from_manifest(workspace)
        reference_root = Path(workspace_path) / "inputs" / "reference"
        reference_path = load_reference_entry(reference_root)
        config_rel = f"state/profile.inline.{_slugify(args.config_slug)}.json"
        env = _config_env(workspace, "profile", 1, args.config_slug, config, config_rel)
        reference_config = extract_config_payload(env["CUDA_EXEC_CONFIG_JSON"])

        reference_payload = None
        reference_summary = None
        generated_payload = None
        generated_summary = None

        if args.mode in {"reference_only", "dual"}:
            reference_module = load_reference_module(reference_path)
            measured = measure_reference(reference_module, reference_config, device=device)
            reference_summary = {
                **measured["performance"],
                "metadata": reference_config,
            }
            reference_payload = {
                "output": {"metadata": reference_config},
                "summary": reference_summary,
            }

        if args.mode in {"generated_only", "dual"}:
            generated_run = _run_generated(target_path, env, workspace_path, args.timeout)
            generated_summary = _generated_summary(generated_run["payload"], generated_run["duration_ms"], reference_config)
            generated_payload = {
                "output": generated_run["payload"].get("output", {}),
                "summary": generated_summary,
                "logs": {
                    "stdout": generated_run["stdout"],
                    "stderr": generated_run["stderr"],
                },
            }

        if args.mode == "reference_only":
            summary = reference_summary
        elif args.mode == "generated_only":
            summary = generated_summary
        else:
            summary = _dual_summary(reference_summary, generated_summary, reference_config)

        result = {
            "metadata": metadata.model_dump(),
            "config_slug": args.config_slug,
            "mode": args.mode,
            "status": "ok",
            "reference": reference_payload,
            "generated": generated_payload,
            "summary": summary,
        }
        print(json.dumps(result, indent=2))
        return 0

    except TimeoutError:
        error_result = {
            "metadata": metadata.model_dump(),
            "config_slug": args.config_slug,
            "mode": args.mode,
            "status": "timeout",
            "error": "profile watchdog timeout expired",
        }
        print(json.dumps(error_result, indent=2))
        return 1

    except Exception as exc:
        error_result = {
            "metadata": metadata.model_dump(),
            "config_slug": args.config_slug,
            "mode": args.mode,
            "status": "error",
            "error": str(exc),
        }
        print(json.dumps(error_result, indent=2))
        return 1

    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        try:
            gpu_cleanup(device)
        except Exception:
            pass

        release_device_lock(lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
