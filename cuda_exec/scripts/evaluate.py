#!/usr/bin/env python3
"""Comparison runner for cuda_exec evaluate stage.

Aligned with kbEvalCli.py patterns:
- CUDA event timing (matching time_execution_with_cuda_event)
- allclose correctness with shape check (matching verify_correctness)
- Seed control and torch.no_grad (matching eval_kernel_custom)
- Device locking (matching FileLock)
- Watchdog timeout (matching ArmableWatchdog concept)
- GPU cleanup (matching graceful_eval_cleanup)
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import signal
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from _cli_common import add_metadata_args, ensure_repo_root_on_path

ensure_repo_root_on_path()

from cuda_exec.models import Metadata  # noqa: E402
from cuda_exec.runner import resolve_workspace_bundle  # noqa: E402
from cuda_exec.tasks import _config_env, _primary_artifact_from_manifest, _slugify  # noqa: E402

# ---------------------------------------------------------------------------
# Alignment defaults (matching kbEvalCli.py / kbEvalUtil.py defaults)
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42
NUM_CORRECTNESS_TRIALS = 3  # kbEvalCli num_verify_trials
NUM_WARMUP_RUNS = 5  # kbEvalCli num_warmups
NUM_PERF_TRIALS = 10  # kbEvalCli num_perf_trials
ATOL = 1e-02  # kbEvalCli verify_correctness atol
RTOL = 1e-02  # kbEvalCli verify_correctness rtol


# ---------------------------------------------------------------------------
# Seed control (matching kbEvalUtil.set_seed)
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    """Set CPU and GPU random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# ---------------------------------------------------------------------------
# Device locking (matching kbEvalCli FileLock pattern)
# ---------------------------------------------------------------------------

def _acquire_device_lock(device: torch.device) -> int | None:
    """Acquire exclusive fcntl lock for a CUDA device.

    Returns the lock file descriptor on success.  Raises ``RuntimeError``
    if the device is already locked by another process.
    """
    lock_dir = Path.home() / ".cuda_exec"
    lock_dir.mkdir(parents=True, exist_ok=True)
    device_index = device.index if device.index is not None else 0
    lock_path = lock_dir / f".lock_cuda_{device_index}"
    try:
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        return fd
    except BlockingIOError:
        raise RuntimeError(
            f"CUDA device {device_index} is locked by another process. "
            f"Lock file: {lock_path}"
        )


def _release_device_lock(lock_fd: int | None) -> None:
    """Release a previously-acquired device lock."""
    if lock_fd is not None:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# GPU cleanup (matching kbEvalUtil.graceful_eval_cleanup)
# ---------------------------------------------------------------------------

def _gpu_cleanup(device: torch.device) -> None:
    """Clear CUDA cache, reset memory stats, and synchronize."""
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize(device=device)


# ---------------------------------------------------------------------------
# Watchdog (simplified ArmableWatchdog — evaluate.py is already a subprocess)
# ---------------------------------------------------------------------------

def _watchdog_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("evaluate watchdog timeout expired")


# ---------------------------------------------------------------------------
# Shape inference from nested lists
# ---------------------------------------------------------------------------

def _infer_shape(value: Any) -> tuple[int, ...]:
    """Infer tensor shape from a nested list structure."""
    if isinstance(value, (int, float)):
        return ()
    if isinstance(value, list):
        if not value:
            return (0,)
        return (len(value),) + _infer_shape(value[0])
    return ()


# ---------------------------------------------------------------------------
# allclose on flattened numeric data
# ---------------------------------------------------------------------------

def _allclose_check(
    ref_values: list[float],
    gen_values: list[float],
    atol: float = ATOL,
    rtol: float = RTOL,
) -> tuple[bool, float, float]:
    """Check allclose on flattened numeric values.

    Returns ``(passed, max_diff, avg_diff)``.
    Matches ``torch.allclose`` semantics: ``abs(a - b) <= atol + rtol * abs(b)``.
    """
    if not ref_values and not gen_values:
        return True, 0.0, 0.0

    abs_diffs = [abs(a - b) for a, b in zip(ref_values, gen_values)]
    tolerances = [atol + rtol * abs(b) for b in gen_values]
    passed = all(d <= t for d, t in zip(abs_diffs, tolerances))
    max_diff = max(abs_diffs) if abs_diffs else 0.0
    avg_diff = statistics.fmean(abs_diffs) if abs_diffs else 0.0
    return passed, max_diff, avg_diff


# ---------------------------------------------------------------------------
# Reference loading (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tensor serialisation helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Reference measurement — CUDA event timing (aligned with kbEvalCli)
# ---------------------------------------------------------------------------

def _measure_reference(
    module: Any,
    config: dict[str, Any],
    device: torch.device,
    seed: int = DEFAULT_SEED,
    num_warmups: int = NUM_WARMUP_RUNS,
    num_trials: int = NUM_PERF_TRIALS,
) -> dict[str, Any]:
    """Measure reference model performance using CUDA event timing.

    Aligned with kbEvalCli's ``time_execution_with_cuda_event`` pattern:

    - Seed control before instantiation
    - Explicit ``.cuda(device)`` on init_inputs, model, and inputs
    - ``torch.no_grad()`` context
    - Warmup runs before measurement
    - ``torch.cuda.Event`` timing for each trial
    """
    model_cls, get_inputs, get_init_inputs = _normalize_reference_contract(module)

    hardware = torch.cuda.get_device_name(device=device)

    with torch.no_grad(), torch.cuda.device(device):
        # Instantiate model with seed and explicit device placement
        _set_seed(seed)
        init_inputs = list(get_init_inputs())
        init_inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x
            for x in init_inputs
        ]
        model = model_cls(*init_inputs)
        model = model.cuda(device=device)

        # Prepare inputs with explicit device placement
        inputs = list(get_inputs(config))
        inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]

        # Warmup runs (matching kbEvalCli warmup pattern)
        for _ in range(num_warmups):
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            model(*inputs)
            end_ev.record()
            torch.cuda.synchronize(device=device)

        # Measurement runs with CUDA event timing
        latencies_ms: list[float] = []
        last_output: Any = None
        for _ in range(num_trials):
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            last_output = model(*inputs)
            end_ev.record()
            torch.cuda.synchronize(device=device)
            latencies_ms.append(start_ev.elapsed_time(end_ev))

    if last_output is None:
        raise RuntimeError("reference contract did not produce an output")

    output_json = _tensor_to_jsonable(last_output)

    # Stats matching kbEvalUtil.get_timing_stats
    std_val = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0

    return {
        "output": output_json,
        "performance": {
            "metadata": {
                "hardware": hardware,
                "device": str(device),
            },
            "latency_ms": {
                "min": min(latencies_ms),
                "median": statistics.median(latencies_ms),
                "max": max(latencies_ms),
                "mean": statistics.fmean(latencies_ms),
                "std": std_val,
            },
            "runs": len(latencies_ms),
        },
    }


# ---------------------------------------------------------------------------
# Generated binary execution (unchanged)
# ---------------------------------------------------------------------------

def _run_generated(
    target_path: Path,
    env: dict[str, str],
    workspace_path: str,
    timeout_seconds: int,
) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Correctness verification (aligned with kbEvalCli verify_correctness)
# ---------------------------------------------------------------------------

def _verify_correctness(
    module: Any,
    config: dict[str, Any],
    generated_payload: dict[str, Any],
    device: torch.device,
    num_trials: int = NUM_CORRECTNESS_TRIALS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Verify correctness with shape check and allclose tolerance.

    Aligned with kbEvalCli's ``verify_correctness``:

    - Shape check before value comparison
    - ``allclose`` with ``atol=1e-02``, ``rtol=1e-02``
    - Multi-trial with seed rotation
    - Reports ``trials``, ``passed_trials``, ``max_diff``, ``avg_diff``,
      ``output_shape``
    """
    model_cls, get_inputs, get_init_inputs = _normalize_reference_contract(module)

    gen_output = generated_payload.get("output", {}).get("result", [])
    gen_shape = _infer_shape(gen_output)
    gen_values = _flatten_numeric(gen_output) if gen_output else []

    # Generate trial seeds deterministically (matching kbEvalCli pattern)
    torch.manual_seed(seed)
    trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_trials)
    ]

    passed_trials = 0
    worst_max_diff = 0.0
    worst_avg_diff = 0.0
    output_shape_str = ""

    with torch.no_grad(), torch.cuda.device(device):
        for trial_seed in trial_seeds:
            _set_seed(trial_seed)
            init_inputs = list(get_init_inputs())
            init_inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in init_inputs
            ]

            _set_seed(trial_seed)
            model = model_cls(*init_inputs)
            model = model.cuda(device=device)

            _set_seed(trial_seed)
            inputs = list(get_inputs(config))
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]

            ref_output = model(*inputs)
            torch.cuda.synchronize(device=device)

            ref_json = _tensor_to_jsonable(ref_output)
            ref_shape = _infer_shape(ref_json)
            output_shape_str = "x".join(str(d) for d in ref_shape) if ref_shape else "scalar"

            # Shape check first (matching kbEvalCli shape mismatch check)
            if ref_shape != gen_shape:
                # Shape mismatch — remaining trials cannot pass either
                return {
                    "passed": False,
                    "reason": f"shape mismatch: reference={ref_shape} generated={gen_shape}",
                    "output_shape": output_shape_str,
                    "max_abs_error": None,
                    "mean_abs_error": None,
                    "trials": f"{passed_trials}/{num_trials}",
                    "total_trials": num_trials,
                    "passed_trials": passed_trials,
                }

            ref_values = _flatten_numeric(ref_json)
            if len(ref_values) != len(gen_values):
                worst_max_diff = float("inf")
                continue

            passed, max_diff, avg_diff = _allclose_check(ref_values, gen_values)
            if passed:
                passed_trials += 1
            worst_max_diff = max(worst_max_diff, max_diff)
            worst_avg_diff = max(worst_avg_diff, avg_diff)

    all_passed = passed_trials == num_trials
    return {
        "passed": all_passed,
        "output_shape": output_shape_str,
        "max_abs_error": worst_max_diff,
        "mean_abs_error": worst_avg_diff,
        "trials": f"{passed_trials}/{num_trials}",
        "total_trials": num_trials,
        "passed_trials": passed_trials,
    }


# ---------------------------------------------------------------------------
# Comparison payload builder
# ---------------------------------------------------------------------------

def _comparison_payload(
    correctness: dict[str, Any],
    reference_performance: dict[str, Any],
    generated_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build comparison payload with correctness and performance summaries."""
    ref_perf = reference_performance.get("latency_ms", {})
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Comparison runner for cuda_exec evaluate. Runs reference and generated for one config.",
    )
    add_metadata_args(parser)
    parser.add_argument("--config-slug", required=True, help="Stable runtime config slug")
    parser.add_argument("--config-json", default="{}", help="Kernel-specific config payload as JSON object")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    parser.add_argument("--num-warmups", type=int, default=NUM_WARMUP_RUNS, help="Warmup runs before timing")
    parser.add_argument("--num-perf-trials", type=int, default=NUM_PERF_TRIALS, help="Measurement runs for timing")
    parser.add_argument("--num-correctness-trials", type=int, default=NUM_CORRECTNESS_TRIALS, help="Correctness verification trials")
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

    # Install watchdog (matching ArmableWatchdog concept)
    old_handler = signal.signal(signal.SIGALRM, _watchdog_handler)
    signal.alarm(args.timeout)

    try:
        # Acquire device lock (matching kbEvalCli FileLock pattern)
        lock_fd = _acquire_device_lock(device)

        workspace = resolve_workspace_bundle(**metadata.model_dump())
        workspace_path = workspace["workspace_path"]
        target_path, _ = _primary_artifact_from_manifest(workspace)
        reference_root = Path(workspace_path) / "inputs" / "reference"
        reference_path = _load_reference_entry(reference_root)
        config_rel = f"state/evaluate.inline.{_slugify(args.config_slug)}.json"
        env = _config_env(workspace, "evaluate", 1, args.config_slug, config, config_rel)

        reference_module = _load_reference_module(reference_path)
        reference_config = _extract_config_payload(env["CUDA_EXEC_CONFIG_JSON"])

        # --- Performance measurement (CUDA event timing) ---
        reference_result = _measure_reference(
            reference_module,
            reference_config,
            device=device,
            seed=args.seed,
            num_warmups=args.num_warmups,
            num_trials=args.num_perf_trials,
        )

        # --- Run generated binary ---
        generated_run = _run_generated(target_path, env, workspace_path, args.timeout)
        generated_payload = generated_run["payload"]

        # --- Correctness verification (multi-trial, shape check, allclose) ---
        correctness = _verify_correctness(
            reference_module,
            reference_config,
            generated_payload,
            device=device,
            num_trials=args.num_correctness_trials,
            seed=args.seed,
        )

        # --- Build comparison ---
        comparison = _comparison_payload(
            correctness,
            reference_result["performance"],
            generated_payload,
        )

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
                    "metadata": {
                        **reference_config,
                        **reference_result["performance"].get("metadata", {}),
                    },
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
            "comparison": comparison,
        }
        print(json.dumps(result, indent=2))
        return 0

    except TimeoutError:
        error_result = {
            "metadata": metadata.model_dump(),
            "config_slug": args.config_slug,
            "status": "timeout",
            "error": "evaluate watchdog timeout expired",
        }
        print(json.dumps(error_result, indent=2))
        return 1

    except Exception as exc:
        error_result = {
            "metadata": metadata.model_dump(),
            "config_slug": args.config_slug,
            "status": "error",
            "error": str(exc),
        }
        print(json.dumps(error_result, indent=2))
        return 1

    finally:
        # Cancel watchdog
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        # GPU cleanup (matching graceful_eval_cleanup)
        try:
            _gpu_cleanup(device)
        except Exception:
            pass

        # Release device lock
        _release_device_lock(lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
