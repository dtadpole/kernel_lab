#!/usr/bin/env python3
"""Comparison runner for cuda_exec trial stage.

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
import json
import os
import signal
import struct
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from _cli_common import add_metadata_args, ensure_repo_root_on_path

ensure_repo_root_on_path()

from cuda_exec.models import Metadata  # noqa: E402
from cuda_exec.runner import resolve_workspace_bundle  # noqa: E402
from cuda_exec.tasks import _config_env, _primary_artifact_from_manifest, _slugify  # noqa: E402
from cuda_exec.scripts.eval_support import (  # noqa: E402
    ATOL,
    RTOL,
    DEFAULT_SEED,
    NUM_CORRECTNESS_TRIALS,
    NUM_WARMUP_RUNS,
    NUM_PERF_TRIALS,
    set_seed,
    acquire_device_lock,
    release_device_lock,
    cleanup_lockfile,
    gpu_cleanup,
    watchdog_handler,
    load_reference_entry,
    load_cudnn_entry,
    load_reference_module,
    normalize_reference_contract,
    extract_config_payload,
    generate_inputs,
    tensor_to_jsonable,
    flatten_numeric,
    infer_shape,
    allclose_check,
    measure_reference,
)


# ---------------------------------------------------------------------------
# Generated binary execution
# ---------------------------------------------------------------------------

def _run_generated(
    target_path: Path,
    env: dict[str, str],
    workspace_path: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    # Create temp dir for binary output files from the harness
    output_dir = tempfile.mkdtemp(prefix="cuda_exec_output_")
    run_env = {**os.environ, **env, "CUDA_EXEC_OUTPUT_DIR": output_dir}

    completed = subprocess.run(
        [str(target_path)],
        cwd=workspace_path,
        env=run_env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError("generated execution produced empty stdout")
    payload = json.loads(stdout)

    # Keep binary output files for correctness verification.
    # Never inline large tensors into JSON — 8192×8192 = 67M floats = 2GB JSON.
    output_bin = Path(output_dir) / "output_0.bin"
    if output_bin.exists():
        payload.setdefault("output", {})["binary_path"] = str(output_bin)
        payload.setdefault("output", {})["result"] = []  # empty, data is in binary file

    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "payload": payload,
    }


def _bf16_to_float(bf16_uint16: int) -> float:
    """Convert a BF16 value (as uint16) to Python float."""
    # BF16 is the upper 16 bits of IEEE 754 float32
    fp32_bits = bf16_uint16 << 16
    return struct.unpack("<f", struct.pack("<I", fp32_bits))[0]


# ---------------------------------------------------------------------------
# Correctness verification (aligned with kbEvalCli verify_correctness)
# ---------------------------------------------------------------------------

def _harness_fill_random_bf16(count: int, seed: int) -> torch.Tensor:
    """Reproduce eval_harness.cu fill_random_bf16 PRNG in Python.

    Must match the C code EXACTLY (eval_harness.cu line ~112):
        h = idx ^ seed; h = (h^61)^(h>>16); h += h<<3;
        h ^= h>>4; h *= 0x27d4eb2d; h ^= h>>15;
        val = (float)(h & 0xFFFF) / 65536.0f - 0.5f;
        buf[idx] = __float2bfloat16(val);

    Range: [-0.5, 0.5) — kept small to avoid fp overflow in matmul.
    """
    import numpy as np
    idx = np.arange(count, dtype=np.uint32)
    h = idx ^ np.uint32(seed)
    h = (h ^ np.uint32(61)) ^ (h >> np.uint32(16))
    h = (h + (h << np.uint32(3))) & np.uint32(0xFFFFFFFF)
    h = h ^ (h >> np.uint32(4))
    h = (h * np.uint32(0x27d4eb2d)) & np.uint32(0xFFFFFFFF)
    h = h ^ (h >> np.uint32(15))
    f = (h & np.uint32(0xFFFF)).astype(np.float32) / 65536.0 - 0.5
    return torch.from_numpy(f).to(torch.bfloat16)


def _verify_correctness(
    module: Any,
    config: dict[str, Any],
    generated_payload: dict[str, Any],
    device: torch.device,
    num_trials: int = NUM_CORRECTNESS_TRIALS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    model_cls = getattr(module, "Model", None)
    get_init_inputs = getattr(module, "get_init_inputs", None)
    if model_cls is None:
        raise RuntimeError("reference module must export Model(nn.Module)")

    # Load generated output from binary file if available, otherwise from JSON
    gen_output_section = generated_payload.get("output", {})
    binary_path = gen_output_section.get("binary_path")
    if binary_path and Path(binary_path).exists():
        raw = Path(binary_path).read_bytes()
        n_elems = len(raw) // 2
        bf16_ints = struct.unpack(f"<{n_elems}H", raw)
        gen_values = [_bf16_to_float(v) for v in bf16_ints]
    else:
        gen_output = gen_output_section.get("result", [])
        gen_values = flatten_numeric(gen_output) if gen_output else []

    # The harness correctness pass uses fill_random_bf16 with seed 0xC0DE0000+j.
    # Reproduce this PRNG exactly in Python so the reference model gets identical inputs.
    input_size = int(config.get("input_size", 0))
    num_inputs = len(generate_inputs(config, device))  # determine input count
    shape = [int(v) for v in config.get("shape", [input_size])]

    passed_trials = 0
    worst_max_diff = 0.0
    worst_avg_diff = 0.0
    output_shape_str = ""

    with torch.no_grad(), torch.cuda.device(device):
        init_inputs = list(get_init_inputs()) if get_init_inputs else []
        init_inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x
            for x in init_inputs
        ]
        model = model_cls(*init_inputs)
        model = model.cuda(device=device)

        # Reproduce harness fill_random_bf16 PRNG for each input tensor
        # Seeds are 1, 2, 3, ... matching eval_harness.cu correctness pass.
        # generate_inputs may return non-tensor items (e.g. causal bool for FA4);
        # only generate PRNG tensors for the tensor positions, pass-through the rest.
        template_inputs = generate_inputs(config, device)
        harness_inputs = []
        tensor_idx = 0
        for item in template_inputs:
            if isinstance(item, torch.Tensor):
                correctness_seed = tensor_idx + 1
                t = _harness_fill_random_bf16(input_size, correctness_seed)
                harness_inputs.append(t.to(device).reshape(item.shape))
                tensor_idx += 1
            else:
                harness_inputs.append(item)  # pass-through bool, int, etc.

        ref_output = model(*harness_inputs)
        torch.cuda.synchronize(device=device)

        ref_json = tensor_to_jsonable(ref_output)
        ref_shape = infer_shape(ref_json)
        output_shape_str = "x".join(str(d) for d in ref_shape) if ref_shape else "scalar"

        ref_values = flatten_numeric(ref_json)
        if len(ref_values) != len(gen_values):
            worst_max_diff = float("inf")
        else:
            passed, max_diff, avg_diff = allclose_check(ref_values, gen_values)
            if passed:
                passed_trials = 1
            worst_max_diff = max(worst_max_diff, max_diff)
            worst_avg_diff = max(worst_avg_diff, avg_diff)

    all_passed = passed_trials == 1
    return {
        "passed": all_passed,
        "output_shape": output_shape_str,
        "max_abs_error": worst_max_diff,
        "mean_abs_error": worst_avg_diff,
        "trials": f"{passed_trials}/1",
        "total_trials": 1,
        "passed_trials": passed_trials,
    }


# ---------------------------------------------------------------------------
# Comparison payload builder
# ---------------------------------------------------------------------------

def _comparison_payload(
    correctness: dict[str, Any],
    reference_performance: dict[str, Any],
    generated_payload: dict[str, Any],
    cudnn_performance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ref_perf = reference_performance.get("latency_ms", {})
    gen_perf = generated_payload.get("performance", {}).get("latency_ms", {})
    ref_median = float(ref_perf.get("median", 0.0) or 0.0)
    gen_median = float(gen_perf.get("median", 0.0) or 0.0)
    speedup = (ref_median / gen_median) if gen_median > 0 else None

    perf: dict[str, Any] = {
        "reference_median_ms": ref_median,
        "generated_median_ms": gen_median,
        "delta_ms": gen_median - ref_median,
        "speedup": speedup,
    }

    if cudnn_performance is not None:
        cudnn_perf = cudnn_performance.get("latency_ms", {})
        cudnn_median = float(cudnn_perf.get("median", 0.0) or 0.0)
        perf["cudnn_median_ms"] = cudnn_median
        perf["speedup_vs_cudnn"] = (cudnn_median / gen_median) if gen_median > 0 else None

    return {
        "correctness": correctness,
        "performance": perf,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Comparison runner for cuda_exec trial. Runs reference and generated for one config.",
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

    old_handler = signal.signal(signal.SIGALRM, watchdog_handler)
    signal.alarm(args.timeout)

    def _ts() -> str:
        return datetime.now().strftime("%H:%M:%S")

    try:
        lock_fd = acquire_device_lock(device)

        t0 = time.perf_counter()
        print(f"[{_ts()}] setup start", file=sys.stderr)
        workspace = resolve_workspace_bundle(**metadata.model_dump())
        workspace_path = workspace["workspace_path"]
        target_path, _ = _primary_artifact_from_manifest(workspace)
        reference_root = Path(workspace_path) / "inputs" / "reference"
        reference_path = load_reference_entry(reference_root)
        config_rel = f"state/trial.inline.{_slugify(args.config_slug)}.json"
        env = _config_env(workspace, "trial", 1, args.config_slug, config, config_rel)

        reference_module = load_reference_module(reference_path)
        reference_config = extract_config_payload(env["CUDA_EXEC_CONFIG_JSON"])
        print(f"[{_ts()}] setup done ({time.perf_counter() - t0:.1f}s)", file=sys.stderr)

        t1 = time.perf_counter()
        print(f"[{_ts()}] reference start", file=sys.stderr)
        reference_result = measure_reference(
            reference_module,
            reference_config,
            device=device,
            seed=args.seed,
            num_warmups=args.num_warmups,
            num_trials=args.num_perf_trials,
        )
        print(f"[{_ts()}] reference done ({time.perf_counter() - t1:.1f}s)", file=sys.stderr)

        # --- cuDNN vendor baseline (optional) ---
        cudnn_root = Path(workspace_path) / "inputs" / "cudnn"
        cudnn_path = load_cudnn_entry(cudnn_root)
        cudnn_result = None
        if cudnn_path is not None:
            t2 = time.perf_counter()
            print(f"[{_ts()}] cudnn start", file=sys.stderr)
            cudnn_module = load_reference_module(cudnn_path)
            cudnn_result = measure_reference(
                cudnn_module,
                reference_config,
                device=device,
                seed=args.seed,
                num_warmups=args.num_warmups,
                num_trials=args.num_perf_trials,
            )
            print(f"[{_ts()}] cudnn done ({time.perf_counter() - t2:.1f}s)", file=sys.stderr)

        t3 = time.perf_counter()
        print(f"[{_ts()}] generated start", file=sys.stderr)
        generated_run = _run_generated(target_path, env, workspace_path, args.timeout)
        generated_payload = generated_run["payload"]
        print(f"[{_ts()}] generated done ({time.perf_counter() - t3:.1f}s)", file=sys.stderr)

        t4 = time.perf_counter()
        print(f"[{_ts()}] correctness start", file=sys.stderr)
        correctness = _verify_correctness(
            reference_module,
            reference_config,
            generated_payload,
            device=device,
            num_trials=args.num_correctness_trials,
            seed=args.seed,
        )
        print(f"[{_ts()}] correctness done ({time.perf_counter() - t4:.1f}s)", file=sys.stderr)

        comparison = _comparison_payload(
            correctness,
            reference_result["performance"],
            generated_payload,
            cudnn_performance=cudnn_result["performance"] if cudnn_result else None,
        )

        result = {
            "metadata": metadata.model_dump(),
            "config_slug": args.config_slug,
            "status": "ok",
            "reference": {
                "output": {
                    "result": [],  # binary file, never inline large tensors
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

        if cudnn_result is not None:
            result["cudnn"] = {
                "output": {
                    "result": [],  # binary file, never inline large tensors
                    "metadata": reference_config,
                },
                "performance": {
                    **cudnn_result["performance"],
                    "metadata": {
                        **reference_config,
                        **cudnn_result["performance"].get("metadata", {}),
                    },
                },
            }
        print(json.dumps(result, indent=2))
        return 0

    except TimeoutError:
        error_result = {
            "metadata": metadata.model_dump(),
            "config_slug": args.config_slug,
            "status": "timeout",
            "error": "trial watchdog timeout expired",
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
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        try:
            gpu_cleanup(device)
        except Exception:
            pass

        release_device_lock(lock_fd)

        # Clean up stale lock if our process is about to exit abnormally
        device_index = device.index if device.index is not None else 0
        lock_path = Path.home() / ".cuda_exec" / f".lock_cuda_{device_index}"
        cleanup_lockfile(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
