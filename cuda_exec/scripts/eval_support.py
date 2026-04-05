"""Shared utilities for cuda_exec trial scripts.

Provides device locking, watchdog timeout, GPU cleanup, reference module
loading/measurement, and correctness helpers.
"""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import signal
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Alignment defaults (matching kbEvalCli.py / kbEvalUtil.py)
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42
NUM_CORRECTNESS_TRIALS = 3
NUM_WARMUP_RUNS = 5
NUM_PERF_TRIALS = 10
ATOL = 1e-02
RTOL = 1e-02


# ---------------------------------------------------------------------------
# Seed control
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# ---------------------------------------------------------------------------
# Device locking (aligned with triton-ag configEndpoints.py / kbEvalCli.py)
# ---------------------------------------------------------------------------
MAX_LOCK_AGE = 30  # seconds — force-delete lock older than this
MAX_LOCK_WAIT = 15  # seconds — max time to wait for lock before giving up


def cleanup_lockfile(lock_path: Path) -> None:
    """Remove stale lock files left by dead or timed-out processes."""
    if not lock_path.exists():
        return
    my_pid = os.getpid()
    lock_mtime = lock_path.stat().st_mtime
    try:
        with open(lock_path, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            # Shared lock succeeded → exclusive lock is released.
            # Check if the owning process is still alive.
            first_line = f.readline().strip()
            digits = "".join(c for c in first_line if c.isdigit())
            if digits:
                file_pid = int(digits)
                if file_pid != my_pid:
                    try:
                        os.kill(file_pid, 0)  # probe — no signal sent
                    except ProcessLookupError:
                        # Process is dead — remove stale lock
                        try:
                            os.remove(lock_path)
                        except OSError:
                            pass
    except BlockingIOError:
        # Cannot get shared lock → someone holds exclusive lock.
        # Safety net: if lock is older than MAX_LOCK_AGE, force-delete.
        if lock_mtime < time.time() - MAX_LOCK_AGE:
            try:
                os.remove(lock_path)
            except OSError:
                pass


def acquire_device_lock(device: torch.device) -> int | None:
    """Acquire GPU device lock with wait + stale cleanup."""
    lock_dir = Path.home() / ".cuda_exec"
    lock_dir.mkdir(parents=True, exist_ok=True)
    device_index = device.index if device.index is not None else 0
    lock_path = lock_dir / f".lock_cuda_{device_index}"

    deadline = time.monotonic() + MAX_LOCK_WAIT
    while True:
        cleanup_lockfile(lock_path)
        try:
            fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
            return fd
        except BlockingIOError:
            os.close(fd)
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"CUDA device {device_index} is locked by another process "
                    f"(waited {MAX_LOCK_WAIT}s). Lock file: {lock_path}"
                )
            time.sleep(0.5)


def release_device_lock(lock_fd: int | None) -> None:
    if lock_fd is not None:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# GPU cleanup
# ---------------------------------------------------------------------------

def gpu_cleanup(device: torch.device) -> None:
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize(device=device)


# ---------------------------------------------------------------------------
# Watchdog (signal-based, kept for backward compat)
# ---------------------------------------------------------------------------

def watchdog_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("watchdog timeout expired")




# ---------------------------------------------------------------------------
# Reference loading
# ---------------------------------------------------------------------------

def load_reference_entry(reference_root: Path) -> Path:
    candidates = sorted(reference_root.rglob("cutedsl.py"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"reference execution requires exactly one cutedsl.py under {reference_root}; found {len(candidates)}"
        )
    return candidates[0]


def load_cudnn_entry(cudnn_root: Path) -> Path | None:
    """Return the cudnn.py entry point if it exists, else None."""
    candidates = sorted(cudnn_root.rglob("cudnn.py"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def load_reference_module(reference_path: Path):
    spec = importlib.util.spec_from_file_location("cuda_exec_reference_module", reference_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load reference module from {reference_path}")
    module = importlib.util.module_from_spec(spec)
    # Add the reference directory to sys.path so sibling imports (e.g. cute_gemm) work
    ref_dir = str(reference_path.parent)
    path_added = ref_dir not in sys.path
    if path_added:
        sys.path.insert(0, ref_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        if path_added and ref_dir in sys.path:
            sys.path.remove(ref_dir)
    return module


def normalize_reference_contract(module: Any) -> tuple[Any, Any]:
    """Validate fixture module contract: Model(nn.Module) + get_init_inputs().

    Fixtures define only the kernel logic (Model) and initialization args.
    Input generation is the harness's responsibility via generate_inputs().
    """
    model_cls = getattr(module, "Model", None)
    get_init_inputs = getattr(module, "get_init_inputs", None)
    if model_cls is None:
        raise RuntimeError(
            "reference module contract requires Model(nn.Module) and get_init_inputs()"
        )
    if not isinstance(model_cls, type) or not issubclass(model_cls, nn.Module):
        raise RuntimeError("reference module Model must be a subclass of torch.nn.Module")
    return model_cls, get_init_inputs


def extract_config_payload(env_json: str) -> dict[str, Any]:
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

def tensor_to_jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [tensor_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): tensor_to_jsonable(v) for k, v in value.items()}
    return value


def flatten_numeric(value: Any) -> list[float]:
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(flatten_numeric(item))
        return out
    if isinstance(value, (int, float)):
        return [float(value)]
    raise RuntimeError(f"reference output contains non-numeric value {value!r}")


def infer_shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, (int, float)):
        return ()
    if isinstance(value, list):
        if not value:
            return (0,)
        return (len(value),) + infer_shape(value[0])
    return ()


def allclose_check(
    ref_values: list[float],
    gen_values: list[float],
    atol: float = ATOL,
    rtol: float = RTOL,
) -> tuple[bool, float, float]:
    if not ref_values and not gen_values:
        return True, 0.0, 0.0
    abs_diffs = [abs(a - b) for a, b in zip(ref_values, gen_values)]
    tolerances = [atol + rtol * abs(b) for b in gen_values]
    passed = all(d <= t for d, t in zip(abs_diffs, tolerances))
    max_diff = max(abs_diffs) if abs_diffs else 0.0
    avg_diff = statistics.fmean(abs_diffs) if abs_diffs else 0.0
    return passed, max_diff, avg_diff


# ---------------------------------------------------------------------------
# Harness input generation — single source of truth
# ---------------------------------------------------------------------------

def generate_inputs(
    config: dict[str, Any],
    device: torch.device,
) -> list[torch.Tensor]:
    """Generate input tensors from config. Called by the harness, not fixtures.

    This is the single source of truth for input generation — all three
    implementations (cuBLAS, CuTe DSL, Generated CUDA) receive inputs
    from this function. Fixtures define only Model, not get_inputs().
    """
    shape = [int(v) for v in config["shape"]]
    family = config.get("family", "")

    if "matrix-multiplication" in family or len(shape) == 2:
        M = shape[0]
        K = shape[1] if len(shape) > 1 else shape[0]
        N = shape[1] if len(shape) > 1 else shape[0]
        A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        B = torch.randn(K, N, dtype=torch.bfloat16, device=device)
        return [A, B]

    if "fa4" in family or "mha" in family or len(shape) == 4:
        batch, seq_len, num_heads, head_dim = shape
        num_kv_heads = int(config.get("num_kv_heads", num_heads))
        causal = bool(config.get("causal", False))
        Q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        K = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
        V = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
        return [Q, K, V, causal]

    if len(shape) == 1:
        # Vector ops (e.g., vecadd)
        A = torch.randn(shape[0], dtype=torch.bfloat16, device=device)
        B = torch.randn(shape[0], dtype=torch.bfloat16, device=device)
        return [A, B]

    raise ValueError(f"Cannot generate inputs for config: shape={shape}, family={family}")


# ---------------------------------------------------------------------------
# Reference measurement — CUDA event timing
# ---------------------------------------------------------------------------

def measure_reference(
    module: Any,
    config: dict[str, Any],
    device: torch.device,
    seed: int = DEFAULT_SEED,
    num_warmups: int = NUM_WARMUP_RUNS,
    num_trials: int = NUM_PERF_TRIALS,
) -> dict[str, Any]:
    model_cls = getattr(module, "Model", None)
    get_init_inputs = getattr(module, "get_init_inputs", None)
    if model_cls is None:
        raise RuntimeError("reference module must export Model(nn.Module)")
    hardware = torch.cuda.get_device_name(device=device)

    with torch.no_grad(), torch.cuda.device(device):
        set_seed(seed)
        init_inputs = list(get_init_inputs()) if get_init_inputs else []
        init_inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x
            for x in init_inputs
        ]
        model = model_cls(*init_inputs)
        model = model.cuda(device=device)

        # Harness generates inputs — fixtures do NOT own input generation
        inputs = generate_inputs(config, device)

        # L2 cache flush buffer (Triton do_bench pattern)
        l2_size = torch.cuda.get_device_properties(device).L2_cache_size
        l2_flush = torch.empty(l2_size, dtype=torch.uint8, device=device) if l2_size > 0 else None

        for _ in range(num_warmups):
            model(*inputs)
        torch.cuda.synchronize(device=device)

        latencies_ms: list[float] = []
        last_output: Any = None
        for trial_idx in range(num_trials):
            if l2_flush is not None:
                l2_flush.zero_()
            # Fresh input buffers — new allocations per trial (new pointers),
            # matching the C eval_harness.cu methodology.
            inputs = generate_inputs(config, device)
            torch.cuda.synchronize(device=device)
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            last_output = model(*inputs)
            end_ev.record()
            end_ev.synchronize()
            latencies_ms.append(start_ev.elapsed_time(end_ev))

        # Run once more with deterministic inputs for correctness output
        set_seed(seed)
        inputs = generate_inputs(config, device)
        last_output = model(*inputs)
        torch.cuda.synchronize(device=device)

    if last_output is None:
        raise RuntimeError("reference contract did not produce an output")

    # Write output as binary file (matching eval_harness.cu pattern).
    # Never serialize large tensors to JSON — 8192×8192 = 67M floats = 2GB JSON.
    output_dir = os.environ.get("CUDA_EXEC_OUTPUT_DIR")
    if output_dir and hasattr(last_output, "cpu"):
        os.makedirs(output_dir, exist_ok=True)
        bin_path = os.path.join(output_dir, "reference_output_0.bin")
        last_output.detach().cpu().contiguous().numpy().tofile(bin_path)

    std_val = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0

    return {
        "output": [],  # binary file, not JSON (same as eval_harness.cu)
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
