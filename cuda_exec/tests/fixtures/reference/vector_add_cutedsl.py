"""CuTeDSL-style reference fixture exposed through a Kernel Bench / kbEval-like module contract.

Contract for cuda_exec reference Python files:
- export `class Model(torch.nn.Module)`
- export `get_inputs(config)`
- export `get_init_inputs()`

This fixture keeps the interface module-based while restoring a real CuTe DSL
vector-add kernel definition for the reference side.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

import cutlass.cute as cute  # type: ignore
from cutlass.cute.arch import block_idx, thread_idx  # type: ignore
import torch
from torch import nn


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got {type(config)!r}")

    missing = [key for key in ("shape", "rank", "shape_kind", "input_size") if key not in config]
    if missing:
        raise ValueError(f"config missing required keys: {missing}")

    shape = config["shape"]
    if not isinstance(shape, list) or not shape:
        raise ValueError("config['shape'] must be a non-empty list")

    normalized_shape = [int(v) for v in shape]
    input_size = int(config["input_size"])
    shape_size = 1
    for dim in normalized_shape:
        shape_size *= dim
    if shape_size != input_size:
        raise ValueError(
            f"config shape product {shape_size} does not match input_size {input_size}"
        )

    return {
        **config,
        "shape": normalized_shape,
        "rank": int(config["rank"]),
        "shape_kind": str(config["shape_kind"]),
        "input_size": input_size,
    }


def _config_from_env() -> dict[str, Any]:
    raw = os.environ.get("CUDA_EXEC_CONFIG_JSON")
    if raw:
        payload = json.loads(raw)
        params = payload.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("CUDA_EXEC_CONFIG_JSON.params must be an object")
        return _normalize_config(params)

    shape = json.loads(os.environ["CUDA_EXEC_PARAM_SHAPE"])
    return _normalize_config(
        {
            "shape": shape,
            "input_size": int(os.environ["CUDA_EXEC_PARAM_INPUT_SIZE"]),
            "rank": int(os.environ["CUDA_EXEC_PARAM_RANK"]),
            "shape_kind": os.environ["CUDA_EXEC_PARAM_SHAPE_KIND"],
        }
    )


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.elements_per_thread = 4
        self.threads = 256

        elements_per_thread = self.elements_per_thread
        threads = self.threads

        @cute.kernel
        def vector_add_kernel(x_ptr, y_ptr, out_ptr, n):
            block_id_x, _, _ = block_idx()
            thread_id_x, _, _ = thread_idx()
            base = (block_id_x * threads + thread_id_x) * elements_per_thread

            for lane in range(elements_per_thread):
                idx = base + lane
                if idx < n:
                    out_ptr[idx] = x_ptr[idx] + y_ptr[idx]

        self.kernel = vector_add_kernel

        @cute.jit
        def launch_vector_add(x_ptr, y_ptr, out_ptr, n, blocks, threads_per_block):
            self.kernel(x_ptr, y_ptr, out_ptr, n).launch(
                grid=[blocks, 1, 1],
                block=[threads_per_block, 1, 1],
            )

        self.launch_vector_add = launch_vector_add

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape:
            raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
        if x.dtype != torch.float32 or y.dtype != torch.float32:
            raise ValueError(f"expected float32 inputs, got {x.dtype} and {y.dtype}")
        if not x.is_cuda or not y.is_cuda:
            raise ValueError("CuTe DSL reference kernel requires CUDA tensors")
        if not x.is_contiguous() or not y.is_contiguous():
            raise ValueError("CuTe DSL reference kernel requires contiguous tensors")

        out = torch.empty_like(x)
        n = x.numel()
        blocks = math.ceil(n / (self.threads * self.elements_per_thread))
        self.launch_vector_add(x, y, out, n, blocks, self.threads)
        torch.cuda.synchronize(x.device)
        return out


def get_init_inputs() -> list[Any]:
    return []


def get_inputs(config: dict[str, Any]) -> list[torch.Tensor]:
    cfg = _normalize_config(config)
    shape = tuple(int(v) for v in cfg["shape"])
    device = torch.device("cuda")
    numel = int(cfg["input_size"])
    x = torch.arange(numel, dtype=torch.float32, device=device).reshape(shape).contiguous()
    y = torch.arange(numel, dtype=torch.float32, device=device).reshape(shape).contiguous()
    return [x, y]


def _latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    ordered = sorted(latencies_ms)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 == 1 else (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "min": ordered[0],
        "median": median,
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def main() -> int:
    config = _config_from_env()
    model = Model(*get_init_inputs())
    latencies_ms: list[float] = []
    result = None
    for _ in range(5):
        x, y = get_inputs(config)
        started = time.perf_counter()
        result = model(x, y)
        torch.cuda.synchronize(x.device)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)

    assert result is not None
    payload = {
        "output": {
            "result": result.detach().cpu().tolist(),
            "metadata": config,
        },
        "correctness": {
            "metadata": config,
            "passed": True,
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
        },
        "performance": {
            "metadata": config,
            "latency_ms": _latency_summary(latencies_ms),
            "runs": len(latencies_ms),
        },
        "summary": {
            "metadata": config,
            "latency_ms": _latency_summary(latencies_ms),
            "runs": len(latencies_ms),
        },
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
