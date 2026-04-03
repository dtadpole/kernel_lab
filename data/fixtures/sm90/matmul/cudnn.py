"""cuBLAS BF16 GEMM vendor baseline for cuda_exec evaluation.

Uses torch.mm() which dispatches directly to cuBLAS cublasGemmEx on CUDA.
cuBLAS is NVIDIA's most optimized GEMM implementation — cuDNN does not add
value for pure matrix multiplication (cuDNN internally calls cuBLAS for GEMM).

This file serves as the vendor-optimized baseline to compare against
hand-written CUDA kernels and CuTe DSL implementations.

Input layout: A (M, K), B (K, N) — BF16
Output layout: C (M, N) — BF16 (FP32 accumulation is automatic in cuBLAS)

Contract for cuda_exec reference Python files:
- export ``class Model(torch.nn.Module)``
- export ``get_inputs(config)``
- export ``get_init_inputs()``
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch
from torch import nn


# ---------------------------------------------------------------------------
#  Config helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
#  Model — cuBLAS GEMM via torch.mm
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """cuBLAS BF16 GEMM vendor baseline.

    Computes C = A @ B where A is M×K, B is K×N, C is M×N.
    torch.mm dispatches to cublasGemmEx with BF16 inputs and FP32
    internal accumulation. cuBLAS automatically selects the best
    Tensor Core kernel for the current GPU architecture.
    """

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError(f"expected 2-D tensors, got {A.ndim}-D and {B.ndim}-D")
        if A.shape[1] != B.shape[0]:
            raise ValueError(
                f"inner dimensions mismatch: A is {tuple(A.shape)}, B is {tuple(B.shape)}"
            )
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError(f"expected bfloat16 inputs, got {A.dtype} and {B.dtype}")
        if not A.is_cuda or not B.is_cuda:
            raise ValueError("cuBLAS GEMM requires CUDA tensors")

        return torch.mm(A, B)


# ---------------------------------------------------------------------------
#  Contract functions
# ---------------------------------------------------------------------------

def get_init_inputs() -> list[Any]:
    return []


def get_inputs(config: dict[str, Any]) -> list[torch.Tensor]:
    cfg = _normalize_config(config)
    shape = tuple(int(v) for v in cfg["shape"])
    device = torch.device("cuda")
    M = shape[0]
    K = shape[1] if len(shape) > 1 else shape[0]
    N = shape[1] if len(shape) > 1 else shape[0]
    A = torch.arange(M * K, dtype=torch.bfloat16, device=device).reshape(M, K).contiguous()
    B = torch.arange(K * N, dtype=torch.bfloat16, device=device).reshape(K, N).contiguous()
    return [A, B]


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
    device = torch.device("cuda")
    model = Model(*get_init_inputs())
    model = model.cuda(device=device)
    A, B = get_inputs(config)

    # Warmup — 5 runs
    for _ in range(5):
        model(A, B)
    torch.cuda.synchronize(device)

    # Timed runs — 10 trials with CUDA event timing
    latencies_ms: list[float] = []
    result = None
    for _ in range(10):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        result = model(A, B)
        end_ev.record()
        end_ev.synchronize()
        latencies_ms.append(start_ev.elapsed_time(end_ev))

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
