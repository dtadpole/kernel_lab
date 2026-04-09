"""PyTorch BF16 GEMM reference for cuda_exec evaluation.

Uses torch.mm() which dispatches to cuBLAS internally.
This is the Python-level reference — for a pure C++ cuBLAS reference,
see ref-cublas (cublas.cu).

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
    """PyTorch BF16 GEMM reference (dispatches to cuBLAS-Lt internally)."""

    def __init__(self):
        super().__init__()
        # Use cublasLt backend for better algorithm selection
        torch.backends.cuda.preferred_blas_library('cublaslt')

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


def get_inputs(config: dict[str, Any] | None = None) -> list[torch.Tensor]:
    """Generate input tensors for a given config.

    Returns [A, B] where A is (M, K) and B is (K, N), both BF16 on CUDA.
    """
    if config is None:
        config = _config_from_env()
    else:
        config = _normalize_config(config)

    shape = config["shape"]
    M, N = shape[0], shape[1]
    K = M  # square matmul

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    return [A, B]


def get_init_inputs() -> list[Any]:
    return []
