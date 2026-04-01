"""Blackwell GeForce CuTe DSL BF16 GEMM reference fixture for cuda_exec evaluation.

Wraps ``Sm120Gemm`` from ``cute_gemm.py`` into the ``Model(nn.Module)`` contract
expected by the evaluation harness.  All inputs are ``torch.bfloat16``; the
accumulator is ``torch.float32`` and FP32->BF16 conversion is done in the kernel
epilogue (no host-side conversion).  Output buffer is pre-allocated and reused.

Convention for all inputs/outputs:
    A: (M, K)  row-major BF16  — passed directly, zero-copy
    B: (K, N)  row-major BF16  — B.t() view (zero-copy) gives (N, K) N-major
    C: (M, N)  row-major BF16  — FP32->BF16 conversion in kernel epilogue

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

from cute_gemm import Sm120Gemm


# ---------------------------------------------------------------------------
#  Config helpers (unchanged from the original fixture contract)
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
    # For square matrices: shape is [N, N], input_size is N*N per tensor
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
#  Model — nn.Module wrapper around Sm120Gemm (cute_gemm.py)
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """CuTe DSL BF16 GEMM reference kernel for Blackwell (SM120).

    Computes C = A @ B where A is M×K, B is K×N, C is M×N.
    Inputs are BF16 with FP32 accumulation; output is BF16 (converted in-kernel).

    Zero-copy from row-major PyTorch tensors:
        - A passed directly as (M, K) K-major CuTe tensor
        - B passed as B.t() (zero-copy view) giving (N, K) N-major CuTe tensor
        - C is BF16 row-major — FP32->BF16 conversion done in the kernel epilogue

    The kernel is JIT-compiled once per unique (A.shape, B.shape) via
    ``cute.compile()`` and reused for subsequent calls with the same shape.
    Output buffer C is pre-allocated and reused across calls.
    """

    def __init__(self):
        super().__init__()
        self._gemm = Sm120Gemm(output_bf16=True)
        self._compiled = None
        self._stream = None
        self._cached_shape = None
        self._cached_ptrs = None  # (A.data_ptr, B.data_ptr)
        self._a_cute = None
        self._b_cute = None
        self._c_cute = None
        self._C = None

    def _ensure_compiled(self, A: torch.Tensor, B: torch.Tensor) -> None:
        """JIT-compile the kernel on first call or when the matrix shape changes."""
        shape_key = (A.shape, B.shape)
        if self._compiled is not None and self._cached_shape == shape_key:
            return

        import cuda.bindings.driver as cuda_driver
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack

        M, K = A.shape
        _, N = B.shape
        self._C = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)

        self._a_cute = from_dlpack(A, assumed_align=16)
        self._b_cute = from_dlpack(B.t(), assumed_align=16)
        self._c_cute = from_dlpack(self._C, assumed_align=16)

        self._stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
        self._compiled = cute.compile(self._gemm, self._a_cute, self._b_cute,
                                      self._c_cute, stream=self._stream)
        self._cached_shape = shape_key
        self._cached_ptrs = (A.data_ptr(), B.data_ptr())

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
            raise ValueError("Sm120Gemm reference kernel requires CUDA tensors")

        self._ensure_compiled(A, B)

        # Reuse cached CuTe tensors if input pointers haven't changed
        ptr_key = (A.data_ptr(), B.data_ptr())
        if ptr_key != self._cached_ptrs:
            from cutlass.cute.runtime import from_dlpack
            self._a_cute = from_dlpack(A, assumed_align=16)
            self._b_cute = from_dlpack(B.t(), assumed_align=16)
            self._cached_ptrs = ptr_key

        self._compiled(self._a_cute, self._b_cute, self._c_cute, self._stream)

        return self._C


# ---------------------------------------------------------------------------
#  Contract functions
# ---------------------------------------------------------------------------

def get_init_inputs() -> list[Any]:
    return []


def get_inputs(config: dict[str, Any]) -> list[torch.Tensor]:
    cfg = _normalize_config(config)
    shape = tuple(int(v) for v in cfg["shape"])
    device = torch.device("cuda")
    # For square matrices: shape = [N, N], M = N = K = shape[0]
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

    # Warmup — 5 runs to JIT-compile and warm GPU caches
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
