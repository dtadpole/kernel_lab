"""Sample CuTeDSL-style vector add source for integration tests.

This fixture is config-aware. It treats the runtime config as shape metadata and
builds a flattened vector-add view over 1D / 2D / 3D inputs.
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    import cutlass.cute as cute  # type: ignore
except Exception:  # pragma: no cover - fixture may be read/copied without CuTe installed
    cute = None


def _config_from_env() -> dict[str, Any]:
    shape_raw = os.environ.get("CUDA_EXEC_EXTRA_SHAPE", "[1048576]")
    try:
        shape = json.loads(shape_raw)
    except json.JSONDecodeError:
        shape = [1048576]
    if not isinstance(shape, list) or not shape:
        shape = [1048576]

    input_size = int(os.environ.get("CUDA_EXEC_EXTRA_INPUT_SIZE", str(_product(shape))))
    rank = int(os.environ.get("CUDA_EXEC_EXTRA_RANK", str(len(shape))))
    shape_kind = os.environ.get("CUDA_EXEC_EXTRA_SHAPE_KIND", f"{rank}d")

    return {
        "shape": shape,
        "rank": rank,
        "shape_kind": shape_kind,
        "input_size": input_size,
    }


def _product(values: list[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def build_vector_add_from_config(config: dict[str, Any], elements_per_thread: int = 4):
    shape = [int(v) for v in config.get("shape", [config.get("input_size", 1 << 20)])]
    rank = int(config.get("rank", len(shape)))
    length = int(config.get("input_size", _product(shape)))
    shape_kind = str(config.get("shape_kind", f"{rank}d"))

    threads = 256
    blocks = (length + threads * elements_per_thread - 1) // (threads * elements_per_thread)

    if cute is not None:

        @cute.kernel
        def vector_add_kernel(x_ptr, y_ptr, out_ptr, n):
            block_idx = cute.block_idx.x
            thread_idx = cute.thread_idx.x
            base = (block_idx * threads + thread_idx) * elements_per_thread

            for lane in range(elements_per_thread):
                idx = base + lane
                if idx < n:
                    out_ptr[idx] = x_ptr[idx] + y_ptr[idx]

        kernel = vector_add_kernel
    else:
        kernel = None

    return {
        "kernel": kernel,
        "length": length,
        "threads": threads,
        "blocks": blocks,
        "elements_per_thread": elements_per_thread,
        "rank": rank,
        "shape": shape,
        "shape_kind": shape_kind,
    }


def build_vector_add(length: int = 1 << 20, elements_per_thread: int = 4):
    return build_vector_add_from_config(
        {
            "shape": [length],
            "rank": 1,
            "shape_kind": "1d",
            "input_size": length,
        },
        elements_per_thread=elements_per_thread,
    )


if __name__ == "__main__":
    spec = build_vector_add_from_config(_config_from_env())
    print(spec)
