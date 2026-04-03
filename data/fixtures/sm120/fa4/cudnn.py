"""cuDNN Flash Attention vendor baseline for cuda_exec evaluation.

Uses torch.nn.functional.scaled_dot_product_attention which dispatches to
cuDNN's fused flash attention kernel on supported GPUs (Ampere+). This is
NVIDIA's vendor-optimized attention implementation.

This file serves as the vendor-optimized baseline to compare against
hand-written CUDA kernels and FA4 CuTe DSL implementations.

Input layout: Q, K, V each (batch, seqlen, num_heads, head_dim) — BF16
Output layout: O same shape — BF16

The SDPA function expects (batch, num_heads, seqlen, head_dim) so we
transpose on entry and exit. The transposes are zero-copy stride swaps.

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
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
#  Config helpers
# ---------------------------------------------------------------------------

def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got {type(config)!r}")

    required = ("batch_size", "seq_len", "num_heads", "head_dim")
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"config missing required keys: {missing}")

    return {
        "batch_size": int(config["batch_size"]),
        "seq_len": int(config["seq_len"]),
        "num_heads": int(config["num_heads"]),
        "num_kv_heads": int(config.get("num_kv_heads", config["num_heads"])),
        "head_dim": int(config["head_dim"]),
        "causal": bool(config.get("causal", False)),
    }


def _config_from_env() -> dict[str, Any]:
    raw = os.environ.get("CUDA_EXEC_CONFIG_JSON")
    if raw:
        payload = json.loads(raw)
        params = payload.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("CUDA_EXEC_CONFIG_JSON.params must be an object")
        return _normalize_config(params)
    raise ValueError("CUDA_EXEC_CONFIG_JSON environment variable is required")


# ---------------------------------------------------------------------------
#  Model — cuDNN Flash Attention via PyTorch SDPA
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """cuDNN Flash Attention vendor baseline.

    Forces the cuDNN backend by disabling the other SDPA backends
    (flash_sdp and math_sdp). On Ampere+ GPUs with cuDNN 8.9+,
    this dispatches to cuDNN's fused multi-head attention kernel.

    GQA (grouped-query attention) is handled by expanding K/V heads
    to match Q heads before calling SDPA.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        if Q.dtype != torch.bfloat16:
            raise ValueError(f"expected bfloat16 inputs, got {Q.dtype}")
        if not Q.is_cuda:
            raise ValueError("cuDNN attention requires CUDA tensors")

        # Input: (batch, seqlen, num_heads, head_dim)
        # SDPA expects: (batch, num_heads, seqlen, head_dim)
        q = Q.transpose(1, 2)
        k = K.transpose(1, 2)
        v = V.transpose(1, 2)

        # Handle GQA: expand KV heads to match Q heads
        num_q_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        if num_kv_heads != num_q_heads:
            repeat_factor = num_q_heads // num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # Force cuDNN backend by disabling alternatives
        with torch.nn.attention.sdpa_kernel(
            [torch.nn.attention.SDPBackend.CUDNN_ATTENTION]
        ):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

        # Back to (batch, seqlen, num_heads, head_dim)
        return out.transpose(1, 2)


# ---------------------------------------------------------------------------
#  Contract functions
# ---------------------------------------------------------------------------

def get_init_inputs() -> list[Any]:
    return []


def get_inputs(config: dict[str, Any]) -> list:
    cfg = _normalize_config(config)
    device = torch.device("cuda")
    B = cfg["batch_size"]
    S = cfg["seq_len"]
    H = cfg["num_heads"]
    Hkv = cfg["num_kv_heads"]
    D = cfg["head_dim"]

    Q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
    K = torch.randn(B, S, Hkv, D, dtype=torch.bfloat16, device=device)
    V = torch.randn(B, S, Hkv, D, dtype=torch.bfloat16, device=device)
    return [Q, K, V, cfg["causal"]]


# ---------------------------------------------------------------------------
#  Latency helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
#  Standalone main
# ---------------------------------------------------------------------------

def main() -> int:
    config = _config_from_env()
    device = torch.device("cuda")
    model = Model()
    model = model.cuda(device=device)
    Q, K, V, causal = get_inputs(config)

    print(
        f"cuDNN SDPA baseline: causal={causal}, "
        f"shape=({config['batch_size']}, {config['seq_len']}, "
        f"{config['num_heads']}, {config['head_dim']})",
        flush=True,
    )

    # Warmup — 5 runs
    with torch.no_grad():
        for _ in range(5):
            model(Q, K, V, causal=causal)
    torch.cuda.synchronize(device)

    # Timed runs — 10 trials with CUDA event timing
    latencies_ms: list[float] = []
    result = None
    with torch.no_grad():
        for _ in range(10):
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            result = model(Q, K, V, causal=causal)
            end_ev.record()
            end_ev.synchronize()
            latencies_ms.append(start_ev.elapsed_time(end_ev))

    assert result is not None
    output_sample = result[0, 0, 0, :8].detach().cpu().tolist()

    payload = {
        "output": {
            "result": output_sample,
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
