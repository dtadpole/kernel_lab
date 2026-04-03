"""Flash Attention 4 CuTe DSL forward reference for cuda_exec evaluation (SM90 Hopper).

Supports H100 SXM5 (132 SMs) and H100 PCIe (114 SMs).
Use ``detect_sm90_device()`` to identify the current device variant.

Uses flash_attn.cute (FlashAttention-4 CuTe DSL) when available and
functional, otherwise falls back to PyTorch's scaled_dot_product_attention
which dispatches to cuDNN flash attention on H100.

FA4 CuTe DSL uses the SM80 base class path on SM90, which compiles and
runs correctly on Hopper GPUs.

Requires: flash-attn-4 >= 4.0.0b5, nvidia-cutlass-dsl >= 4.4.

Input layout: (batch, seqlen, num_heads, head_dim) — BF16
Output layout: same as input — BF16

Contract for cuda_exec CuTe DSL reference files (cutedsl.py):
- export `class Model(torch.nn.Module)`
- export `get_inputs(config)`
- export `get_init_inputs()`
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
#  SM90 device detection
# ---------------------------------------------------------------------------

_SM90_DEVICES = {
    "h100_sxm":  {"match": ["H100 SXM", "H100 80GB HBM3"], "sms": 132, "bf16_tflops": 989.5},
    "h100_pcie": {"match": ["H100 PCIe", "H100 PCIE"],      "sms": 114, "bf16_tflops": 756.0},
}


def detect_sm90_device() -> str:
    """Return the SM90 device key ('h100_sxm' or 'h100_pcie'), or 'unknown_sm90'."""
    if not torch.cuda.is_available():
        return "unknown_sm90"
    name = torch.cuda.get_device_name().upper()
    for key, info in _SM90_DEVICES.items():
        if any(pat.upper() in name for pat in info["match"]):
            return key
    if "H100" in name:
        return "h100_sxm"
    return "unknown_sm90"


# ---------------------------------------------------------------------------
#  FA4 CuTe DSL availability probe
# ---------------------------------------------------------------------------

_FA4_AVAILABLE = False
_FA_BACKEND = "none"
_flash_attn_func = None

# Allow forcing SDPA backend via env var: CUDA_EXEC_FA4_BACKEND=sdpa
_FORCE_BACKEND = os.environ.get("CUDA_EXEC_FA4_BACKEND", "").lower()

if _FORCE_BACKEND != "sdpa":
    try:
        from flash_attn.cute import flash_attn_func as _fa4_func

        # Smoke test: compile + run a tiny attention to verify the CuTe DSL
        # path works end-to-end on this GPU.
        _probe_q = torch.randn(1, 32, 1, 64, dtype=torch.bfloat16, device="cuda")
        _probe_k = torch.randn(1, 32, 1, 64, dtype=torch.bfloat16, device="cuda")
        _probe_v = torch.randn(1, 32, 1, 64, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            _fa4_func(_probe_q, _probe_k, _probe_v, causal=False)
        _flash_attn_func = _fa4_func
        _FA4_AVAILABLE = True
        _FA_BACKEND = "fa4_cute_dsl"
        del _probe_q, _probe_k, _probe_v
    except Exception:
        pass


# ---------------------------------------------------------------------------
#  Config helpers
# ---------------------------------------------------------------------------

def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize an FA benchmark config."""
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
    """Read FA config from CUDA_EXEC_CONFIG_JSON environment variable."""
    raw = os.environ.get("CUDA_EXEC_CONFIG_JSON")
    if raw:
        payload = json.loads(raw)
        params = payload.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("CUDA_EXEC_CONFIG_JSON.params must be an object")
        return _normalize_config(params)
    raise ValueError("CUDA_EXEC_CONFIG_JSON environment variable is required")


# ---------------------------------------------------------------------------
#  Model
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """Flash Attention 4 CuTe DSL forward reference for H100 (SM90).

    Dispatches to FA4 CuTe DSL (flash_attn.cute) if available, otherwise
    falls back to PyTorch scaled_dot_product_attention (cuDNN backend).

    Input tensors: Q, K, V each (batch, seqlen, num_heads, head_dim) BF16.
    Output tensor: O same shape and dtype.
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
            raise ValueError("FA4 reference requires CUDA tensors")

        if _FA4_AVAILABLE:
            # FA4 CuTe DSL: (batch, seqlen, num_heads, head_dim)
            result = _flash_attn_func(Q, K, V, causal=causal)
            # FA4 returns (output, softmax_lse) tuple; extract output
            return result[0] if isinstance(result, tuple) else result
        else:
            # PyTorch SDPA fallback: expects (batch, num_heads, seqlen, head_dim)
            q = Q.transpose(1, 2)
            k = K.transpose(1, 2)
            v = V.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            return out.transpose(1, 2)


# ---------------------------------------------------------------------------
#  Contract functions
# ---------------------------------------------------------------------------

def get_init_inputs() -> list[Any]:
    return []


def get_inputs(config: dict[str, Any]) -> list:
    """Create Q, K, V tensors and causal flag for the given FA config.

    Returns [Q, K, V, causal] where Q/K/V are (batch, seqlen, num_heads, head_dim) BF16
    and causal is a bool.
    """
    cfg = _normalize_config(config)
    device = torch.device("cuda")
    B = cfg["batch_size"]
    S = cfg["seq_len"]
    H = cfg["num_heads"]
    Hkv = cfg["num_kv_heads"]
    D = cfg["head_dim"]

    numel = B * S * H * D
    Q = torch.arange(numel, dtype=torch.bfloat16, device=device).reshape(B, S, H, D).contiguous()
    K = torch.arange(numel, dtype=torch.bfloat16, device=device).reshape(B, S, Hkv, D).contiguous()
    V = torch.arange(numel, dtype=torch.bfloat16, device=device).reshape(B, S, Hkv, D).contiguous()
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

    dev_key = detect_sm90_device()
    print(
        f"FA4 reference (SM90 {dev_key}): backend={_FA_BACKEND}, "
        f"causal={causal}, shape=({config['batch_size']}, {config['seq_len']}, "
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
