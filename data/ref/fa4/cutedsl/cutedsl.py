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

from flash_attn.cute import flash_attn_func as _flash_attn_func


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

    Calls FA4 CuTe DSL directly. No fallback — fails hard if FA4 is not
    available so we don't silently measure the wrong kernel.

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

        result = _flash_attn_func(Q, K, V, causal=causal)
        return result[0] if isinstance(result, tuple) else result


# ---------------------------------------------------------------------------
#  Contract functions
# ---------------------------------------------------------------------------

def get_init_inputs() -> list[Any]:
    return []
