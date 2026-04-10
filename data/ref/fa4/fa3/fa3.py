"""Flash Attention 3 (Meta internal, SM90 Hopper) reference for cuda_exec.

Calls Meta's FA3 kernels from fbcode/fa3/hopper/ via the compiled
fa3_kernels.so. Unlike the pip flash-attn 2.x (which runs SM80 FA2
kernels), this uses the real SM90 Hopper kernels with TMA + WGMMA +
warp specialization.

Requires: Run `python -m fa3.build` once to compile the .so.

Input layout: (batch, seqlen, num_heads, head_dim) -- BF16
Output layout: same as input -- BF16

Contract for cuda_exec reference files:
- export `class Model(torch.nn.Module)`
- export `get_init_inputs()`
"""

from __future__ import annotations

import sys
import os
from typing import Any

import torch
from torch import nn

# Add kernel_lab root to sys.path so we can import fa3.wrapper
_KERNEL_LAB_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _KERNEL_LAB_ROOT not in sys.path:
    sys.path.insert(0, _KERNEL_LAB_ROOT)

from fa3.wrapper import flash_attn_func  # noqa: E402


class Model(nn.Module):
    """FA3 SM90 Hopper reference — calls Meta's internal FA3 kernels.

    Uses WGMMA + TMA + warp specialization on H100 (SM90).
    Supports MHA and GQA (num_kv_heads < num_heads).
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
            raise ValueError("FA3 reference requires CUDA tensors")

        out, _lse = flash_attn_func(Q, K, V, causal=causal)
        return out


def get_init_inputs() -> list[Any]:
    return []
