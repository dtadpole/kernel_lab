"""Hopper (SM90) CuTe DSL BF16 GEMM reference — cuda_exec evaluation contract.

Wraps HopperWgmmaGemmKernel from cute_gemm_sm90.py.
WGMMA 128x256 tile, 2 warpgroups cooperative, TMA G2S + TMA S2G.
"""
from __future__ import annotations
import json, os
from typing import Any
import torch
from torch import nn
from cute_gemm_sm90 import HopperWgmmaGemmKernel

def _normalize_config(config):
    shape = [int(v) for v in config["shape"]]
    return {**config, "shape": shape, "rank": int(config["rank"]),
            "shape_kind": str(config["shape_kind"]), "input_size": int(config["input_size"])}

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        import cutlass
        self._gemm = HopperWgmmaGemmKernel(acc_dtype=cutlass.Float32, tile_shape_mn=(128, 256), cluster_shape_mn=(1, 1))
        self._compiled = self._stream = self._cached_shape = None
        self._C = None

    def _ensure_compiled(self, A, B):
        shape_key = (A.shape, B.shape)
        if self._compiled is not None and self._cached_shape == shape_key:
            return
        import cutlass.cute as cute
        import cuda.bindings.driver as cuda_driver
        from cutlass.cute.runtime import from_dlpack
        M, K = A.shape; _, N = B.shape
        self._C = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)
        # B.t() is a zero-cost view — transpose happens in SMEM/registers via TMA layout
        a_cute = from_dlpack(A.unsqueeze(-1), assumed_align=16)
        b_cute = from_dlpack(B.t().unsqueeze(-1), assumed_align=16)
        c_cute = from_dlpack(self._C.unsqueeze(-1), assumed_align=16)
        self._stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
        self._compiled = cute.compile(self._gemm, a_cute, b_cute, c_cute, self._stream)
        self._cached_shape = shape_key

    def forward(self, A, B):
        self._ensure_compiled(A, B)
        from cutlass.cute.runtime import from_dlpack
        a_cute = from_dlpack(A.unsqueeze(-1), assumed_align=16)
        b_cute = from_dlpack(B.t().unsqueeze(-1), assumed_align=16)
        c_cute = from_dlpack(self._C.unsqueeze(-1), assumed_align=16)
        self._compiled(a_cute, b_cute, c_cute, self._stream)
        return self._C

def get_init_inputs(): return []
