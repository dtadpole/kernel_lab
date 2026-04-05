#!/usr/bin/env python3
"""Profile harness for CuTe DSL reference kernels under NCU.

Wraps a cutedsl.py module with L2 cache flush and controlled
warmup/trial counts.  Used by the Makefile's profile-ncu-reference
target so that NCU profiling gets the same measurement environment
as trial (L2 flush before each run).

Usage:
    python profile_reference.py /path/to/cutedsl.py

Environment variables (same as eval_harness.cu):
    CUDA_EXEC_PARAM_SHAPE, CUDA_EXEC_PARAM_INPUT_SIZE, etc.
    CUDA_EXEC_NUM_WARMUPS  (default 5)
    CUDA_EXEC_NUM_TRIALS   (default 10)
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: profile_reference.py <cutedsl.py>", file=sys.stderr)
        return 2

    ref_path = sys.argv[1]
    num_warmups = int(os.environ.get("CUDA_EXEC_NUM_WARMUPS", "5"))
    num_trials = int(os.environ.get("CUDA_EXEC_NUM_TRIALS", "10"))

    # Load the reference module (add its directory to sys.path for sibling imports)
    ref_dir = os.path.dirname(os.path.abspath(ref_path))
    if ref_dir not in sys.path:
        sys.path.insert(0, ref_dir)
    spec = importlib.util.spec_from_file_location("cutedsl", ref_path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)

    device = torch.device("cuda")
    model = ref.Model(*ref.get_init_inputs())
    model = model.cuda(device=device)

    # Build config from env (same as cutedsl.py's _config_from_env)
    config = ref._config_from_env() if hasattr(ref, "_config_from_env") else {}
    inputs = list(ref.get_inputs(config))
    inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]

    # L2 cache flush buffer (Triton do_bench / NVBench pattern)
    l2_size = torch.cuda.get_device_properties(device).L2_cache_size
    l2_flush = torch.empty(l2_size, dtype=torch.uint8, device=device) if l2_size > 0 else None

    # Warmup
    for _ in range(num_warmups):
        model(*inputs)
    torch.cuda.synchronize(device)

    # Timed trials with L2 flush
    for _ in range(num_trials):
        if l2_flush is not None:
            l2_flush.zero_()
        model(*inputs)
        torch.cuda.synchronize(device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
