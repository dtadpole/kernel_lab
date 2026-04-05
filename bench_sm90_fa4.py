#!/usr/bin/env python3
"""Benchmark FA4 implementations on H100 with cold-L2 flushing.

Methodology matches eval_harness.cu:
- L2 cache flush (cudaMemsetAsync on L2-sized buffer) before each trial
- Fresh random inputs per config (not per trial — matches harness)
- CUDA event timing per trial
- Median latency reported

FA FLOPs formula (forward only):
  non-causal: 4 * B * H * S^2 * D   (QK^T matmul + PV matmul)
  causal:     2 * B * H * S^2 * D   (half elements due to triangular mask)
"""
import argparse
import ctypes
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
#  L2 cache flush (mirrors eval_harness.cu pattern)
# ---------------------------------------------------------------------------

_l2_flush_buf = None

def _init_l2_flush():
    global _l2_flush_buf
    props = torch.cuda.get_device_properties(0)
    l2_size = props.L2_cache_size
    if l2_size > 0:
        _l2_flush_buf = torch.empty(l2_size, dtype=torch.uint8, device="cuda")

def _flush_l2():
    if _l2_flush_buf is not None:
        _l2_flush_buf.zero_()

# ---------------------------------------------------------------------------
#  FLOPs / TFLOPS helpers
# ---------------------------------------------------------------------------

def fa_flops(B, S, H, D, causal):
    if causal:
        return 2 * B * H * S * S * D
    return 4 * B * H * S * S * D

def to_tflops(flops, latency_ms):
    return flops / (latency_ms * 1e-3) / 1e12

def make_inputs(cfg):
    B, S, H, D = cfg["batch_size"], cfg["seq_len"], cfg["num_heads"], cfg["head_dim"]
    Q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    return Q, K, V

# ---------------------------------------------------------------------------
#  Benchmark core — cold-L2 per trial
# ---------------------------------------------------------------------------

def bench_fn(fn, Q, K, V, causal, num_warmup, num_trials):
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(num_warmup):
            fn(Q, K, V, causal)
    torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(num_trials):
            _flush_l2()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn(Q, K, V, causal)
            e.record()
            e.synchronize()
            latencies.append(s.elapsed_time(e))
    latencies.sort()
    return latencies[len(latencies) // 2]

# ---------------------------------------------------------------------------
#  Implementation loaders
# ---------------------------------------------------------------------------

def load_fa4():
    try:
        from flash_attn.cute import flash_attn_func as fa4_cute_func
        _pq = torch.randn(1, 32, 1, 64, dtype=torch.bfloat16, device="cuda")
        _pk = torch.randn(1, 32, 1, 64, dtype=torch.bfloat16, device="cuda")
        _pv = torch.randn(1, 32, 1, 64, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            fa4_cute_func(_pq, _pk, _pv, causal=False)
        del _pq, _pk, _pv

        def fa4_fn(Q, K, V, causal):
            out = fa4_cute_func(Q, K, V, causal=causal)
            return out[0] if isinstance(out, tuple) else out
        return fa4_fn
    except Exception as e:
        print(f"FA4 CuTe DSL not available: {e}")
        return None

def load_cudnn():
    def cudnn_fn(Q, K, V, causal):
        q = Q.transpose(1, 2)
        k = K.transpose(1, 2)
        v = V.transpose(1, 2)
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.CUDNN_ATTENTION]):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return out.transpose(1, 2)
    return cudnn_fn

def load_generated():
    gen_so = Path("/tmp/fa4_sm90_bench.so")
    gen_cu = REPO / "data/generated/sm90/fa4/generated.cu"
    if not gen_so.exists() or gen_so.stat().st_mtime < gen_cu.stat().st_mtime:
        print("Compiling generated.cu ...", flush=True)
        subprocess.run([
            "/usr/local/cuda-12.9/bin/nvcc",
            "-gencode", "arch=compute_90a,code=sm_90a",
            "-O2", "-Xcompiler", "-fPIC",
            "--shared", "-o", str(gen_so), str(gen_cu),
            "-I/usr/local/cuda-12.9/include", "-lcuda",
        ], check=True)
        print("Compiled.", flush=True)

    lib = ctypes.CDLL(str(gen_so))
    lib.kernel_run.restype = ctypes.c_int
    lib.kernel_run.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
        ctypes.c_int, ctypes.c_void_p,
    ]

    def gen_fn(Q, K, V, causal):
        B, S, H, D = Q.shape
        O = torch.zeros_like(Q)
        n = B * S * H * D
        inputs = (ctypes.c_void_p * 3)(Q.data_ptr(), K.data_ptr(), V.data_ptr())
        outputs = (ctypes.c_void_p * 1)(O.data_ptr())
        stream = torch.cuda.current_stream().cuda_stream
        os.environ["CUDA_EXEC_PARAM_BATCH_SIZE"] = str(B)
        os.environ["CUDA_EXEC_PARAM_SEQ_LEN"] = str(S)
        os.environ["CUDA_EXEC_PARAM_NUM_HEADS"] = str(H)
        os.environ["CUDA_EXEC_PARAM_HEAD_DIM"] = str(D)
        os.environ["CUDA_EXEC_PARAM_CAUSAL"] = "true" if causal else "false"
        lib.kernel_run(inputs, 3, outputs, 1, n, ctypes.c_void_p(stream))
        return O
    return gen_fn

# ---------------------------------------------------------------------------
#  Version detection
# ---------------------------------------------------------------------------

def get_versions():
    cudnn_v = torch.backends.cudnn.version()
    cudnn_str = f"{cudnn_v // 10000}.{(cudnn_v % 10000) // 100}.{cudnn_v % 100}"
    try:
        fa4_ver = importlib.metadata.version("flash-attn-4")
    except Exception:
        fa4_ver = "unknown"
    return {
        "torch": torch.__version__,
        "cudnn": cudnn_str,
        "fa4": fa4_ver,
        "gpu": torch.cuda.get_device_name(0),
    }

# ---------------------------------------------------------------------------
#  Comparison table output (box-drawing format from SKILL.md)
# ---------------------------------------------------------------------------

def print_comparison_table(results, versions, peak_tflops, hostname):
    gpu_name = versions["gpu"]
    cudnn_ver = versions["cudnn"]
    fa4_ver = f"v{versions['fa4']}"
    torch_ver = versions["torch"]

    C0 = 24  # first column width (content, excl borders)
    C1 = 18  # data column width
    C2 = 18
    C3 = 18
    C4 = 10  # ratio column width
    C5 = 10

    row1_c0 = f" {gpu_name} ({hostname})"
    # Truncate torch version to fit column: "2.11.0+cu128" -> "2.11+cu128"
    short_torch = torch_ver.split("+")
    tv = short_torch[0].rsplit(".", 1)[0] if len(short_torch[0].split(".")) > 2 else short_torch[0]
    tv = tv + "+" + short_torch[1] if len(short_torch) > 1 else tv
    row2_c0 = f" GPU4, torch {tv}"

    def hr(left, mid, right, fill="─"):
        return f"{left}{fill*C0}{mid}{fill*C1}{mid}{fill*C2}{mid}{fill*C3}{mid}{fill*C4}{mid}{fill*C5}{right}"

    def row(c0, c1, c2, c3, c4, c5):
        return f"│{c0:<{C0}}│{c1:^{C1}}│{c2:^{C2}}│{c3:^{C3}}│{c4:^{C4}}│{c5:^{C5}}│"

    # Header
    print(hr("┌", "┬", "┐"))
    print(row(row1_c0, f"cuDNN {cudnn_ver}", "FA4 CuTe DSL", "Generated CUDA", "FA4 DSL", "Gen CUDA"))
    print(row(row2_c0, "TFLOPS   (ms)", f"{fa4_ver}  (ms)", "TFLOPS   (ms)", "vs cuDNN", "vs cuDNN"))
    print(hr("├", "┼", "┤"))

    best_cudnn = 0
    best_fa4 = 0
    best_gen = 0

    for r in results:
        slug = r["slug"]
        cudnn_tf, cudnn_ms = r.get("cudnn_tf", 0), r.get("cudnn_ms", 0)
        fa4_tf, fa4_ms = r.get("fa4_tf", 0), r.get("fa4_ms", 0)
        gen_tf, gen_ms = r.get("gen_tf", 0), r.get("gen_ms", 0)

        best_cudnn = max(best_cudnn, cudnn_tf)
        best_fa4 = max(best_fa4, fa4_tf)
        best_gen = max(best_gen, gen_tf)

        fa4_ratio = f"{fa4_tf / cudnn_tf:.2f}×" if cudnn_tf > 0 else "N/A"
        gen_ratio = f"{gen_tf / cudnn_tf:.2f}×" if cudnn_tf > 0 else "N/A"

        def data_col(tf, ms):
            if tf:
                return f"{tf:6.1f}  ({ms:.3f})"
            return "N/A"

        print(row(f" {slug}", data_col(cudnn_tf, cudnn_ms), data_col(fa4_tf, fa4_ms),
                  data_col(gen_tf, gen_ms), fa4_ratio, gen_ratio))
        print(hr("├", "┼", "┤"))

    # Footer — % of peak
    cudnn_pct = f"{best_cudnn / peak_tflops * 100:.1f}%"
    fa4_pct = f"{best_fa4 / peak_tflops * 100:.1f}%"
    gen_pct = f"{best_gen / peak_tflops * 100:.1f}%"
    peak_str = f"{peak_tflops:.1f}TF"

    print(row(" % of peak (best cfg)", cudnn_pct, fa4_pct, gen_pct, "", peak_str))
    print(hr("└", "┴", "┘"))

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FA4 3-way benchmark on SM90 H100")
    parser.add_argument("--gpu", type=int, default=4, help="GPU index (default: 4)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--trials", type=int, default=20, help="Timed trials")
    parser.add_argument("--peak-tflops", type=float, default=800.0, help="GPU peak BF16 TFLOPS for %% of peak")
    parser.add_argument("--hostname", default=None, help="Host short name for table header")
    parser.add_argument("--output-format", choices=["raw", "comparison-table"], default="comparison-table")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Force CUDA init before anything else
    torch.cuda.init()
    _init_l2_flush()

    configs = json.loads((REPO / "data/fixtures/sm90/fa4/configs.json").read_text())
    versions = get_versions()

    hostname = args.hostname
    if hostname is None:
        import socket
        fqdn = socket.gethostname()
        hostname = "h8_3" if "8490" in fqdn else "h8_4" if "8491" in fqdn else fqdn.split(".")[0]

    # Load implementations
    fa4_fn = load_fa4()
    cudnn_fn = load_cudnn()
    gen_fn = load_generated()

    print(f"GPU {args.gpu} — {versions['gpu']}")
    print(f"torch {versions['torch']}, cuDNN {versions['cudnn']}, FA4 {versions['fa4']}")
    print(f"Warmup: {args.warmup}, Trials: {args.trials}, Cold-L2: {'yes' if _l2_flush_buf is not None else 'no'}")
    print()

    results = []
    for slug, cfg in configs.items():
        B, S, H, D = cfg["batch_size"], cfg["seq_len"], cfg["num_heads"], cfg["head_dim"]
        causal = cfg["causal"]
        flops = fa_flops(B, S, H, D, causal)
        Q, K, V = make_inputs(cfg)

        r = {"slug": slug}

        # cuDNN
        try:
            ms = bench_fn(cudnn_fn, Q, K, V, causal, args.warmup, args.trials)
            r["cudnn_ms"], r["cudnn_tf"] = ms, to_tflops(flops, ms)
        except Exception:
            r["cudnn_ms"], r["cudnn_tf"] = 0, 0

        # FA4 CuTe DSL
        if fa4_fn:
            ms = bench_fn(fa4_fn, Q, K, V, causal, args.warmup, args.trials)
            r["fa4_ms"], r["fa4_tf"] = ms, to_tflops(flops, ms)
        else:
            r["fa4_ms"], r["fa4_tf"] = 0, 0

        # Generated CUDA
        ms = bench_fn(gen_fn, Q, K, V, causal, args.warmup, args.trials)
        r["gen_ms"], r["gen_tf"] = ms, to_tflops(flops, ms)

        results.append(r)
        print(f"  {slug}: cuDNN={r['cudnn_tf']:.1f} FA4={r['fa4_tf']:.1f} Gen={r['gen_tf']:.1f} TFLOPS", flush=True)

    print()
    if args.output_format == "comparison-table":
        print_comparison_table(results, versions, args.peak_tflops, hostname)
    else:
        print(json.dumps({"versions": versions, "results": results}, indent=2))

if __name__ == "__main__":
    main()
