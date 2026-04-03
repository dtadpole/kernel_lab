#!/usr/bin/env python3
"""Benchmark cuBLAS vs CuTe DSL (WGMMA) vs Generated (CUTLASS C++) on SM90 H100."""
import ctypes, json, os, sys, time
os.environ.setdefault("CUTE_DSL_ARCH", "sm_90a")

import torch
torch.cuda.init()

# Monkey-patch cuda.bindings.runtime for driver compat
import cuda.bindings.runtime as crt
import cuda.bindings.driver as cdr
_orig_gdc = crt.cudaGetDeviceCount
def _p_gdc():
    e, c = cdr.cuDeviceGetCount()
    return (crt.cudaError_t.cudaSuccess, c) if e == cdr.CUresult.CUDA_SUCCESS else _orig_gdc()
crt.cudaGetDeviceCount = _p_gdc
_orig_sd = getattr(crt, "cudaSetDevice", None)
def _p_sd(d):
    e, ctx = cdr.cuDevicePrimaryCtxRetain(cdr.CUdevice(d))
    if e == cdr.CUresult.CUDA_SUCCESS: cdr.cuCtxSetCurrent(ctx); return (crt.cudaError_t.cudaSuccess,)
    return _orig_sd(d) if _orig_sd else (crt.cudaError_t.cudaSuccess,)
crt.cudaSetDevice = _p_sd

import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cute_gemm_sm90 import HopperWgmmaGemmKernel

SIZES = [256, 512, 1024, 2048, 4096, 8192]
W, T = 10, 20
DEV = torch.device("cuda")

def bench(fn, N):
    A = torch.randn(N, N, dtype=torch.bfloat16, device=DEV)
    B = torch.randn(N, N, dtype=torch.bfloat16, device=DEV)
    for _ in range(W): fn(A, B)
    torch.cuda.synchronize()
    ts = []
    for _ in range(T):
        A.normal_(); B.normal_(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(A, B); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]

def tf(N, ms): return 2*N*N*N / (ms*1e-3) / 1e12

# --- CuTe DSL ---
class DSL:
    def __init__(self):
        self.gemm = HopperWgmmaGemmKernel(acc_dtype=cutlass.Float32, tile_shape_mn=(128,256), cluster_shape_mn=(1,1))
        self._c = self._compiled = self._shape = self._ptrs = None
    def __call__(self, A, B):
        sk = (A.shape, B.shape)
        if self._compiled is None or self._shape != sk:
            M, K = A.shape; _, N_ = B.shape
            self._C = torch.empty(M, N_, dtype=torch.bfloat16, device=A.device)
            self._Bnk = B.t().contiguous()
            self._a = from_dlpack(A.unsqueeze(-1), assumed_align=16)
            self._b = from_dlpack(self._Bnk.unsqueeze(-1), assumed_align=16)
            self._c = from_dlpack(self._C.unsqueeze(-1), assumed_align=16)
            self._stream = cdr.CUstream(torch.cuda.current_stream().cuda_stream)
            self._compiled = cute.compile(self.gemm, self._a, self._b, self._c, self._stream)
            self._shape = sk; self._ptrs = (A.data_ptr(), B.data_ptr())
        pk = (A.data_ptr(), B.data_ptr())
        if pk != self._ptrs:
            self._Bnk = B.t().contiguous()
            self._a = from_dlpack(A.unsqueeze(-1), assumed_align=16)
            self._b = from_dlpack(self._Bnk.unsqueeze(-1), assumed_align=16)
            self._ptrs = pk
        self._compiled(self._a, self._b, self._c, self._stream)
        return self._C

# --- Generated (CUTLASS C++) ---
GEN_SO = "/tmp/sm90_matmul_generated.so"

class Generated:
    def __init__(self):
        if not os.path.exists(GEN_SO):
            raise FileNotFoundError(f"{GEN_SO} not found — compile first")
        self._lib = ctypes.CDLL(GEN_SO)
        self._lib.kernel_run.restype = ctypes.c_int
        self._lib.kernel_run.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
            ctypes.c_int, ctypes.c_void_p,
        ]
        self._C = None

    def __call__(self, A, B):
        M, K = A.shape; _, N_ = B.shape
        if self._C is None or self._C.shape != (M, N_):
            self._C = torch.empty(M, N_, dtype=torch.bfloat16, device=A.device)
        ins = (ctypes.c_void_p * 2)(A.data_ptr(), B.data_ptr())
        outs = (ctypes.c_void_p * 1)(self._C.data_ptr())
        stream = torch.cuda.current_stream().cuda_stream
        input_size = M * K  # kernel_run expects input_size = total elements per tensor
        rc = self._lib.kernel_run(ins, 2, outs, 1, input_size, ctypes.c_void_p(stream))
        if rc != 0:
            raise RuntimeError(f"kernel_run returned {rc}")
        return self._C

def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Benchmark: {W} warmup + {T} trials, median\n")

    cublas_fn = lambda A, B: torch.mm(A, B)
    dsl = DSL()
    try:
        gen = Generated()
        has_gen = True
    except Exception as ex:
        print(f"Generated kernel not available: {ex}")
        has_gen = False
        gen = None

    # Correctness
    print("Correctness check (1024x1024)...")
    A = torch.randn(1024, 1024, dtype=torch.bfloat16, device=DEV)
    B = torch.randn(1024, 1024, dtype=torch.bfloat16, device=DEV)
    ref = torch.mm(A, B)
    d1 = (ref.float() - dsl(A, B).float()).abs().max().item()
    print(f"  CuTe DSL vs cuBLAS: {d1:.6f} {'PASS' if d1 < 0.5 else 'FAIL'}")
    if has_gen:
        d2 = (ref.float() - gen(A, B).float()).abs().max().item()
        print(f"  Generated vs cuBLAS: {d2:.6f} {'PASS' if d2 < 0.5 else 'FAIL'}")
    print()

    # Header
    h = f"{'Config':>12s} | {'cuBLAS':>18s} | {'CuTe DSL':>18s} | {'Generated':>18s} | {'DSL/cub':>8s} | {'Gen/cub':>8s}"
    s = f"{'':>12s} | {'ms':>7s} {'TFLOPS':>8s} | {'ms':>7s} {'TFLOPS':>8s} | {'ms':>7s} {'TFLOPS':>8s} | {'':>8s} | {'':>8s}"
    print(h); print(s); print("-" * len(h))

    results = []
    for N in SIZES:
        c_ms = bench(cublas_fn, N);  c_tf = tf(N, c_ms)
        d_ms = bench(dsl, N);       d_tf = tf(N, d_ms)
        g_ms = bench(gen, N) if has_gen else float('nan'); g_tf = tf(N, g_ms) if has_gen else float('nan')

        dr = f"{d_tf/c_tf*100:.0f}%" if c_tf > 0 else "N/A"
        gr = f"{g_tf/c_tf*100:.0f}%" if c_tf > 0 and g_tf == g_tf else "N/A"

        print(f" {N:>5d}x{N:<5d} | {c_ms:>7.4f} {c_tf:>8.1f} | {d_ms:>7.4f} {d_tf:>8.1f} | {g_ms:>7.4f} {g_tf:>8.1f} | {dr:>8s} | {gr:>8s}")
        results.append({"size": N, "cublas_ms": c_ms, "cublas_tf": c_tf, "dsl_ms": d_ms, "dsl_tf": d_tf, "gen_ms": g_ms, "gen_tf": g_tf})

    out = f"/tmp/sm90_bench_all_{int(time.time())}.json"
    with open(out, "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(), "results": results}, f, indent=2)
    print(f"\nJSON: {out}")

if __name__ == "__main__":
    main()
