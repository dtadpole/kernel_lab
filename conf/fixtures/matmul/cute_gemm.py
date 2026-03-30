"""CuTe DSL BF16 GEMM for SM120 — multi-stage cp.async + mma.sync kernel.

Zero-copy from row-major PyTorch tensors.
No TMA, no persistence, no warp specialization. Double-buffered SMEM
to overlap global memory loads with MMA computation.

Convention (CuTe):
    A: (M, K), K-major (stride K,1) — row-major PyTorch tensor
    B: (N, K), N-major (stride 1,N) — from PyTorch B.t() (zero-copy view)
    C: (M, N), N-major (stride N,1) — row-major output
"""
from __future__ import annotations

import time

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cute.runtime import from_dlpack


class Sm120Gemm:
    """BF16 GEMM for SM120: cp.async G2S, ldmatrix S2R, mma.sync.

    One tile per CTA, multi-stage SMEM double-buffering, all threads
    do load+compute.
    """

    def __init__(
        self,
        bM: int = 64,
        bN: int = 64,
        bK: int = 32,
        num_warps: int = 4,
        num_stages: int = 2,
    ):
        self.bM = bM
        self.bN = bN
        self.bK = bK
        self.num_warps = num_warps
        self.num_threads = num_warps * 32
        self.num_stages = num_stages
        self.dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        stream: cuda_driver.CUstream,
    ):
        a_dtype = mA.element_type
        b_dtype = mB.element_type

        a_layout_enum = utils.LayoutEnum.from_tensor(mA)
        b_layout_enum = utils.LayoutEnum.from_tensor(mB)

        M = mA.layout.shape[0]
        K = mA.layout.shape[1]
        N = mB.layout.shape[0]

        tile_mnk = (self.bM, self.bN, self.bK)

        # ---- MMA atom ----
        mma_mnk = (16, 8, 16)
        op = warp.MmaF16BF16Op(a_dtype, self.acc_dtype, mma_mnk)
        atom_layout = (self.num_warps, 1, 1)
        permutation_mnk = (
            atom_layout[0] * mma_mnk[0],
            atom_layout[1] * mma_mnk[1] * 2,
            atom_layout[2] * mma_mnk[2],
        )
        tiled_mma = cute.make_tiled_mma(
            op,
            cute.make_layout(atom_layout),
            permutation_mnk=permutation_mnk,
        )

        # ---- SMEM layouts via sm90_utils (handles K-major vs MN-major) ----
        # Multi-stage: (M/N, K, num_stages)
        sA_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout_enum, tile_mnk, a_dtype, self.num_stages,
        )
        sB_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout_enum, tile_mnk, b_dtype, self.num_stages,
        )

        # Single-stage slice for partitioning (MMA, ldmatrix, copy setup)
        sA_layout = cute.slice_(sA_layout_staged, (None, None, 0))
        sB_layout = cute.slice_(sB_layout_staged, (None, None, 0))

        # ---- Global->SMEM tiled copy (cp.async, 128-bit vectorized) ----
        copy_bits = 128
        copy_elems = copy_bits // a_dtype.width  # 8 bf16 elements
        g2s_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(),
            a_dtype,
            num_bits_per_copy=copy_bits,
        )

        # A: K-major, vectorize along K (mode 1)
        a_k_threads = self.bK // copy_elems
        a_m_threads = self.num_threads // a_k_threads
        thr_layout_A = cute.make_ordered_layout(
            (a_m_threads, a_k_threads), order=(1, 0)
        )
        val_layout_A = cute.make_layout((1, copy_elems))
        gmem_copy_A = cute.make_tiled_copy_tv(
            g2s_atom, thr_layout_A, val_layout_A
        )

        # B: N-major, vectorize along N (mode 0)
        b_n_threads = self.bN // copy_elems
        b_k_threads = self.num_threads // b_n_threads
        thr_layout_B = cute.make_ordered_layout(
            (b_n_threads, b_k_threads), order=(0, 1)
        )
        val_layout_B = cute.make_layout((copy_elems, 1))
        gmem_copy_B = cute.make_tiled_copy_tv(
            g2s_atom, thr_layout_B, val_layout_B
        )

        # ---- SMEM->Register copy atoms (ldmatrix) ----
        a_is_m_major = a_layout_enum.is_m_major_a()
        b_is_n_major = b_layout_enum.is_n_major_b()
        ldm_atom_A = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=a_is_m_major, num_matrices=4),
            a_dtype,
        )
        ldm_atom_B = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=b_is_n_major, num_matrices=4),
            b_dtype,
        )
        smem_copy_A = cute.make_tiled_copy_A(ldm_atom_A, tiled_mma)
        smem_copy_B = cute.make_tiled_copy_B(ldm_atom_B, tiled_mma)

        # ---- Shared storage: sized for full staged layout ----
        @cute.struct
        class SharedStorage:
            sa: cute.struct.Align[
                cute.struct.MemRange[a_dtype, cute.cosize(sA_layout_staged)],
                128,
            ]
            sb: cute.struct.Align[
                cute.struct.MemRange[b_dtype, cute.cosize(sB_layout_staged)],
                128,
            ]

        self._shared_storage = SharedStorage

        # ---- Grid: one CTA per (m_tile, n_tile) ----
        grid = (M // self.bM, N // self.bN, 1)

        self.kernel(
            mA,
            mB,
            mC,
            tiled_mma,
            sA_layout_staged,
            sB_layout_staged,
            sA_layout,
            sB_layout,
            gmem_copy_A,
            gmem_copy_B,
            smem_copy_A,
            smem_copy_B,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        tiled_mma: cute.TiledMma,
        sA_layout_staged: cute.ComposedLayout,
        sB_layout_staged: cute.ComposedLayout,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        gmem_copy_A: cute.TiledCopy,
        gmem_copy_B: cute.TiledCopy,
        smem_copy_A: cute.TiledCopy,
        smem_copy_B: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()

        num_stages = self.num_stages

        # ---- Allocate SMEM (full staged) ----
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self._shared_storage)
        sA_full = storage.sa.get_tensor(
            sA_layout_staged.outer, swizzle=sA_layout_staged.inner
        )
        sB_full = storage.sb.get_tensor(
            sB_layout_staged.outer, swizzle=sB_layout_staged.inner
        )

        # ---- Global tiles for this CTA ----
        gA = cute.local_tile(mA, (self.bM, self.bK), (bx, None))
        gB = cute.local_tile(mB, (self.bN, self.bK), (by, None))
        gC = cute.local_tile(mC, (self.bM, self.bN), (bx, by))

        k_tile_count = cute.size(gA, mode=[2])

        # ---- Set up G2S, MMA, S2R for single-stage slice (stage 0) ----
        sA0 = cute.slice_(sA_full, (None, None, 0))
        sB0 = cute.slice_(sB_full, (None, None, 0))

        thr_g2s_A = gmem_copy_A.get_slice(tidx)
        tAgA = thr_g2s_A.partition_S(gA)
        tAsA0 = thr_g2s_A.partition_D(sA0)

        thr_g2s_B = gmem_copy_B.get_slice(tidx)
        tBgB = thr_g2s_B.partition_S(gB)
        tBsB0 = thr_g2s_B.partition_D(sB0)

        thr_mma = tiled_mma.get_slice(tidx)
        tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA0))
        tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB0))

        thr_s2r_A = smem_copy_A.get_slice(tidx)
        tCrA_view = thr_s2r_A.retile(tCrA)

        thr_s2r_B = smem_copy_B.get_slice(tidx)
        tCrB_view = thr_s2r_B.retile(tCrB)

        # Per-stage S2R source and G2S dest partitions
        tCsA0 = thr_s2r_A.partition_S(sA0)
        tCsB0 = thr_s2r_B.partition_S(sB0)

        num_k_mma = cute.size(tCsA0, mode=[2])

        # Also set up stage 1 partitions
        sA1 = cute.slice_(sA_full, (None, None, 1))
        sB1 = cute.slice_(sB_full, (None, None, 1))
        tAsA1 = thr_g2s_A.partition_D(sA1)
        tBsB1 = thr_g2s_B.partition_D(sB1)
        tCsA1 = thr_s2r_A.partition_S(sA1)
        tCsB1 = thr_s2r_B.partition_S(sB1)

        # ---- Accumulator (zero-initialized) ----
        acc_shape = thr_mma.partition_shape_C((self.bM, self.bN))
        acc = cute.make_fragment(acc_shape, self.acc_dtype)
        acc.fill(0.0)

        # ---- Prelude: load first 2 tiles into stages 0 and 1 ----
        cute.copy(gmem_copy_A, tAgA[None, None, None, 0], tAsA0)
        cute.copy(gmem_copy_B, tBgB[None, None, None, 0], tBsB0)
        cute.arch.cp_async_commit_group()

        if k_tile_count > 1:
            cute.copy(gmem_copy_A, tAgA[None, None, None, 1], tAsA1)
            cute.copy(gmem_copy_B, tBgB[None, None, None, 1], tBsB1)
        cute.arch.cp_async_commit_group()

        # ---- Main loop over K tiles ----
        for kt in cutlass.range(k_tile_count):
            # Wait for current stage to be ready (1 group still in flight)
            cute.arch.cp_async_wait_group(1)
            cute.arch.sync_threads()

            # Select S2R source based on even/odd stage
            if kt % 2 == 0:
                for k in cutlass.range_constexpr(num_k_mma):
                    cute.copy(
                        smem_copy_A,
                        tCsA0[None, None, k],
                        tCrA_view[None, None, k],
                    )
                    cute.copy(
                        smem_copy_B,
                        tCsB0[None, None, k],
                        tCrB_view[None, None, k],
                    )
                    cute.gemm(
                        tiled_mma, acc,
                        tCrA[None, None, k], tCrB[None, None, k], acc,
                    )
            else:
                for k in cutlass.range_constexpr(num_k_mma):
                    cute.copy(
                        smem_copy_A,
                        tCsA1[None, None, k],
                        tCrA_view[None, None, k],
                    )
                    cute.copy(
                        smem_copy_B,
                        tCsB1[None, None, k],
                        tCrB_view[None, None, k],
                    )
                    cute.gemm(
                        tiled_mma, acc,
                        tCrA[None, None, k], tCrB[None, None, k], acc,
                    )

            cute.arch.sync_threads()

            # Start loading the NEXT tile (kt + 2) if it exists
            next_kt = kt + 2
            if next_kt < k_tile_count:
                if kt % 2 == 0:
                    cute.copy(
                        gmem_copy_A,
                        tAgA[None, None, None, next_kt],
                        tAsA0,
                    )
                    cute.copy(
                        gmem_copy_B,
                        tBgB[None, None, None, next_kt],
                        tBsB0,
                    )
                else:
                    cute.copy(
                        gmem_copy_A,
                        tAgA[None, None, None, next_kt],
                        tAsA1,
                    )
                    cute.copy(
                        gmem_copy_B,
                        tBgB[None, None, None, next_kt],
                        tBsB1,
                    )
            cute.arch.cp_async_commit_group()

        # Drain any remaining async groups
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # ---- Store accumulator to global memory ----
        st_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.acc_dtype
        )
        tCgC = thr_mma.partition_C(gC)
        cute.copy(st_atom, acc, tCgC)


def test():
    """Verify correctness at 256x256 and benchmark larger sizes."""
    M, K, N = 256, 256, 256

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    # Zero-copy CuTe tensors
    a_cute = from_dlpack(A, assumed_align=16)  # (M, K) K-major
    b_cute = from_dlpack(B.t(), assumed_align=16)  # (N, K) N-major
    c_cute = from_dlpack(C, assumed_align=16)  # (M, N) N-major

    print(
        f"A cute: shape={a_cute.shape}, "
        f"layout={utils.LayoutEnum.from_tensor(a_cute)}"
    )
    print(
        f"B cute: shape={b_cute.shape}, "
        f"layout={utils.LayoutEnum.from_tensor(b_cute)}"
    )
    print(f"C cute: shape={c_cute.shape}")

    gemm = Sm120Gemm(bM=64, bN=64, bK=32, num_warps=4, num_stages=2)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    print("Compiling (MLIR JIT)...")
    t0 = time.time()
    compiled = cute.compile(gemm, a_cute, b_cute, c_cute, stream=stream)
    t_compile = time.time() - t0
    print(f"Compiled in {t_compile:.1f}s")

    print("Running 256x256x256...")
    compiled(a_cute, b_cute, c_cute, stream)
    torch.cuda.synchronize()

    ref = A.float() @ B.float()
    diff = (C - ref).abs().max().item()
    print(f"max_diff = {diff:.6f}")
    passed = diff < 1.0
    print(f"Correctness: {'PASSED' if passed else 'FAILED'}")

    if not passed:
        return False

    # Benchmark larger sizes (each needs its own compilation)
    for sz in [1024, 4096]:
        A2 = torch.randn(sz, sz, dtype=torch.bfloat16, device="cuda")
        B2 = torch.randn(sz, sz, dtype=torch.bfloat16, device="cuda")
        C2 = torch.zeros(sz, sz, dtype=torch.float32, device="cuda")
        a2 = from_dlpack(A2, assumed_align=16)
        b2 = from_dlpack(B2.t(), assumed_align=16)
        c2 = from_dlpack(C2, assumed_align=16)

        gemm2 = Sm120Gemm(bM=64, bN=64, bK=32, num_warps=4, num_stages=2)
        compiled2 = cute.compile(gemm2, a2, b2, c2, stream=stream)

        # Warmup
        compiled2(a2, b2, c2, stream)
        torch.cuda.synchronize()

        # Timed runs
        n_iter = 20
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            compiled2(a2, b2, c2, stream)
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) / n_iter * 1000  # ms

        ref2 = A2.float() @ B2.float()
        diff2 = (C2 - ref2).abs().max().item()
        ok = diff2 < 1.0
        flops = 2 * sz * sz * sz / (elapsed / 1000) / 1e12
        print(
            f"{sz}x{sz}: {elapsed:.3f} ms, {flops:.2f} TFLOPS, "
            f"diff={diff2:.4f}, {'PASSED' if ok else 'FAILED'}"
        )
        if not ok:
            passed = False

    return passed


if __name__ == "__main__":
    import sys

    sys.exit(0 if test() else 1)
