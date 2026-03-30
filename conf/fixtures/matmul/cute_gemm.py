"""CuTe DSL BF16 GEMM for SM120 — persistent TMA G2S + warp-specialised mma.sync.

Zero-copy from row-major PyTorch tensors.
Uses TMA for global->SMEM, warp specialization (1 DMA warp + 4 MMA warps),
ldmatrix S2R, and mma.sync compute.  Multi-stage SMEM pipeline with
PipelineTmaAsync barriers.  StaticPersistentTileScheduler drives a work loop
so each CTA processes multiple output tiles, improving occupancy for small
matrices and reducing launch overhead.

Convention (CuTe):
    A: (M, K), K-major (stride K,1) -- row-major PyTorch tensor
    B: (N, K), N-major (stride 1,N) -- from PyTorch B.t() (zero-copy view)
    C: (M, N), N-major (stride N,1) -- row-major output
"""
from __future__ import annotations

import time

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cute.runtime import from_dlpack


class Sm120Gemm:
    """BF16 GEMM for SM120: TMA G2S, ldmatrix S2R, mma.sync.

    Warp-specialised: 1 DMA warp (TMA producer) + 4 MMA warps (consumer).
    Multi-stage SMEM pipeline with PipelineTmaAsync barriers.
    Persistent kernel: StaticPersistentTileScheduler assigns multiple output
    tiles to each CTA via a work loop.
    """

    def __init__(
        self,
        bM: int = 128,
        bN: int = 128,
        bK: int = 64,
        num_mma_warps: int = 4,
        num_stages: int = 2,
        output_bf16: bool = False,
    ):
        self.bM = bM
        self.bN = bN
        self.bK = bK
        self.num_mma_warps = num_mma_warps
        self.num_threads = (num_mma_warps + 1) * 32  # +1 DMA warp
        self.num_stages = num_stages
        self.dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.output_bf16 = output_bf16
        # Tile shape in (M, N, K) order for use with slice_ helpers
        self.tile_shape_mnk = (bM, bN, bK)

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

        tile_mnk = self.tile_shape_mnk

        # ---- MMA atom (2x2x1 warp layout for 128x128 tile) ----
        mma_mnk = (16, 8, 16)
        op = warp.MmaF16BF16Op(a_dtype, self.acc_dtype, mma_mnk)
        atom_layout = (2, 2, 1)
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
        sA_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout_enum, tile_mnk, a_dtype, self.num_stages,
        )
        sB_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout_enum, tile_mnk, b_dtype, self.num_stages,
        )

        sA_layout = cute.slice_(sA_layout_staged, (None, None, 0))
        sB_layout = cute.slice_(sB_layout_staged, (None, None, 0))

        # ---- TMA atoms for G2S ----
        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_op, mA, sA_layout, (self.bM, self.bK), num_multicast=1,
        )
        tma_op_b = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            tma_op_b, mB, sB_layout, (self.bN, self.bK), num_multicast=1,
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

        # ---- Shared storage: mbar array + A+B staged ----
        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            sa: cute.struct.Align[
                cute.struct.MemRange[a_dtype, cute.cosize(sA_layout_staged)],
                1024,
            ]
            sb: cute.struct.Align[
                cute.struct.MemRange[b_dtype, cute.cosize(sB_layout_staged)],
                1024,
            ]

        self._shared_storage = SharedStorage

        # ---- Persistent tile scheduler ----
        num_ctas_mnl = (M // self.bM, N // self.bN, 1)
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl,
        )
        max_active = cutlass.const_expr(170)  # RTX 5090: 170 SMs
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active,
        )

        # ---- CTA layout for cluster (trivial: 1x1x1) ----
        cta_layout_mnk = cute.make_layout((1, 1, 1))

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            mC,
            tiled_mma,
            cta_layout_mnk,
            sA_layout_staged,
            sB_layout_staged,
            smem_copy_A,
            smem_copy_B,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nk: cute.Tensor,
        mC: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        sA_layout_staged: cute.ComposedLayout,
        sB_layout_staged: cute.ComposedLayout,
        smem_copy_A: cute.TiledCopy,
        smem_copy_B: cute.TiledCopy,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        num_stages = self.num_stages
        num_mma_warps = self.num_mma_warps

        # ---- Prefetch TMA descriptors ----
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        # ---- Allocate SMEM ----
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self._shared_storage)
        sA = storage.sa.get_tensor(
            sA_layout_staged.outer, swizzle=sA_layout_staged.inner
        )
        sB = storage.sb.get_tensor(
            sB_layout_staged.outer, swizzle=sB_layout_staged.inner
        )

        # ---- Compute tma_copy_bytes for pipeline tx_count ----
        sA_1stage = cute.slice_(sA_layout_staged, (None, None, 0))
        sB_1stage = cute.slice_(sB_layout_staged, (None, None, 0))
        tma_copy_bytes = (
            cute.size_in_bytes(self.dtype, sA_1stage)
            + cute.size_in_bytes(self.dtype, sB_1stage)
        )

        # ---- Pipeline setup ----
        mainloop_pipeline_array_ptr = storage.mbar.data_ptr()

        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cons_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_mma_warps
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        pipe = pipeline.PipelineTmaAsync.create(
            num_stages=num_stages,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        # ---- Global tiles: partition TMA tensors ----
        # mA_mk is the TMA tensor for A (shape: (M, K))
        # tile with (bM, bK) -> (bM, bK, M/bM, K/bK)
        gA_mk = cute.local_tile(
            mA_mk,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None),
        )
        # mB_nk is the TMA tensor for B (shape: (N, K))
        # tile with (bN, bK) -> (bN, bK, N/bN, K/bK)
        gB_nk = cute.local_tile(
            mB_nk,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None),
        )

        k_tile_count = cute.size(gA_mk, mode=[3])

        # ---- TMA partitions ----
        # For cluster (1,1,1): cta_crd = 0, cta_layout = make_layout((1,))
        a_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (0, None, 0)).shape
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (None, 0, 0)).shape
        )

        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, 0, a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mk, 0, 2),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nk, 0, 2),
        )

        # ---- MMA fragments (from single-stage slice) ----
        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])

        # ---- ldmatrix S2R retiling ----
        thr_s2r_A = smem_copy_A.get_slice(tidx)
        thr_s2r_B = smem_copy_B.get_slice(tidx)
        tCsA_copy_view = thr_s2r_A.partition_S(sA)
        tCrA_copy_view = thr_s2r_A.retile(tCrA)
        tCsB_copy_view = thr_s2r_B.partition_S(sB)
        tCrB_copy_view = thr_s2r_B.retile(tCrB)

        num_k_mma = cute.size(tCrA, mode=[2])

        # ---- Accumulator ----
        acc_shape = thr_mma.partition_shape_C((self.bM, self.bN))
        acc = cute.make_fragment(acc_shape, self.acc_dtype)

        # ---- Pipeline states ----
        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, num_stages,
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_stages,
        )

        # ---- Persistent tile scheduler ----
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim(),
        )
        work_tile = tile_sched.initial_work_tile_info()

        # Barrier sync before entering warp-specialized regions
        pipeline.sync(barrier_id=1)

        # ================================================================
        # MMA warps (consumers): ldmatrix + mma.sync
        # ================================================================
        if warp_idx < num_mma_warps:
            cute.arch.setmaxregister_increase(232)

            while work_tile.is_valid_tile:
                tile_m, tile_n, _ = work_tile.tile_idx

                # Clear accumulator for this output tile
                acc.fill(0.0)

                # Compute C tile for this work tile
                gC = cute.local_tile(mC, (self.bM, self.bN), (tile_m, tile_n))

                # Reset consumer pipeline count for new tile
                cons_state.reset_count()

                # Pre-wait for first K tile
                peek_status = cutlass.Boolean(1)
                if cons_state.count < k_tile_count:
                    peek_status = pipe.consumer_try_wait(cons_state)

                pipe.consumer_wait(cons_state, peek_status)
                tCsA_p = tCsA_copy_view[None, None, None, cons_state.index]
                tCsB_p = tCsB_copy_view[None, None, None, cons_state.index]
                cute.copy(
                    smem_copy_A,
                    tCsA_p[None, None, 0],
                    tCrA_copy_view[None, None, 0],
                )
                cute.copy(
                    smem_copy_B,
                    tCsB_p[None, None, 0],
                    tCrB_copy_view[None, None, 0],
                )

                # Main loop over K tiles (all but last)
                for kt in range(0, k_tile_count - 1, 1, unroll=1):
                    for k in cutlass.range_constexpr(num_k_mma):
                        k_next = 0 if k + 1 == num_k_mma else k + 1

                        if k == num_k_mma - 1:
                            # Release current stage, advance to next
                            pipe.consumer_release(cons_state)
                            cons_state.advance()

                            peek_status = cutlass.Boolean(1)
                            peek_status = pipe.consumer_try_wait(cons_state)

                            tCsA_p = tCsA_copy_view[
                                None, None, None, cons_state.index
                            ]
                            tCsB_p = tCsB_copy_view[
                                None, None, None, cons_state.index
                            ]
                            pipe.consumer_wait(cons_state, peek_status)

                        # Prefetch next k-block from SMEM to registers
                        cute.copy(
                            smem_copy_A,
                            tCsA_p[None, None, k_next],
                            tCrA_copy_view[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B,
                            tCsB_p[None, None, k_next],
                            tCrB_copy_view[None, None, k_next],
                        )

                        # MMA for current k-block
                        cute.gemm(
                            tiled_mma, acc,
                            tCrA[None, None, k], tCrB[None, None, k], acc,
                        )

                # Last K tile (hoisted out of loop)
                for k in cutlass.range_constexpr(num_k_mma):
                    k_next = 0 if k + 1 == num_k_mma else k + 1

                    if k == num_k_mma - 1:
                        pipe.consumer_release(cons_state)
                        cons_state.advance()

                    if k_next > 0:
                        cute.copy(
                            smem_copy_A,
                            tCsA_p[None, None, k_next],
                            tCrA_copy_view[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B,
                            tCsB_p[None, None, k_next],
                            tCrB_copy_view[None, None, k_next],
                        )

                    cute.gemm(
                        tiled_mma, acc,
                        tCrA[None, None, k], tCrB[None, None, k], acc,
                    )

                # ---- Store accumulator to global memory ----
                tCgC = thr_mma.partition_C(gC)
                out_dtype = self.dtype if self.output_bf16 else self.acc_dtype
                out_frag = cute.make_fragment(acc_shape, out_dtype)
                acc_vec = acc.load()
                out_frag.store(acc_vec.to(out_dtype))
                st_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), out_dtype
                )
                cute.copy(st_atom, out_frag, tCgC)

                # Advance to next work tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # ================================================================
        # DMA warp (producer): TMA loads
        # ================================================================
        elif warp_idx == num_mma_warps:
            cute.arch.setmaxregister_decrease(40)

            while work_tile.is_valid_tile:
                tile_m, tile_n, _ = work_tile.tile_idx

                # Select global tile slices for this work tile
                # tAgA shape: (tma, M_tiles, K_tiles)
                tAgA_mk = tAgA[(None, tile_m, None)]
                # tBgB shape: (tma, N_tiles, K_tiles)
                tBgB_nk = tBgB[(None, tile_n, None)]

                # Reset producer pipeline count for new tile
                prod_state.reset_count()

                for kt in range(0, k_tile_count, 1, unroll=1):
                    pipe.producer_acquire(prod_state)

                    tAgA_k = tAgA_mk[(None, prod_state.count)]
                    tAsA_pipe = tAsA[(None, prod_state.index)]

                    tBgB_k = tBgB_nk[(None, prod_state.count)]
                    tBsB_pipe = tBsB[(None, prod_state.index)]

                    cute.copy(
                        tma_atom_a, tAgA_k, tAsA_pipe,
                        tma_bar_ptr=pipe.producer_get_barrier(prod_state),
                        mcast_mask=0,
                    )
                    cute.copy(
                        tma_atom_b, tBgB_k, tBsB_pipe,
                        tma_bar_ptr=pipe.producer_get_barrier(prod_state),
                        mcast_mask=0,
                    )

                    pipe.producer_commit(prod_state)
                    prod_state.advance()

                # Advance to next work tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Drain: wait for all consumer releases
            pipe.producer_tail(prod_state)


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

    gemm = Sm120Gemm(bM=128, bN=128, bK=64, num_mma_warps=4, num_stages=2)
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

    # Benchmark various sizes (each needs its own compilation)
    # Small sizes (256, 512) benefit from persistent scheduling
    for sz in [256, 512, 1024, 4096]:
        A2 = torch.randn(sz, sz, dtype=torch.bfloat16, device="cuda")
        B2 = torch.randn(sz, sz, dtype=torch.bfloat16, device="cuda")
        C2 = torch.zeros(sz, sz, dtype=torch.float32, device="cuda")
        a2 = from_dlpack(A2, assumed_align=16)
        b2 = from_dlpack(B2.t(), assumed_align=16)
        c2 = from_dlpack(C2, assumed_align=16)

        gemm2 = Sm120Gemm(bM=128, bN=128, bK=64, num_mma_warps=4, num_stages=2)
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
