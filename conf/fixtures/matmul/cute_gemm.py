"""CuTe DSL BF16 GEMM for SM120 — zero-copy row-major inputs.

Persistent kernel: TMA global→SMEM, ldmatrix SMEM→register, mma.sync compute.
Accepts PyTorch row-major A (M,K) and B (K,N) directly — no copy, no transpose.

Convention (CuTe):
    A: (M, K), K-major (stride K,1) — row-major PyTorch tensor
    B: (N, K), N-major (stride 1,N) — from PyTorch B.t() (zero-copy view)
    C: (M, N), N-major (stride N,1) — row-major output
"""
from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cute.runtime import from_dlpack


class Sm120Gemm:
    """BF16 GEMM for SM120: TMA + mma.sync + persistent kernel."""

    def __init__(self):
        self.bM = 128
        self.bN = 128
        self.bK = 64
        self.stage = 2  # pipeline stages (fits in SM120's 99KB SMEM)
        self.acc_dtype = cutlass.Float32
        self.atom_layout = (2, 2, 1)
        self.num_mma_warps = 4
        self.threads_per_cta = (self.num_mma_warps + 1) * 32  # 160

        self.mma_inst_mnk = (16, 8, 16)
        self.load_register_requirement = 40
        self.mma_register_requirement = 232

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

        # ---- MMA ----
        op = cute.nvgpu.warp.MmaF16BF16Op(a_dtype, self.acc_dtype, self.mma_inst_mnk)
        tC = cute.make_layout(self.atom_layout)
        permutation_mnk = (
            self.atom_layout[0] * self.mma_inst_mnk[0],
            self.atom_layout[1] * self.mma_inst_mnk[1] * 2,
            self.atom_layout[2] * self.mma_inst_mnk[2],
        )
        tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)

        # ---- SMEM layouts ----
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout_enum, (self.bM, self.bN, self.bK), a_dtype, self.stage,
        )
        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout_enum, (self.bM, self.bN, self.bK), b_dtype, self.stage,
        )

        # ---- TMA load atoms ----
        tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()

        a_smem_1stage = cute.slice_(a_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_op, mA, a_smem_1stage, (self.bM, self.bK), num_multicast=1,
        )
        b_smem_1stage = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_op, mB, b_smem_1stage, (self.bN, self.bK), num_multicast=1,
        )

        # ---- Grid ----
        M = mA.layout.shape[0]
        N = mB.layout.shape[0]
        num_ctas_mnl = (M // self.bM, N // self.bN, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, (1, 1, 1)
        )
        max_active = cutlass.const_expr(170)
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active
        )

        # ---- Shared storage ----
        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cutlass.Int64, self.stage * 2]
            sa: cute.struct.Align[
                cute.struct.MemRange[a_dtype, cute.cosize(a_smem_layout_staged)], 1024
            ]
            sb: cute.struct.Align[
                cute.struct.MemRange[b_dtype, cute.cosize(b_smem_layout_staged)], 1024
            ]

        self.shared_storage = SharedStorage

        self._a_is_m_major = a_layout_enum.is_m_major_a()
        self._b_is_n_major = b_layout_enum.is_n_major_b()

        self.kernel(
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            mC, tiled_mma,
            a_smem_layout_staged, b_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nk: cute.Tensor,
        mC_mn: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        # ---- Shared memory ----
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # ---- Pipeline ----
        a_smem_1 = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_1 = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_bytes = (
            cute.size_in_bytes(cutlass.BFloat16, a_smem_1)
            + cute.size_in_bytes(cutlass.BFloat16, b_smem_1)
        )

        cta_layout = cute.make_layout((1, 1, 1))
        cta_layout_v = cute.make_layout((1, *cta_layout.shape))

        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cons_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            size=self.num_mma_warps * 32,
        )

        pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar.data_ptr(),
            num_stages=self.stage,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=tma_bytes,
            cta_layout_vmnk=cta_layout_v,
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.stage
        )
        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.stage
        )

        # ---- SMEM tensors ----
        sA = storage.sa.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sb.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        # ---- Global partitions ----
        gA = cute.local_tile(mA_mk, (self.bM, self.bK), (None, None))
        gB = cute.local_tile(mB_nk, (self.bN, self.bK), (None, None))
        gC = cute.local_tile(mC_mn, (self.bM, self.bN), (None, None))

        # ---- TMA partitions ----
        a_cta = cute.make_layout(cute.slice_(cta_layout, (0, None, 0)).shape)
        sa_grp = cute.group_modes(sA, 0, 2)
        gA_grp = cute.group_modes(gA, 0, 2)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, 1, a_cta, sa_grp, gA_grp,
        )

        b_cta = cute.make_layout(cute.slice_(cta_layout, (None, 0, 0)).shape)
        sb_grp = cute.group_modes(sB, 0, 2)
        gB_grp = cute.group_modes(gB, 0, 2)
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b, 1, b_cta, sb_grp, gB_grp,
        )

        # ---- MMA partitions ----
        thr_mma = tiled_mma.get_slice(tidx)
        tCgC = thr_mma.partition_C(gC)

        # ---- Register fragments (must create before retile) ----
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))

        # ---- ldmatrix: SMEM → registers ----
        atom_ldm_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(self._a_is_m_major, 4),
            cutlass.BFloat16,
        )
        atom_ldm_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(self._b_is_n_major, 4),
            cutlass.BFloat16,
        )
        copy_A = cute.make_tiled_copy_A(atom_ldm_A, tiled_mma)
        copy_B = cute.make_tiled_copy_B(atom_ldm_B, tiled_mma)

        thr_cA = copy_A.get_slice(tidx)
        tCsA = thr_cA.partition_S(sA)
        tCrA_copy = thr_cA.retile(tCrA)  # retile REGISTER fragment

        thr_cB = copy_B.get_slice(tidx)
        tCsB = thr_cB.partition_S(sB)
        tCrB_copy = thr_cB.retile(tCrB)  # retile REGISTER fragment

        k_tile_cnt = cute.size(tAgA, mode=[2])
        num_k_blocks = cute.size(tCrA, mode=[2])

        cute.arch.sync_threads()

        # ---- Persistent scheduler ----
        sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work = sched.initial_work_tile_info()

        # ======== MMA warps (0..3) ========
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            while work.is_valid_tile:
                tile_m, tile_n, _ = work.tile_idx
                # Reset accumulators for new tile
                accum = cute.make_fragment(
                    tCgC[None, None, None, 0, 0].shape, self.acc_dtype
                )

                for _ in cutlass.range(k_tile_cnt):
                    pipe.consumer_wait(cons_state)

                    for kb in cutlass.range(num_k_blocks, unroll_full=True):
                        kc = (None, None, kb, cons_state.index)
                        cute.copy(copy_A, tCsA[kc], tCrA_copy[kc])
                        cute.copy(copy_B, tCsB[kc], tCrB_copy[kc])
                        cute.gemm(tiled_mma, accum, tCrA[kc], tCrB[kc], accum)

                    pipe.consumer_release(cons_state)
                    cons_state.advance()

                # Store results
                epilog_bar = pipeline.NamedBarrier(
                    barrier_id=2, num_threads=self.num_mma_warps * 32
                )
                epilog_bar.arrive_and_wait()

                st_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), self.acc_dtype
                )
                cute.copy(st_atom, accum, tCgC[None, None, None, tile_m, tile_n])

                sched.advance_to_next_work()
                work = sched.get_current_work()

        # ======== DMA warp (4) ========
        elif warp_idx == self.num_mma_warps:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)
            lane_in_warp = cute.arch.warp_idx() % 4
            lane_in_warp = cute.arch.make_warp_uniform(lane_in_warp)

            while work.is_valid_tile:
                tile_m, tile_n, _ = work.tile_idx

                for kt in cutlass.range(k_tile_cnt):
                    pipe.producer_acquire(prod_state)

                    if lane_in_warp == 0:
                        cute.copy(
                            tma_atom_a,
                            tAgA[(None, tile_m, kt)],
                            tAsA[(None, prod_state.index)],
                            tma_bar_ptr=pipe.producer_get_barrier(prod_state),
                            mcast_mask=0,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, tile_n, kt)],
                            tBsB[(None, prod_state.index)],
                            tma_bar_ptr=pipe.producer_get_barrier(prod_state),
                            mcast_mask=0,
                        )

                    pipe.producer_commit(prod_state)
                    prod_state.advance()

                sched.advance_to_next_work()
                work = sched.get_current_work()


def test():
    """Verify correctness."""
    M, K, N = 256, 256, 256

    A = torch.arange(M * K, dtype=torch.bfloat16, device="cuda").reshape(M, K).contiguous()
    B = torch.arange(K * N, dtype=torch.bfloat16, device="cuda").reshape(K, N).contiguous()
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    # Zero-copy
    a_cute = from_dlpack(A, assumed_align=16)
    b_cute = from_dlpack(B.t(), assumed_align=16)  # (N,K) N-major, zero copy
    c_cute = from_dlpack(C, assumed_align=16)

    print(f"A: shape={a_cute.shape}, ld={a_cute.leading_dim}, layout={utils.LayoutEnum.from_tensor(a_cute)}")
    print(f"B: shape={b_cute.shape}, ld={b_cute.leading_dim}, layout={utils.LayoutEnum.from_tensor(b_cute)}")

    gemm = Sm120Gemm()
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    print("Compiling...")
    compiled = cute.compile(gemm, a_cute, b_cute, c_cute, stream=stream)

    print("Running...")
    compiled(a_cute, b_cute, c_cute, stream)
    torch.cuda.synchronize()

    ref = A.float() @ B.float()
    diff = (C - ref).abs().max().item()
    print(f"max_diff = {diff:.6f}")
    print(f"Correctness: {'PASSED' if diff < 1.0 else 'FAILED'}")
    return diff < 1.0


if __name__ == "__main__":
    import sys
    sys.exit(0 if test() else 1)
