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
    """BF16 GEMM for SM120: TMA G2S, ldmatrix S2R, mma.sync, TMA S2G epilogue.

    Warp-specialised: 1 DMA warp (TMA producer) + 4 MMA warps (consumer).
    Multi-stage SMEM pipeline with PipelineTmaAsync barriers.
    TMA store epilogue: stmatrix R2S + TMA S2G for coalesced output writes.
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
        output_bf16: bool = True,
        max_active: int = 188,
        epi_stages: int = 8,
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
        # atom_layout: product must equal num_mma_warps (each atom = 1 warp).
        # Supported: 4 warps = (2,2,1). Larger tile with more warps hits
        # register pressure limits on SM120 (see NCU analysis 2026-04-01).
        _atom_layouts = {2: (2, 1, 1), 4: (2, 2, 1), 6: (3, 2, 1), 8: (4, 2, 1)}
        self.atom_layout = _atom_layouts.get(num_mma_warps, (num_mma_warps, 1, 1))
        self.max_active = max_active
        self.epi_stages = epi_stages
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=num_mma_warps * 32,
        )

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

        # ---- MMA atom: atom_layout set in __init__ based on num_mma_warps ----
        mma_mnk = (16, 8, 16)
        op = warp.MmaF16BF16Op(a_dtype, self.acc_dtype, mma_mnk)
        atom_layout = self.atom_layout
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

        # ---- Epilogue: TMA store (stmatrix R2S + TMA S2G) ----
        c_dtype = self.dtype if self.output_bf16 else self.acc_dtype
        c_layout_enum = utils.LayoutEnum.from_tensor(mC)
        self._c_layout_enum = c_layout_enum
        epi_tile = sm90_utils.compute_tile_shape_or_override(
            tile_mnk, c_dtype, is_cooperative=False,
        )
        self._epi_tile = epi_tile
        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            c_dtype, c_layout_enum, epi_tile, self.epi_stages,
        )
        self._epi_smem_layout_staged = epi_smem_layout_staged

        # TMA store atom for S2G
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mC, epi_smem_layout, epi_tile,
        )

        # ---- Shared storage: mbar + A+B staged + C epilogue ----
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
            sc: cute.struct.Align[
                cute.struct.MemRange[c_dtype, cute.cosize(epi_smem_layout_staged)],
                1024,
            ]

        self._shared_storage = SharedStorage

        # ---- Persistent tile scheduler ----
        num_ctas_mnl = (M // self.bM, N // self.bN, 1)
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl,
        )
        max_active = cutlass.const_expr(self.max_active)
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
            tma_atom_c,
            tma_tensor_c,
            mC,
            tiled_mma,
            cta_layout_mnk,
            sA_layout_staged,
            sB_layout_staged,
            epi_smem_layout_staged,
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
        tma_atom_c: cute.CopyAtom,
        mC_tma: cute.Tensor,
        mC_mn: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        sA_layout_staged: cute.ComposedLayout,
        sB_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
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
            cpasync.prefetch_descriptor(tma_atom_c)

        # ---- Allocate SMEM ----
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self._shared_storage)
        sA = storage.sa.get_tensor(
            sA_layout_staged.outer, swizzle=sA_layout_staged.inner
        )
        sB = storage.sb.get_tensor(
            sB_layout_staged.outer, swizzle=sB_layout_staged.inner
        )
        sC = storage.sc.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
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

        # ---- Global C tensor partitioning for epilogue ----
        gC_mn = cute.local_tile(
            mC_mn, (self.bM, self.bN), (None, None),
        )
        tCgC = thr_mma.partition_C(gC_mn)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # ---- ldmatrix S2R retiling ----
        thr_s2r_A = smem_copy_A.get_slice(tidx)
        thr_s2r_B = smem_copy_B.get_slice(tidx)
        tCsA_copy_view = thr_s2r_A.partition_S(sA)
        tCrA_copy_view = thr_s2r_A.retile(tCrA)
        tCsB_copy_view = thr_s2r_B.partition_S(sB)
        tCrB_copy_view = thr_s2r_B.retile(tCrB)

        num_k_mma = cute.size(tCrA, mode=[2])

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
                accumulators.fill(0.0)

                gC_mn_slice = gC_mn[(None, None, tile_m, tile_n)]

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
                            tiled_mma, accumulators,
                            tCrA[None, None, k], tCrB[None, None, k],
                            accumulators,
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
                        tiled_mma, accumulators,
                        tCrA[None, None, k], tCrB[None, None, k],
                        accumulators,
                    )

                # ---- TMA store epilogue: stmatrix R2S + TMA S2G ----
                out_dtype = self.dtype if self.output_bf16 else self.acc_dtype

                copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                    self._c_layout_enum,
                    elem_ty_d=out_dtype,
                    elem_ty_acc=self.acc_dtype,
                )
                copy_atom_C = cute.make_copy_atom(
                    warp.StMatrix8x8x16bOp(
                        self._c_layout_enum.is_m_major_c(), 4,
                    ),
                    out_dtype,
                )
                tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(
                    copy_atom_C, tiled_mma,
                )
                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s, tiled_copy_C_Atom,
                )

                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                tRS_rAcc = tiled_copy_r2s.retile(accumulators)

                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
                size_tRS_rD = cute.size(tRS_rD)

                sepi_for_tma = cute.group_modes(sC, 0, 2)
                tcgc_for_tma = cute.zipped_divide(gC_mn_slice, self._epi_tile)

                bSG_sD, bSG_gD = cpasync.tma_partition(
                    tma_atom_c, 0, cute.make_layout(1),
                    sepi_for_tma, tcgc_for_tma,
                )

                epi_tile_num = cute.size(tcgc_for_tma, mode=[1])
                epi_tile_shape = tcgc_for_tma.shape[1]
                epi_tile_layout = cute.make_layout(
                    epi_tile_shape, stride=(1, epi_tile_shape[0]),
                )

                tma_store_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    num_mma_warps * 32,
                )
                tma_store_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.epi_stages,
                    producer_group=tma_store_producer_group,
                )

                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

                    tRS_rD_out = cute.make_rmem_tensor(
                        tRS_rD_layout.shape, out_dtype,
                    )
                    acc_vec = tRS_rD.load()
                    tRS_rD_out.store(acc_vec.to(out_dtype))

                    epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
                    cute.copy(
                        tiled_copy_r2s, tRS_rD_out,
                        tRS_sD[(None, None, None, epi_buffer)],
                    )

                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()

                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    if warp_idx == 0:
                        cute.copy(
                            tma_atom_c,
                            bSG_sD[(None, epi_buffer)],
                            bSG_gD[(None, gmem_coord)],
                        )
                        tma_store_pipeline.producer_commit()
                        tma_store_pipeline.producer_acquire()

                # Advance to next work tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                tma_store_pipeline.producer_tail()

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


class Sm120GemmCooperative:
    """BF16 GEMM for SM120: cooperative TMA — no warp specialization.

    All warps do MMA compute. Thread 0 also issues TMA loads inline
    between K-tile boundaries (after consumer_release, before next
    consumer_wait). This eliminates the dedicated DMA warp, giving
    all warps to MMA and improving effective occupancy.
    """

    def __init__(
        self,
        bM: int = 128,
        bN: int = 128,
        bK: int = 64,
        num_mma_warps: int = 4,
        num_stages: int = 3,
        output_bf16: bool = False,
        max_active: int = 188,
    ):
        self.bM = bM
        self.bN = bN
        self.bK = bK
        self.num_mma_warps = num_mma_warps
        self.num_threads = num_mma_warps * 32  # all warps do MMA
        self.num_stages = num_stages
        self.dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.output_bf16 = output_bf16
        self.tile_shape_mnk = (bM, bN, bK)
        _atom_layouts = {2: (2, 1, 1), 4: (2, 2, 1), 6: (3, 2, 1), 8: (4, 2, 1)}
        self.atom_layout = _atom_layouts.get(num_mma_warps, (num_mma_warps, 1, 1))
        self.max_active = max_active

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

        mma_mnk = (16, 8, 16)
        op = warp.MmaF16BF16Op(a_dtype, self.acc_dtype, mma_mnk)
        atom_layout = self.atom_layout
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

        sA_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout_enum, tile_mnk, a_dtype, self.num_stages,
        )
        sB_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout_enum, tile_mnk, b_dtype, self.num_stages,
        )

        sA_layout = cute.slice_(sA_layout_staged, (None, None, 0))
        sB_layout = cute.slice_(sB_layout_staged, (None, None, 0))

        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_op, mA, sA_layout, (self.bM, self.bK), num_multicast=1,
        )
        tma_op_b = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            tma_op_b, mB, sB_layout, (self.bN, self.bK), num_multicast=1,
        )

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

        num_ctas_mnl = (M // self.bM, N // self.bN, 1)
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl,
        )
        max_active = cutlass.const_expr(self.max_active)
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active,
        )

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

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self._shared_storage)
        sA = storage.sa.get_tensor(
            sA_layout_staged.outer, swizzle=sA_layout_staged.inner
        )
        sB = storage.sb.get_tensor(
            sB_layout_staged.outer, swizzle=sB_layout_staged.inner
        )

        sA_1stage = cute.slice_(sA_layout_staged, (None, None, 0))
        sB_1stage = cute.slice_(sB_layout_staged, (None, None, 0))
        tma_copy_bytes = (
            cute.size_in_bytes(self.dtype, sA_1stage)
            + cute.size_in_bytes(self.dtype, sB_1stage)
        )

        mainloop_pipeline_array_ptr = storage.mbar.data_ptr()

        # Same pipeline setup — thread 0 is producer, all warps are consumers
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

        gA_mk = cute.local_tile(
            mA_mk,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None),
        )
        gB_nk = cute.local_tile(
            mB_nk,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None),
        )

        k_tile_count = cute.size(gA_mk, mode=[3])

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

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])

        thr_s2r_A = smem_copy_A.get_slice(tidx)
        thr_s2r_B = smem_copy_B.get_slice(tidx)
        tCsA_copy_view = thr_s2r_A.partition_S(sA)
        tCrA_copy_view = thr_s2r_A.retile(tCrA)
        tCsB_copy_view = thr_s2r_B.partition_S(sB)
        tCrB_copy_view = thr_s2r_B.retile(tCrB)

        num_k_mma = cute.size(tCrA, mode=[2])

        acc_shape = thr_mma.partition_shape_C((self.bM, self.bN))
        acc = cute.make_fragment(acc_shape, self.acc_dtype)

        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, num_stages,
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_stages,
        )

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim(),
        )
        work_tile = tile_sched.initial_work_tile_info()

        pipeline.sync(barrier_id=1)

        # All warps get full register budget for MMA
        cute.arch.setmaxregister_increase(232)

        # ==============================================================
        # Unified loop: all warps do MMA, warp 0 also issues TMA
        # ==============================================================

        # First tile: warp 0 prelude before main loop
        if work_tile.is_valid_tile:
            tile_m, tile_n, _ = work_tile.tile_idx
            tAgA_mk = tAgA[(None, tile_m, None)]
            tBgB_nk = tBgB[(None, tile_n, None)]

            if warp_idx == 0:
                prod_state.reset_count()
                for s in range(0, num_stages - 1, 1, unroll=1):
                    if s < k_tile_count:
                        pipe.producer_acquire(prod_state)
                        cute.copy(
                            tma_atom_a,
                            tAgA_mk[(None, prod_state.count)],
                            tAsA[(None, prod_state.index)],
                            tma_bar_ptr=pipe.producer_get_barrier(prod_state),
                            mcast_mask=0,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_nk[(None, prod_state.count)],
                            tBsB[(None, prod_state.index)],
                            tma_bar_ptr=pipe.producer_get_barrier(prod_state),
                            mcast_mask=0,
                        )
                        pipe.producer_commit(prod_state)
                        prod_state.advance()

        pipeline.sync(barrier_id=2)

        while work_tile.is_valid_tile:
            tile_m, tile_n, _ = work_tile.tile_idx
            tAgA_mk = tAgA[(None, tile_m, None)]
            tBgB_nk = tBgB[(None, tile_n, None)]

            acc.fill(0.0)
            gC = cute.local_tile(mC, (self.bM, self.bN), (tile_m, tile_n))
            cons_state.reset_count()

            # Wait for first K tile (already loaded in prelude)
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

            # Main K-loop (all but last)
            for kt in range(0, k_tile_count - 1, 1, unroll=1):
                for k in cutlass.range_constexpr(num_k_mma):
                    k_next = 0 if k + 1 == num_k_mma else k + 1

                    if k == num_k_mma - 1:
                        pipe.consumer_release(cons_state)
                        cons_state.advance()

                        # Warp 0: issue TMA for future stage
                        if warp_idx == 0:
                            if prod_state.count < k_tile_count:
                                pipe.producer_acquire(prod_state)
                                cute.copy(
                                    tma_atom_a,
                                    tAgA_mk[(None, prod_state.count)],
                                    tAsA[(None, prod_state.index)],
                                    tma_bar_ptr=pipe.producer_get_barrier(
                                        prod_state
                                    ),
                                    mcast_mask=0,
                                )
                                cute.copy(
                                    tma_atom_b,
                                    tBgB_nk[(None, prod_state.count)],
                                    tBsB[(None, prod_state.index)],
                                    tma_bar_ptr=pipe.producer_get_barrier(
                                        prod_state
                                    ),
                                    mcast_mask=0,
                                )
                                pipe.producer_commit(prod_state)
                                prod_state.advance()

                        peek_status = cutlass.Boolean(1)
                        peek_status = pipe.consumer_try_wait(cons_state)
                        tCsA_p = tCsA_copy_view[
                            None, None, None, cons_state.index
                        ]
                        tCsB_p = tCsB_copy_view[
                            None, None, None, cons_state.index
                        ]
                        pipe.consumer_wait(cons_state, peek_status)

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

            # Last K tile
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

            # Store accumulator to global memory
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

            # Warp 0: prelude for NEXT tile (while other warps idle after store)
            if warp_idx == 0:
                if work_tile.is_valid_tile:
                    tile_m_next, tile_n_next, _ = work_tile.tile_idx
                    tAgA_mk_next = tAgA[(None, tile_m_next, None)]
                    tBgB_nk_next = tBgB[(None, tile_n_next, None)]
                    prod_state.reset_count()
                    for s in range(0, num_stages - 1, 1, unroll=1):
                        if s < k_tile_count:
                            pipe.producer_acquire(prod_state)
                            cute.copy(
                                tma_atom_a,
                                tAgA_mk_next[(None, prod_state.count)],
                                tAsA[(None, prod_state.index)],
                                tma_bar_ptr=pipe.producer_get_barrier(
                                    prod_state
                                ),
                                mcast_mask=0,
                            )
                            cute.copy(
                                tma_atom_b,
                                tBgB_nk_next[(None, prod_state.count)],
                                tBsB[(None, prod_state.index)],
                                tma_bar_ptr=pipe.producer_get_barrier(
                                    prod_state
                                ),
                                mcast_mask=0,
                            )
                            pipe.producer_commit(prod_state)
                            prod_state.advance()


def test():
    """Verify correctness at 256x256 and benchmark larger sizes vs cuBLAS."""
    M, K, N = 256, 256, 256

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    # Zero-copy CuTe tensors
    a_cute = from_dlpack(A, assumed_align=16)  # (M, K) K-major
    b_cute = from_dlpack(B.t(), assumed_align=16)  # (N, K) N-major
    c_cute = from_dlpack(C, assumed_align=16)  # (M, N) N-major

    gemm = Sm120Gemm(
        bM=128, bN=128, bK=64, num_mma_warps=4, num_stages=3, output_bf16=True,
    )
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
    diff = (C.float() - ref).abs().max().item()
    print(f"max_diff = {diff:.6f}")
    passed = diff < 1.0
    print(f"Correctness: {'PASSED' if passed else 'FAILED'}")

    if not passed:
        return False

    # Benchmark various sizes vs cuBLAS (each needs its own compilation)
    print(f"\n{'Size':>10s}  {'CuTe(ms)':>9s}  {'cuBLAS(ms)':>10s}  "
          f"{'CuTe TFLOPS':>11s}  {'cuBLAS TFLOPS':>13s}  {'Ratio':>6s}  {'Check':>5s}")
    print("-" * 78)

    for sz in [256, 512, 1024, 2048, 4096, 8192]:
        A2 = torch.randn(sz, sz, dtype=torch.bfloat16, device="cuda")
        B2 = torch.randn(sz, sz, dtype=torch.bfloat16, device="cuda")
        C2 = torch.zeros(sz, sz, dtype=torch.bfloat16, device="cuda")
        a2 = from_dlpack(A2, assumed_align=16)
        b2 = from_dlpack(B2.t(), assumed_align=16)
        c2 = from_dlpack(C2, assumed_align=16)

        gemm2 = Sm120Gemm(
            bM=128, bN=128, bK=64, num_mma_warps=4, num_stages=3,
            output_bf16=True,
        )
        compiled2 = cute.compile(gemm2, a2, b2, c2, stream=stream)

        # Warmup
        compiled2(a2, b2, c2, stream)
        torch.cuda.synchronize()

        # Timed runs — CuTe kernel
        n_iter = 20
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            compiled2(a2, b2, c2, stream)
        torch.cuda.synchronize()
        cute_ms = (time.time() - t0) / n_iter * 1000

        # Timed runs — cuBLAS (torch.mm)
        for _ in range(5):
            torch.mm(A2, B2)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            torch.mm(A2, B2)
        torch.cuda.synchronize()
        cublas_ms = (time.time() - t0) / n_iter * 1000

        ref2 = A2.float() @ B2.float()
        ok = torch.allclose(C2.float(), ref2, atol=1e-1, rtol=1e-2)
        diff2 = (C2.float() - ref2).abs().max().item()

        flop = 2 * sz * sz * sz
        cute_tflops = flop / (cute_ms / 1000) / 1e12
        cublas_tflops = flop / (cublas_ms / 1000) / 1e12
        ratio = cute_tflops / cublas_tflops

        print(
            f"{sz:5d}x{sz:<5d}  {cute_ms:9.3f}  {cublas_ms:10.3f}  "
            f"{cute_tflops:11.2f}  {cublas_tflops:13.2f}  {ratio:5.0%}  "
            f"{'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            passed = False

    return passed


if __name__ == "__main__":
    import sys

    sys.exit(0 if test() else 1)
