# CuTe DSL SM120 GEMM — Zero-Copy Row-Major Inputs

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write a CuTe DSL BF16 GEMM kernel for SM120 that accepts row-major PyTorch tensors directly — no copy, no transpose.

**Architecture:** Build incrementally in 4 stages, each producing a working, testable kernel:
1. Simplest possible CuTe DSL GEMM (cp.async, no TMA, no persistence) — verify correctness
2. Add multi-stage pipeline for latency hiding
3. Add TMA for efficient global→SMEM loads
4. Add persistent scheduling for full SM utilization

Each stage is a self-contained kernel that passes correctness tests. This avoids the MLIR compilation hang that occurred when all features were combined at once.

**Tech Stack:** CuTe DSL (`cutlass.cute`), `from_dlpack` for zero-copy, `MmaF16BF16Op` + `LdMatrix8x8x16bOp` for SM120 mma.sync, `cp.async` (stage 1-2) then TMA (stage 3-4).

**Key convention:**
- A: `from_dlpack(A_pytorch)` → (M, K), K-major (strides K,1) → `LayoutEnum.ROW_MAJOR`
- B: `from_dlpack(B_pytorch.t())` → (N, K), N-major (strides 1,N) → `LayoutEnum.COL_MAJOR` — zero-copy view
- C: `from_dlpack(C_pytorch)` → (M, N), N-major (strides N,1) → `LayoutEnum.ROW_MAJOR`

**Reference code:** simveit's Hopper persistent GEMM (fetched to `/tmp/simveit_gemm.py`) and CUTLASS Ampere SGEMM example.

---

### Task 1: Minimal CuTe DSL GEMM — cp.async + mma.sync (no TMA, no persistence)

**Files:**
- Create: `conf/fixtures/matmul/cute_gemm.py`

The simplest possible kernel:
- Single tile per block (grid = M/bM × N/bN)
- cp.async for global→SMEM (no TMA)
- ldmatrix for SMEM→register
- mma.sync via `cute.gemm(tiled_mma, ...)`
- All threads participate (no warp specialization)
- Single pipeline stage (load → sync → compute → repeat)

- [ ] **Step 1: Write minimal kernel class with test**

Write `conf/fixtures/matmul/cute_gemm.py` with:

```python
"""Minimal CuTe DSL BF16 GEMM for SM120.

Stage 1: cp.async + ldmatrix + mma.sync. No TMA, no persistence.
Zero-copy row-major inputs via from_dlpack.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cute.runtime import from_dlpack


class Sm120Gemm:
    def __init__(self):
        self.bM = 128
        self.bN = 128
        self.bK = 32  # smaller K tile for simplicity
        self.acc_dtype = cutlass.Float32
        self.atom_layout = (2, 2, 1)
        self.num_mma_warps = 4
        self.mma_inst_mnk = (16, 8, 16)
        self.num_threads = self.num_mma_warps * 32  # 128 threads, no DMA warp

    @cute.jit
    def __call__(self, mA, mB, mC, stream):
        a_dtype = mA.element_type
        b_dtype = mB.element_type
        a_layout_enum = utils.LayoutEnum.from_tensor(mA)
        b_layout_enum = utils.LayoutEnum.from_tensor(mB)

        # MMA
        op = cute.nvgpu.warp.MmaF16BF16Op(a_dtype, self.acc_dtype, self.mma_inst_mnk)
        tC = cute.make_layout(self.atom_layout)
        perm = (
            self.atom_layout[0] * self.mma_inst_mnk[0],
            self.atom_layout[1] * self.mma_inst_mnk[1] * 2,
            self.atom_layout[2] * self.mma_inst_mnk[2],
        )
        tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=perm)

        # SMEM layouts (single stage, no staging dimension needed)
        a_smem_layout = sm90_utils.make_smem_layout_a(
            a_layout_enum, (self.bM, self.bN, self.bK), a_dtype, 1,
        )
        # For single stage, slice out the stage dimension
        a_smem = cute.slice_(a_smem_layout, (None, None, 0))

        b_smem_layout = sm90_utils.make_smem_layout_b(
            b_layout_enum, (self.bM, self.bN, self.bK), b_dtype, 1,
        )
        b_smem = cute.slice_(b_smem_layout, (None, None, 0))

        # Copy atoms: cp.async global→shared
        # Thread layout for coalesced access
        copy_atom_a = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), a_dtype,
            num_bits_per_copy=128,  # 16-byte vectorized
        )
        copy_atom_b = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), b_dtype,
            num_bits_per_copy=128,
        )

        # Thread layouts for global→shared copy (distribute threads for coalescing)
        is_a_k_major = a_layout_enum.is_k_major_a()
        if is_a_k_major:
            # K contiguous: threads spread along M, vectorize along K
            thr_layout_a = cute.make_layout(
                (self.num_threads // (self.bK // 8), self.bK // 8),
                stride=(self.bK // 8, 1),
            )
            val_layout_a = cute.make_layout((1, 8), stride=(0, 1))
        else:
            # M contiguous: threads spread along K, vectorize along M
            thr_layout_a = cute.make_layout(
                (self.bM // 8, self.num_threads // (self.bM // 8)),
                stride=(1, self.bM // 8),
            )
            val_layout_a = cute.make_layout((8, 1), stride=(1, 0))

        tiled_copy_a = cute.make_tiled_copy_tv(copy_atom_a, thr_layout_a, val_layout_a)

        is_b_n_major = b_layout_enum.is_n_major_b()
        if is_b_n_major:
            # N contiguous: threads spread along K, vectorize along N
            thr_layout_b = cute.make_layout(
                (self.bN // 8, self.num_threads // (self.bN // 8)),
                stride=(1, self.bN // 8),
            )
            val_layout_b = cute.make_layout((8, 1), stride=(1, 0))
        else:
            # K contiguous: threads spread along N, vectorize along K
            thr_layout_b = cute.make_layout(
                (self.num_threads // (self.bK // 8), self.bK // 8),
                stride=(self.bK // 8, 1),
            )
            val_layout_b = cute.make_layout((1, 8), stride=(0, 1))

        tiled_copy_b = cute.make_tiled_copy_tv(copy_atom_b, thr_layout_b, val_layout_b)

        # Grid
        M = mA.layout.shape[0]
        N = mB.layout.shape[0]
        grid = ((M + self.bM - 1) // self.bM, (N + self.bN - 1) // self.bN, 1)

        # Shared storage
        @cute.struct
        class Storage:
            sa: cute.struct.Align[
                cute.struct.MemRange[a_dtype, cute.cosize(a_smem)], 128
            ]
            sb: cute.struct.Align[
                cute.struct.MemRange[b_dtype, cute.cosize(b_smem)], 128
            ]
        self._storage = Storage

        self.kernel(
            mA, mB, mC, tiled_mma,
            a_smem, b_smem,
            tiled_copy_a, tiled_copy_b,
            a_layout_enum.is_m_major_a(),
            b_layout_enum.is_n_major_b(),
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            smem=self._storage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mA, mB, mC, tiled_mma,
               a_smem_layout, b_smem_layout,
               tiled_copy_a, tiled_copy_b,
               a_is_m_major, b_is_n_major):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self._storage)

        sA = storage.sa.get_tensor(a_smem_layout)  # no swizzle for simplicity first
        sB = storage.sb.get_tensor(b_smem_layout)

        # Global tile for this block
        gA = cute.local_tile(mA, (self.bM, self.bK), (bidy, None))
        gB = cute.local_tile(mB, (self.bN, self.bK), (bidx, None))
        gC = cute.local_tile(mC, (self.bM, self.bN), (bidy, bidx))

        # Global→SMEM copy partitions
        thr_copy_a = tiled_copy_a.get_slice(tidx)
        tAgA = thr_copy_a.partition_S(gA)
        tAsA = thr_copy_a.partition_D(sA)

        thr_copy_b = tiled_copy_b.get_slice(tidx)
        tBgB = thr_copy_b.partition_S(gB)
        tBsB = thr_copy_b.partition_D(sB)

        # MMA partitions
        thr_mma = tiled_mma.get_slice(tidx)
        tCgC = thr_mma.partition_C(gC)

        # ldmatrix atoms
        atom_ldm_a = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(a_is_m_major, 4), cutlass.BFloat16,
        )
        atom_ldm_b = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(b_is_n_major, 4), cutlass.BFloat16,
        )
        smem_copy_a = cute.make_tiled_copy_A(atom_ldm_a, tiled_mma)
        smem_copy_b = cute.make_tiled_copy_B(atom_ldm_b, tiled_mma)

        # Register fragments
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))

        thr_ldm_a = smem_copy_a.get_slice(tidx)
        tCsA = thr_ldm_a.partition_S(sA)
        tCrA_view = thr_ldm_a.retile(tCrA)

        thr_ldm_b = smem_copy_b.get_slice(tidx)
        tCsB = thr_ldm_b.partition_S(sB)
        tCrB_view = thr_ldm_b.retile(tCrB)

        # Accumulator
        accum = cute.make_fragment(tCgC.shape, self.acc_dtype)

        k_tiles = cute.size(gA, mode=[1])  # number of K tiles
        num_k_blocks = cute.size(tCrA, mode=[2])

        # K-loop
        for kt in cutlass.range(k_tiles):
            # Load A tile: global → SMEM
            cute.copy(tiled_copy_a, tAgA[None, kt], tAsA)
            # Load B tile: global → SMEM
            cute.copy(tiled_copy_b, tBgB[None, kt], tBsB)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # SMEM → registers → MMA
            for kb in cutlass.range(num_k_blocks, unroll_full=True):
                cute.copy(smem_copy_a, tCsA[None, None, kb], tCrA_view[None, None, kb])
                cute.copy(smem_copy_b, tCsB[None, None, kb], tCrB_view[None, None, kb])
                cute.gemm(tiled_mma, accum, tCrA[None, None, kb], tCrB[None, None, kb], accum)

            cute.arch.sync_threads()

        # Store result
        store_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype)
        cute.copy(store_atom, accum, tCgC)
```

Test function:

```python
def test(M=256, K=256, N=256):
    A = torch.arange(M*K, dtype=torch.bfloat16, device="cuda").reshape(M, K).contiguous()
    B = torch.arange(K*N, dtype=torch.bfloat16, device="cuda").reshape(K, N).contiguous()
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    a_cute = from_dlpack(A, assumed_align=16)
    b_cute = from_dlpack(B.t(), assumed_align=16)  # zero-copy!
    c_cute = from_dlpack(C, assumed_align=16)

    gemm = Sm120Gemm()
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(gemm, a_cute, b_cute, c_cute, stream=stream)
    compiled(a_cute, b_cute, c_cute, stream)
    torch.cuda.synchronize()

    ref = A.float() @ B.float()
    diff = (C - ref).abs().max().item()
    print(f"max_diff = {diff:.6f}")
    return diff < 1.0
```

- [ ] **Step 2: Run test**

Run: `cd /home/centos/kernel_lab && timeout 600 cuda_exec/.venv/bin/python conf/fixtures/matmul/cute_gemm.py`

If MLIR hangs (>5 min), simplify: reduce tile to 64×64×32, reduce atom_layout to (1,1,1), remove swizzle.

- [ ] **Step 3: Debug and iterate until correctness passes**

Common issues:
- MLIR hang → reduce kernel complexity (fewer MMA warps, smaller tiles)
- Wrong results → check ldmatrix transpose flag, SMEM layout atom, copy thread layout
- Crash → check SMEM size, alignment

- [ ] **Step 4: Commit**

```bash
git add conf/fixtures/matmul/cute_gemm.py
git commit -m "feat: minimal CuTe DSL GEMM — cp.async + mma.sync, zero-copy inputs"
```

---

### Task 2: Add multi-stage pipeline

**Files:**
- Modify: `conf/fixtures/matmul/cute_gemm.py`

Add 2-stage or 3-stage pipeline to overlap global loads with compute:
- Stage SMEM layouts (add stage dimension)
- cp.async.commit_group / wait_group for pipeline control
- Preload first stage, compute from previous stage while loading next

- [ ] **Step 1: Add staging to SMEM layouts**

Change `make_smem_layout_a/b` from `stage=1` to `stage=2`.

- [ ] **Step 2: Restructure K-loop for pipeline overlap**

Prelude: load first stage. Main loop: compute from stage N while loading stage N+1.

- [ ] **Step 3: Test correctness**

Same test as Task 1.

- [ ] **Step 4: Benchmark vs generated kernel**

Compare TFLOPS at 4096×4096.

- [ ] **Step 5: Commit**

---

### Task 3: Replace cp.async with TMA

**Files:**
- Modify: `conf/fixtures/matmul/cute_gemm.py`

TMA advantages:
- Hardware address computation (saves register pressure)
- Handles arbitrary strides natively (key for row-major A and col-major B)
- Uses TMA barrier for synchronization

Changes:
- Add `CopyBulkTensorTileG2SOp` for TMA load
- Add `make_tiled_tma_atom` for TMA descriptor creation
- Add warp specialization: DMA warp (1 warp) + MMA warps (4 warps)
- Add TMA pipeline barriers

- [ ] **Step 1: Add TMA atoms in __call__**
- [ ] **Step 2: Restructure kernel for warp specialization**
- [ ] **Step 3: Test correctness**
- [ ] **Step 4: Benchmark improvement from TMA**
- [ ] **Step 5: Commit**

---

### Task 4: Add persistent scheduling

**Files:**
- Modify: `conf/fixtures/matmul/cute_gemm.py`

Change from one-tile-per-block to persistent:
- Use `StaticPersistentTileScheduler`
- Each block loops over multiple tiles
- Grid size = number of SMs (170)

- [ ] **Step 1: Add persistent scheduler**
- [ ] **Step 2: Test correctness at all matrix sizes**
- [ ] **Step 3: Benchmark vs reference and generated**
- [ ] **Step 4: Commit**

---

### Task 5: Integrate into reference.py

**Files:**
- Modify: `conf/fixtures/matmul/reference.py`

Replace the Sm120GemmKernel with our custom Sm120Gemm:
- Import from cute_gemm.py
- Update Model class to use new kernel
- Verify correctness and performance with evaluate harness

- [ ] **Step 1: Update Model to use Sm120Gemm**
- [ ] **Step 2: Run evaluate on all configs**
- [ ] **Step 3: Run NCU profile and compare TFLOPS**
- [ ] **Step 4: Commit and push**

---

## Verification

1. Each task: `test()` in cute_gemm.py passes (max_diff < 1.0)
2. Task 5: `make evaluate KERNEL=matmul CONFIG=mat-4096x4096` passes correctness
3. Final: TFLOPS comparison shows no regression vs current reference
4. Key metric: A and B copies = 0 (zero-copy from PyTorch)
