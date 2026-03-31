# FA4 Inline Warp Specialization + DMA V-Overlap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate 16.9M local memory spill requests from `__noinline__` ABI overhead and overlap V loading with QK MMA computation, targeting ~95% of FA4 CuTe DSL performance.

**Architecture:** Inline DMA/MMA warp logic back into the kernel body (removing `__noinline__` function calls) with `if (warp_id == 0) / else` branching. Double-buffer V SMEM (2 × 16KB) so DMA warp can load V[n] while MMA warps compute QK on K[n], eliminating the DMA idle gap. Remove BAR_V_EMPTY barrier (no longer needed with double-buffered V).

**Tech Stack:** CUDA C++ (SM120), `bar.sync`/`bar.arrive`, `cp.async.cg`, `mma.sync.m16n8k16`, `ldmatrix`, `exp2f`

---

## File

- **Modify:** `conf/fixtures/fa4/generated.cu` — single file, kernel rewrite

## SMEM Budget Verification

- Q: 128 × 128 × 2 = 32,768 bytes
- K: 2 × 64 × 128 × 2 = 32,768 bytes (double-buffered, unchanged)
- V: 2 × 64 × 128 × 2 = 32,768 bytes (double-buffered, NEW — was single)
- **Total: 98,304 bytes (96KB)** — fits SM120's 99KB limit

---

### Task 1: Inline DMA/MMA + double-buffer V + remove BAR_V_EMPTY

**Files:**
- Modify: `conf/fixtures/fa4/generated.cu`

- [ ] **Step 1: Read the current kernel**

Read `/home/centos/kernel_lab/.claude/worktrees/cuda-exec/conf/fixtures/fa4/generated.cu` completely. Note:
- `dma_warp_fn` (lines ~164-214) — `__noinline__`, loads K (double-buf) + V (single-buf)
- `mma_warp_fn` (lines ~222-510) — `__noinline__`, QK MMA + softmax + PV MMA
- `flash_attention_kernel_ws` (lines ~520-640) — calls dma/mma functions
- `kernel_run` (lines ~650+) — launcher with TB_SIZE=160

- [ ] **Step 2: Rewrite the kernel with inlined warp logic**

Replace `dma_warp_fn`, `mma_warp_fn`, and `flash_attention_kernel_ws` with a single kernel that has DMA/MMA logic inline. Key changes:

**a) Remove `__noinline__` functions entirely.** Move all their logic into the kernel body under `if (warp_id == 0) { /* DMA */ } else { /* MMA */ }`.

**b) Double-buffer V.** Change V_smem from 1 slot to 2 slots:
```cuda
const uint32_t V_smem = K_smem + 2 * BLOCK_KV * DIM * sizeof(nv_bfloat16);
/* V now has 2 slots: V_smem + (kv_id % 2) * BLOCK_KV * DIM * sizeof(...) */
```

**c) Remove BAR_V_EMPTY.** With double-buffered V, DMA warp never overwrites V that MMA is still reading. Only 3 barriers remain: BAR_K_FULL(1), BAR_K_EMPTY(2), BAR_V_FULL(3).

**d) DMA warp pipeline: load V immediately after signaling K_FULL.**
```
DMA loop:
  if (kv_id > 0) bar_sync(BAR_K_EMPTY, 160)  // wait MMA consumed K
  load K[kv_id] → K_smem[kv_id % 2]
  cp.async.wait_all
  bar_arrive(BAR_K_FULL, 160)                  // signal K ready
  // NO wait for V_EMPTY — double buffer protects us
  load V[kv_id] → V_smem[kv_id % 2]           // overlaps with MMA's QK compute
  cp.async.wait_all
  bar_arrive(BAR_V_FULL, 160)                  // signal V ready
```

**e) MMA warp loop uses V_smem[kv_id % 2]:**
```
MMA loop:
  bar_sync(BAR_K_FULL, 160)      // wait K ready
  ... QK MMA + softmax ...
  bar_arrive(BAR_K_EMPTY, 160)   // signal K consumed
  bar_sync(BAR_V_FULL, 160)      // wait V ready
  ... PV MMA from V_smem[kv_id % 2] ...
  // NO bar_arrive(BAR_V_EMPTY) — removed
```

**f) SMEM layout update in kernel_run:**
```cuda
int smem_q  = 128 * 128 * sizeof(nv_bfloat16);           // 32KB
int smem_kv = (2 + 2) * 64 * 128 * sizeof(nv_bfloat16);  // 64KB (K×2 + V×2)
int smem_size = smem_q + smem_kv;                          // 96KB
```

**g) Q loading uses 128 threads (tid < 128)**, same `global_to_shared_swizzle<128, 128, 128>` with guard. Then `__syncthreads()` with all 160.

- [ ] **Step 3: Verify compilation and register usage**

```bash
cd /home/centos/kernel_lab/.claude/worktrees/cuda-exec
/usr/local/cuda/bin/nvcc -arch=sm_120 -std=c++17 -O3 --resource-usage \
  -c conf/fixtures/fa4/generated.cu -o /dev/null
```

Expected: compiles, check registers ≤ 255 and spill bytes. Target: 0 spill or much less than 16.9M requests.

If spill is excessive (> 100 bytes), try:
- Add `__launch_bounds__(160, 1)` — tells compiler max 1 block/SM
- Reduce one register array (e.g., use S_local[8][4] per-mma_id_q instead of full S_rmem)

- [ ] **Step 4: Build and smoke test**

```bash
/usr/local/cuda/bin/nvcc -arch=native -std=c++17 -O3 -lineinfo \
  -I cuda_exec/scripts \
  cuda_exec/scripts/eval_harness.cu conf/fixtures/fa4/generated.cu \
  -o /tmp/fa4_bin/fa4.bin

timeout 15 bash -c '
CUDA_EXEC_PARAM_BATCH_SIZE=1 CUDA_EXEC_PARAM_SEQ_LEN=128 \
CUDA_EXEC_PARAM_NUM_HEADS=2 CUDA_EXEC_PARAM_HEAD_DIM=128 \
CUDA_EXEC_PARAM_CAUSAL=false CUDA_EXEC_PARAM_INPUT_SIZE=$((1*128*2*128)) \
CUDA_EXEC_HARNESS_NUM_INPUTS=3 CUDA_EXEC_HARNESS_NUM_OUTPUTS=1 \
CUDA_EXEC_NUM_WARMUPS=1 CUDA_EXEC_NUM_TRIALS=1 \
/tmp/fa4_bin/fa4.bin | head -3
'
```

Expected: JSON output within 15 seconds (no deadlock). Non-NaN values.

If deadlock: barrier arrive counts are wrong. Check that all 160 threads participate in each barrier.

- [ ] **Step 5: Commit**

```bash
git add conf/fixtures/fa4/generated.cu
git commit -m "perf: inline warp spec + V double-buffer (eliminate ABI spill, overlap V load)"
```

---

### Task 2: Benchmark and validate

- [ ] **Step 1: Run relative benchmark**

Run the 4-config relative benchmark (FA4 vs Generated) with the new binary at `/tmp/fa4_bin/fa4.bin`.

```python
# Use the existing /tmp/bench_fa4_all.py or equivalent
# Compare Gen/FA4 ratio — target > 0.92 on all configs
```

- [ ] **Step 2: NCU profile**

```bash
sudo --preserve-env /usr/local/cuda/bin/ncu --set detailed \
  --kernel-name regex:"flash_attention" --launch-count 1 \
  ... python ncu script ...
```

Check:
- Tensor pipe utilization: target > 88% (was 87.5%)
- Local memory spill: target < 1M (was 16.9M)
- Duration: should decrease from 24.1ms

- [ ] **Step 3: Commit + push + merge**

```bash
git push origin cuda-exec
cd /home/centos/kernel_lab
git checkout main && git pull origin main
git merge cuda-exec --no-ff -m "Merge: FA4 inline warp spec + V overlap"
git push origin main
```

---

## Verification Checklist

- [ ] Compiles with `nvcc -arch=sm_120 -std=c++17 -O3`
- [ ] No deadlock on small config (timeout test)
- [ ] Register spill significantly reduced (target: < 100 bytes)
- [ ] Gen/FA4 ratio improved vs previous (was 0.88-0.96x)
- [ ] Tensor pipe utilization ≥ 87.5%
