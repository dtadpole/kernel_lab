# FA4 Pipeline Optimization — K Prefetch + V Double-Buffer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 5-13% performance gap between the generated FA4 kernel and FA4 CuTe DSL reference on RTX PRO 6000 Blackwell (503.8 TFLOPS BF16 peak).

**Architecture:** Two targeted changes to `conf/fixtures/fa4/generated.cu`: (1) Add K fragment prefetching in the QK inner loop to hide ldmatrix latency behind MMA execution, and (2) double-buffer V in shared memory to eliminate DMA/MMA serialization on V tiles. Both changes preserve the existing warp-specialized architecture (1 DMA + 4 MMA warps).

**Tech Stack:** CUDA C++ (SM120), mma.sync.m16n8k16, cp.async, named barriers, eval harness via Makefile

**Baseline (RTX PRO 6000, current generated.cu):**
| Config | Ref ms | Gen ms | Ref TFLOPS | Gen TFLOPS | Gen % peak |
|---|---|---|---|---|---|
| causal b8-s4096 | 1.649 | 1.730 | 333.4 | 317.8 | 63.1% |
| causal b4-s8192 | 3.088 | 3.274 | 356.1 | 335.8 | 66.7% |
| causal b2-s16384 | 5.953 | 6.371 | 369.4 | 345.2 | 68.5% |
| causal b1-s32768 | 11.962 | 12.581 | 367.7 | 349.6 | 69.4% |
| noncausal b8-s4096 | 2.646 | 3.028 | 415.5 | 363.2 | 72.1% |
| noncausal b4-s8192 | 5.179 | 5.907 | 424.6 | 372.3 | 73.9% |
| noncausal b2-s16384 | 10.405 | 11.802 | 422.7 | 372.6 | 74.0% |
| noncausal b1-s32768 | 21.044 | 24.321 | 418.0 | 361.7 | 71.8% |

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `conf/fixtures/fa4/generated.cu` | Modify | The generated FA4 kernel — all changes go here |

No new files. No test files (correctness verified via `make evaluate`; the eval harness mismatch between arange/random inputs makes `passed=False` expected — timing is the metric).

---

### Task 1: Add K fragment prefetching in QK inner loop

**Files:**
- Modify: `conf/fixtures/fa4/generated.cu:324-333` (the `mma_id_kv` inner loop inside `mma_warp_fn`)

**Problem:** In the QK computation, each K fragment is loaded via `ldmatrix_x2` and immediately consumed by `mma_m16n8k16`. The data dependency means the MMA instruction stalls ~30 cycles waiting for each K load to complete. With 8 K loads per d-step, this wastes ~240 cycles per d-step.

**Fix:** Prefetch K for the next `mma_id_kv` step while the current MMA executes, using the same software-pipelining pattern already used for Q (across d-steps) and V (across d-steps in PV loop).

- [ ] **Step 1: Replace the synchronous K load with a prefetch pipeline**

In `conf/fixtures/fa4/generated.cu`, replace the `mma_id_kv` inner loop (inside the `mma_id_d` loop, inside the `mma_id_q` loop in `mma_warp_fn`). The current code at lines 323-333:

```cpp
                #pragma unroll
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                    uint32_t K_frag[2];
                    {
                        uint32_t kaddr = K_cur;
                        kaddr += mma_id_kv * MMA_N * DIM * sizeof(nv_bfloat16);
                        kaddr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
                        ldmatrix_x2(K_frag, kaddr);
                    }
                    mma_m16n8k16(Q_cur, K_frag, S_local[mma_id_kv]);
                }
```

Replace with:

```cpp
                /* Prefetch first K fragment for this d-step */
                uint32_t K_cur_frag[2];
                {
                    uint32_t kaddr = K_cur;
                    kaddr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
                    ldmatrix_x2(K_cur_frag, kaddr);
                }

                #pragma unroll
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                    /* Prefetch K for next kv step while current MMA runs */
                    uint32_t K_next_frag[2];
                    if (mma_id_kv + 1 < BLOCK_KV / MMA_N) {
                        uint32_t kaddr = K_cur;
                        kaddr += (mma_id_kv + 1) * MMA_N * DIM * sizeof(nv_bfloat16);
                        kaddr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
                        ldmatrix_x2(K_next_frag, kaddr);
                    }
                    mma_m16n8k16(Q_cur, K_cur_frag, S_local[mma_id_kv]);
                    /* Rotate: next becomes current */
                    if (mma_id_kv + 1 < BLOCK_KV / MMA_N) {
                        K_cur_frag[0] = K_next_frag[0];
                        K_cur_frag[1] = K_next_frag[1];
                    }
                }
```

- [ ] **Step 2: Compile and verify no errors**

```bash
cd /home/centos/kernel_lab/cuda_exec/scripts
make clean compile KERNEL=fa4
```

Expected: compile succeeds. ptxas may report 255 registers (unchanged) or slightly more spills. The key metric `Used N registers` should be <= 255. Minor spill increase is acceptable.

- [ ] **Step 3: Run one quick config to verify timing**

```bash
make evaluate KERNEL=fa4 CONFIG=mha-causal-b2-s16384
```

Expected: `generated median` should be lower than baseline 6.37ms. Even a small improvement (6.0-6.3ms) confirms the prefetch is helping.

- [ ] **Step 4: Commit**

```bash
cd /home/centos/kernel_lab
git add conf/fixtures/fa4/generated.cu
git commit -m "perf(fa4): add K fragment prefetching in QK inner loop

Prefetch the next K fragment via ldmatrix while the current mma.sync
executes, hiding ~30 cycles of SMEM read latency per K load.
Same software-pipelining pattern already used for Q and V fragments."
```

---

### Task 2: Double-buffer V in shared memory

**Files:**
- Modify: `conf/fixtures/fa4/generated.cu` — barrier constants, `dma_warp_fn`, `mma_warp_fn`, `flash_attention_kernel_ws` SMEM layout

**Problem:** V uses a single SMEM buffer. DMA must wait for MMA to fully consume V before loading the next V tile (`BAR_V_EMPTY` stall). Double-buffering allows DMA to load the next V into an alternate slot while MMA processes the current V.

**SMEM budget:** 80KB → 96KB (Q:32KB + K:32KB + V:32KB). SM120 supports up to 228KB per block.

- [ ] **Step 1: Add V double-buffer barrier constants**

In `conf/fixtures/fa4/generated.cu`, update the barrier ID section. Replace:

```cpp
static constexpr int BAR_V_FULL  = 3;  /* DMA signals V ready       */
static constexpr int BAR_V_EMPTY = 4;  /* MMA signals V consumed    */
```

With:

```cpp
static constexpr int BAR_V_FULL  = 3;  /* DMA signals V slot 0 ready    */
static constexpr int BAR_V_EMPTY = 4;  /* MMA signals V slot 0 consumed */
static constexpr int BAR_V_FULL1 = 5;  /* DMA signals V slot 1 ready    */
static constexpr int BAR_V_EMPTY1= 6;  /* MMA signals V slot 1 consumed */
```

- [ ] **Step 2: Update dma_warp_fn to double-buffer V**

Replace the V loading section inside the `kv_id` loop in `dma_warp_fn`. Current code:

```cpp
        /* Wait until MMA warps have consumed the previous V.
         * Skip on first iteration — no previous tile to protect. */
        if (kv_id > 0) bar_sync(BAR_V_EMPTY, BAR_THREADS);

        /* Load V[kv_id] into single-buffered V_smem */
        global_to_shared_swizzle<BLOCK_KV, DIM, DMA_THREADS>(
            V_smem, V_ptr, seq_stride, tid);
        V_ptr += BLOCK_KV * seq_stride;

        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");

        /* Signal V is ready */
        bar_arrive(BAR_V_FULL, BAR_THREADS);
```

Replace with:

```cpp
        /* Double-buffered V: select slot-specific barriers */
        const int v_full_bar  = (kv_id % 2 == 0) ? BAR_V_FULL  : BAR_V_FULL1;
        const int v_empty_bar = (kv_id % 2 == 0) ? BAR_V_EMPTY : BAR_V_EMPTY1;

        /* Wait until MMA consumed the V in THIS slot (2 iters ago).
         * Skip first two iterations — each slot is written for the first time. */
        if (kv_id >= 2) bar_sync(v_empty_bar, BAR_THREADS);

        /* Load V[kv_id] into double-buffered V_smem slot */
        const uint32_t V_dst = V_smem +
            (kv_id % 2) * (BLOCK_KV * DIM * (int)sizeof(nv_bfloat16));
        global_to_shared_swizzle<BLOCK_KV, DIM, DMA_THREADS>(
            V_dst, V_ptr, seq_stride, tid);
        V_ptr += BLOCK_KV * seq_stride;

        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");

        /* Signal V slot is ready */
        bar_arrive(v_full_bar, BAR_THREADS);
```

- [ ] **Step 3: Update mma_warp_fn to read from double-buffered V**

In `mma_warp_fn`, update the PV computation section. Replace the V barrier wait and V_smem_thread usage. Find the code block:

```cpp
        /* Wait for DMA to signal V is ready */
        bar_sync(BAR_V_FULL, BAR_THREADS);
```

Replace with:

```cpp
        /* Wait for DMA to signal V slot is ready */
        const int v_full_bar  = (kv_id % 2 == 0) ? BAR_V_FULL  : BAR_V_FULL1;
        bar_sync(v_full_bar, BAR_THREADS);
```

Then find the V_smem_thread usage in the PV prefetch section. Update the initial V prefetch address:

```cpp
            /* Prefetch V for the very first step (kv=0, d=0) */
            uint32_t V_cur[2];
            {
                uint32_t addr = V_smem_thread;
                ldmatrix_x2_trans(V_cur, addr);
            }
```

Replace with:

```cpp
            /* Prefetch V for the very first step (kv=0, d=0) */
            const uint32_t V_smem_cur = V_smem_thread +
                (kv_id % 2) * (BLOCK_KV * DIM * (int)sizeof(nv_bfloat16));
            uint32_t V_cur[2];
            {
                ldmatrix_x2_trans(V_cur, V_smem_cur);
            }
```

And update all V address computations inside the PV nested loop to use `V_smem_cur` instead of `V_smem_thread`:

```cpp
                    if (has_next_d) {
                        uint32_t addr = V_smem_cur;
                        addr += mma_id_kv * MMA_K * DIM * sizeof(nv_bfloat16);
                        addr ^= (mma_id_d + 1) * MMA_N * sizeof(nv_bfloat16);
                        ldmatrix_x2_trans(V_next, addr);
                    } else if (has_next_kv) {
                        uint32_t addr = V_smem_cur;
                        addr += (mma_id_kv + 1) * MMA_K * DIM * sizeof(nv_bfloat16);
                        ldmatrix_x2_trans(V_next, addr);
                    }
```

Finally, replace the V_EMPTY signal at the end:

```cpp
        /* Signal DMA that V buffer is free for next load */
        bar_arrive(BAR_V_EMPTY, BAR_THREADS);
```

With:

```cpp
        /* Signal DMA that V slot is consumed */
        const int v_empty_bar = (kv_id % 2 == 0) ? BAR_V_EMPTY : BAR_V_EMPTY1;
        bar_arrive(v_empty_bar, BAR_THREADS);
```

- [ ] **Step 4: Update SMEM layout in kernel launch**

In `flash_attention_kernel_ws`, update the V_smem allocation:

```cpp
    const uint32_t V_smem  = KV_base + 2 * BLOCK_KV * DIM * sizeof(nv_bfloat16);
```

This stays the same (V starts after K's double-buffer region). But update `smem_v` in `kernel_run`:

```cpp
        int smem_v  = BLOCK_KV * DIM_CONST * (int)sizeof(nv_bfloat16);
```

Replace with:

```cpp
        int smem_v  = 2 * BLOCK_KV * DIM_CONST * (int)sizeof(nv_bfloat16);
```

Also update the SMEM layout comment in `flash_attention_kernel_ws`:

```cpp
    /* Shared memory layout:
     *   Q region:  [BLOCK_Q, DIM] = 32KB (persistent)
     *   K region:  [2, BLOCK_KV, DIM] = 32KB (double-buffered)
     *   V region:  [2, BLOCK_KV, DIM] = 32KB (double-buffered)
     *   Total: 32KB + 32KB + 32KB = 96KB
     */
```

- [ ] **Step 5: Compile and verify**

```bash
cd /home/centos/kernel_lab/cuda_exec/scripts
make clean compile KERNEL=fa4
```

Expected: compile succeeds. ptxas should show `used 7 barriers` (was 5). SMEM usage goes from 80KB to 96KB.

- [ ] **Step 6: Quick eval to verify correctness and performance**

```bash
make evaluate KERNEL=fa4 CONFIG=mha-causal-b2-s16384
```

Expected: timing should show improvement vs baseline (6.37ms → hopefully <6.2ms).

- [ ] **Step 7: Commit**

```bash
cd /home/centos/kernel_lab
git add conf/fixtures/fa4/generated.cu
git commit -m "perf(fa4): double-buffer V in shared memory

Add a second V slot in SMEM (80KB → 96KB) so DMA can load the next
V tile while MMA processes the current one. Uses per-slot named
barriers (BAR_V_FULL/EMPTY + BAR_V_FULL1/EMPTY1) to protect each slot
independently."
```

---

### Task 3: Full evaluation — all 8 configs

**Files:** None modified (evaluation only)

- [ ] **Step 1: Run all 8 configs**

```bash
cd /home/centos/kernel_lab/cuda_exec/scripts
make clean compile KERNEL=fa4
make evaluate KERNEL=fa4 CONFIG=all
```

- [ ] **Step 2: Compute TFLOPS and compare against baseline**

Use the same TFLOPS computation as the baseline table. Report:
- New median latency for each config
- New TFLOPS
- % of peak (503.8)
- Delta vs baseline (improvement or regression)
- Delta vs reference

- [ ] **Step 3: Analyze results and identify remaining bottlenecks**

If performance improved but gap remains:
- Consider loop restructuring (load K once for both mma_id_q)
- Consider BLOCK_KV increase to 128
- Profile with NCU if available

If performance regressed:
- Check ptxas register/spill report
- Revert problematic change
- Try alternative approach
