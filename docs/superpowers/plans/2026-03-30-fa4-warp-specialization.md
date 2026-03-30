# FA4 Warp-Specialized Flash Attention Kernel

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the FA4 generated CUDA kernel with warp specialization to close the 15-20% gap with the FA4 CuTe DSL reference (current: 177-193 TFLOPS, target: 210+ TFLOPS on RTX 5090).

**Architecture:** Split 5 warps (160 threads) into 1 DMA warp (warp 0, loads K/V from GMEM→SMEM via cp.async) and 4 MMA warps (warps 1-4, compute QK/softmax/PV via mma.sync). DMA and MMA warps run concurrently with named barrier (`bar.sync/bar.arrive`) synchronization instead of `__syncthreads()`. DMA warp uses ~40 registers; MMA warps use ~200+ registers. This eliminates the fundamental bottleneck: all threads currently stop computing during loads and stop loading during compute.

**Tech Stack:** CUDA C++ (SM120), `mma.sync.aligned.m16n8k16`, `cp.async.cg`, `ldmatrix`, `bar.sync`/`bar.arrive` named barriers, `nv_bfloat16`.

---

## Background: Why Warp Specialization

Current kernel: all 4 warps (128 threads) do both loading and computing. Every `global_to_shared_swizzle` call issues cp.async from all threads, then `__syncthreads()` blocks everyone. Tensor pipe utilization: 77.4%. FA4 reference: 87.9%.

The `__syncthreads()` barrier means:
1. MMA warps idle while waiting for cp.async to complete
2. Load warps idle while MMA warps compute
3. No overlap between load and compute phases

With warp specialization:
- DMA warp continuously streams K/V tiles into SMEM
- MMA warps continuously compute on SMEM data
- Named barriers synchronize only the relevant warps
- Load and compute overlap fully

## Key Design Decisions

### Thread layout: 5 warps (160 threads)
- Warp 0 (tid 0-31): **DMA warp** — loads K/V from GMEM→SMEM via cp.async
- Warps 1-4 (tid 32-159): **MMA warps** — compute QK, softmax, PV via mma.sync
- Q is loaded once at startup by ALL threads cooperatively (before the warp split)

### SMEM layout (same 80KB total)
- Q region: 128 × 128 × 2 = 32KB (persistent, loaded once)
- K region: 2 × 64 × 128 × 2 = 32KB (double-buffered, DMA warp rotates)
- V region: 64 × 128 × 2 = 16KB (single-buffered, DMA warp fills)

### Synchronization via named barriers
- **Barrier 1 (K_FULL):** DMA warp signals K tile is ready; MMA warps wait
- **Barrier 2 (K_EMPTY):** MMA warps signal K tile consumed; DMA warp waits before overwriting
- **Barrier 3 (V_FULL):** DMA warp signals V tile is ready; MMA warps wait
- **Barrier 4 (V_EMPTY):** MMA warps signal V tile consumed; DMA warp waits

Each barrier: `bar.arrive` from producer side, `bar.sync` from consumer side.
- K_FULL arrive count: 32 (DMA warp) + 128 (MMA warps) = 160
- K_EMPTY arrive count: 160

### Register allocation
- DMA warp: ~40 registers (only needs loop counters + cp.async addresses)
- MMA warps: ~220+ registers (S_local, P_rmem, O_rmem, rowmax, rowsumexp)
- Use `asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS))` if available on SM120, otherwise let compiler decide per warp role via function splitting

### Pipeline flow per KV iteration

```
DMA warp:                          MMA warps:
  bar.sync(K_EMPTY)                  (computing previous PV MMA)
  cp.async K[n] → SMEM               ...
  cp.async.commit_group               ...
  cp.async.wait_all                    ...
  bar.arrive(K_FULL)      ──────→   bar.sync(K_FULL)
                                    ldmatrix K → QK MMA
  bar.sync(V_EMPTY)                  ...QK MMA continues...
  cp.async V[n] → SMEM              ...QK MMA continues...
  cp.async.commit_group              ...
  cp.async.wait_all                   ...
  bar.arrive(V_FULL)      ──────→   softmax
                                    bar.sync(V_FULL)
                                    ldmatrix V → PV MMA
                                    bar.arrive(K_EMPTY) ──→ (DMA warp unblocked)
                                    bar.arrive(V_EMPTY) ──→ (DMA warp unblocked)
```

## Files

- **Modify:** `conf/fixtures/fa4/generated.cu` — complete kernel rewrite
- **No other files change** — kernel_run signature, env parsing, helper functions all stay the same

---

## Task 1: Scaffold the warp-specialized kernel structure

**Files:**
- Modify: `conf/fixtures/fa4/generated.cu`

- [ ] **Step 1: Add named barrier helper functions**

Add after the existing `mma_m16n8k16` function (line ~132):

```cuda
/* Named barrier primitives for warp specialization */
__device__ inline void bar_sync(int barrier_id, int num_threads) {
    asm volatile("bar.sync %0, %1;" :: "r"(barrier_id), "r"(num_threads));
}
__device__ inline void bar_arrive(int barrier_id, int num_threads) {
    asm volatile("bar.arrive %0, %1;" :: "r"(barrier_id), "r"(num_threads));
}

/* Barrier IDs */
static constexpr int BAR_K_FULL  = 1;  /* DMA → MMA: K tile ready */
static constexpr int BAR_K_EMPTY = 2;  /* MMA → DMA: K tile consumed */
static constexpr int BAR_V_FULL  = 3;  /* DMA → MMA: V tile ready */
static constexpr int BAR_V_EMPTY = 4;  /* MMA → DMA: V tile consumed */

static constexpr int DMA_THREADS = 32;   /* 1 warp */
static constexpr int MMA_THREADS = 128;  /* 4 warps */
static constexpr int ALL_THREADS = 160;  /* 5 warps total */
```

- [ ] **Step 2: Create the new kernel signature with 5 warps**

Replace the existing `flash_attention_kernel` template with a new version. Change `NUM_WARPS` from 4 to 5, and `__launch_bounds__` to 160:

```cuda
template<int BLOCK_Q, int BLOCK_KV, int DIM>
__launch_bounds__(ALL_THREADS)
__global__
void flash_attention_kernel_ws(
    const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    nv_bfloat16 *O,
    int B, int S, int H,
    int len_q, int len_kv,
    int is_causal)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int is_dma_warp = (warp_id == 0);

    /* Block/batch/head decomposition (same as before) */
    constexpr int MMA_WARPS = 4;
    constexpr int WARP_Q = BLOCK_Q / MMA_WARPS;
    const int num_q_blocks = cdiv(len_q, BLOCK_Q);
    const int bs_id = blockIdx.x / num_q_blocks;
    const int q_block_id = blockIdx.x % num_q_blocks;
    const int batch_id = bs_id / H;
    const int head_id = bs_id % H;
    const int seq_stride = H * DIM;

    /* Shared memory layout */
    extern __shared__ nv_bfloat16 smem[];
    const uint32_t Q_smem = __cvta_generic_to_shared(smem);
    const uint32_t K_smem = Q_smem + BLOCK_Q * DIM * sizeof(nv_bfloat16);
    const uint32_t V_smem = K_smem + 2 * BLOCK_KV * DIM * sizeof(nv_bfloat16);

    /* Phase 1: ALL threads load Q cooperatively (same as before) */
    /* ... Q load code here ... */

    __syncthreads();  /* Last __syncthreads — after this, only named barriers */

    if (is_dma_warp) {
        /* DMA warp: load K/V tiles in a loop */
        dma_loop(K, V, K_smem, V_smem, ...);
    } else {
        /* MMA warps: compute QK, softmax, PV */
        mma_loop(Q_smem, K_smem, V_smem, O, ...);
    }
}
```

- [ ] **Step 3: Update kernel_run launcher**

Change `NUM_WARPS` from 4 to 5 and thread count to 160:

```cuda
const int TB_SIZE = 160;  /* 5 warps: 1 DMA + 4 MMA */
int num_blocks = effective_bs * cdiv(S, BLOCK_Q);
auto kernel = flash_attention_kernel_ws<128, 64, 128>;
kernel<<<num_blocks, TB_SIZE, smem_size, stream>>>(...);
```

- [ ] **Step 4: Verify compilation (skeleton only)**

Run: `nvcc -arch=sm_120 -std=c++17 -O3 -c conf/fixtures/fa4/generated.cu -o /dev/null`
Expected: compiles with no errors (kernel body is empty at this point)

- [ ] **Step 5: Commit scaffold**

```bash
git add conf/fixtures/fa4/generated.cu
git commit -m "refactor: scaffold warp-specialized FA4 kernel (5 warps, named barriers)"
```

---

## Task 2: Implement the DMA warp loop

**Files:**
- Modify: `conf/fixtures/fa4/generated.cu`

The DMA warp runs a simple loop: for each KV block, load K and V into SMEM, signal MMA warps, wait for them to consume.

- [ ] **Step 1: Write the DMA warp function**

Add as a `__device__` function (not inline — separate function to help compiler allocate fewer registers for DMA warp):

```cuda
template<int BLOCK_KV, int DIM>
__device__ __noinline__
void dma_warp_loop(
    const nv_bfloat16 *K_base,
    const nv_bfloat16 *V_base,
    uint32_t K_smem,
    uint32_t V_smem,
    int seq_stride,
    int max_kv_iter,
    int lane_id)
{
    const nv_bfloat16 *K_ptr = K_base;
    const nv_bfloat16 *V_ptr = V_base;

    for (int kv_id = 0; kv_id < max_kv_iter; kv_id++) {
        /* Wait for MMA warps to finish consuming previous K */
        if (kv_id > 0) bar_sync(BAR_K_EMPTY, ALL_THREADS);

        /* Load K tile into SMEM[kv_id % 2] */
        uint32_t K_dst = K_smem + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(nv_bfloat16));
        /* DMA warp has 32 threads. Each cp.async loads 16 bytes = 8 bf16.
         * Tile = BLOCK_KV * DIM = 64 * 128 = 8192 elements.
         * Each thread loads 8192/32 / 8 = 32 cp.async operations per tile. */
        dma_load_tile<BLOCK_KV, DIM>(K_dst, K_ptr, seq_stride, lane_id);
        K_ptr += BLOCK_KV * seq_stride;
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        /* Signal K is ready */
        bar_arrive(BAR_K_FULL, ALL_THREADS);

        /* Wait for MMA warps to finish consuming previous V */
        if (kv_id > 0) bar_sync(BAR_V_EMPTY, ALL_THREADS);

        /* Load V tile into SMEM */
        dma_load_tile<BLOCK_KV, DIM>(V_smem, V_ptr, seq_stride, lane_id);
        V_ptr += BLOCK_KV * seq_stride;
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        /* Signal V is ready */
        bar_arrive(BAR_V_FULL, ALL_THREADS);
    }
}
```

- [ ] **Step 2: Write the DMA tile load helper (32 threads)**

```cuda
template<int HEIGHT, int WIDTH>
__device__ inline
void dma_load_tile(uint32_t dst, const nv_bfloat16 *src, int src_stride, int lane_id) {
    /* 32 threads load a HEIGHT x WIDTH tile of bf16 via cp.async.
     * Each cp.async: 16 bytes = 8 elements.
     * Total loads: HEIGHT * WIDTH / 8 = 1024 per tile.
     * Per thread: 1024 / 32 = 32 cp.async ops. */
    constexpr int ELEMS_PER_LOAD = 8;
    constexpr int TOTAL_LOADS = HEIGHT * WIDTH / ELEMS_PER_LOAD;
    constexpr int LOADS_PER_THREAD = TOTAL_LOADS / 32;

    #pragma unroll
    for (int i = 0; i < LOADS_PER_THREAD; i++) {
        const int load_idx = i * 32 + lane_id;
        const int row = (load_idx * ELEMS_PER_LOAD) / WIDTH;
        const int col = (load_idx * ELEMS_PER_LOAD) % WIDTH;

        const uint32_t dst_addr = swizzle<WIDTH * sizeof(nv_bfloat16)>(
            dst + (row * WIDTH + col) * sizeof(nv_bfloat16));
        const nv_bfloat16 *src_addr = src + row * src_stride + col;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :: "r"(dst_addr), "l"(src_addr));
    }
}
```

- [ ] **Step 3: Verify compilation**

Run: `nvcc -arch=sm_120 -std=c++17 -O3 -c conf/fixtures/fa4/generated.cu -o /dev/null`
Expected: compiles cleanly

- [ ] **Step 4: Commit DMA warp**

```bash
git add conf/fixtures/fa4/generated.cu
git commit -m "feat: implement DMA warp loop with named barrier signaling"
```

---

## Task 3: Implement the MMA warp compute loop

**Files:**
- Modify: `conf/fixtures/fa4/generated.cu`

MMA warps (warp_id 1-4, tid 32-159) run the attention compute loop. They wait for DMA warp to fill SMEM, then do QK MMA → softmax → PV MMA → signal DMA warp.

- [ ] **Step 1: Write the MMA warp function**

The MMA warp loop is structurally the same as the current kernel's main loop, but:
- Uses `bar.sync(BAR_K_FULL)` instead of `cp.async.wait_group` + `__syncthreads()`
- Uses `bar.sync(BAR_V_FULL)` instead of the V wait
- Signals `bar.arrive(BAR_K_EMPTY)` and `bar.arrive(BAR_V_EMPTY)` after consuming tiles
- `mma_tid = tid - DMA_THREADS` (offset by 32 since warp 0 is DMA)
- `mma_warp_id = warp_id - 1` (warps 1-4 become MMA warps 0-3)
- WARP_Q = BLOCK_Q / 4 = 32 (same as before with 4 MMA warps)

```cuda
template<int BLOCK_Q, int BLOCK_KV, int DIM>
__device__ __noinline__
void mma_warp_loop(
    uint32_t Q_smem,
    uint32_t K_smem,
    uint32_t V_smem,
    nv_bfloat16 *O_base,
    int seq_stride,
    int max_kv_iter,
    int q_block_id,
    int is_causal,
    int mma_warp_id,  /* 0-3 */
    int lane_id)
{
    constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    constexpr int WARP_Q = BLOCK_Q / 4;  /* 32 */

    /* Pre-compute SMEM ldmatrix addresses (same as before but using mma_warp_id) */
    uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
    /* ... same address computation using mma_warp_id instead of warp_id ... */

    const float softmax_scale_log2 = rsqrtf(float(DIM)) * 1.4426950408889634f;
    float rowmax[WARP_Q / MMA_M][2];
    float rowsumexp[WARP_Q / MMA_M][2] = {};
    float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};
    /* ... init rowmax to -FLT_MAX ... */

    for (int kv_id = 0; kv_id < max_kv_iter; kv_id++) {
        /* Wait for DMA warp to fill K[kv_id] */
        bar_sync(BAR_K_FULL, ALL_THREADS);

        uint32_t K_cur = K_smem_thread + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(nv_bfloat16));

        /* QK MMA + softmax (same per-mma_id_q processing as current kernel) */
        uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
        for (int mma_id_q = 0; ...) {
            /* ... same QK MMA + exp2 softmax code ... */
        }

        /* Signal K consumed — DMA warp can overwrite K[kv_id % 2] */
        bar_arrive(BAR_K_EMPTY, ALL_THREADS);

        /* Wait for DMA warp to fill V[kv_id] */
        bar_sync(BAR_V_FULL, ALL_THREADS);

        /* PV MMA (same as current) */
        for (int mma_id_kv = ...) { /* ... */ }

        /* Signal V consumed */
        bar_arrive(BAR_V_EMPTY, ALL_THREADS);
    }

    /* Write O to global memory (same strided output as current) */
    /* ... */
}
```

- [ ] **Step 2: Wire up the kernel to call DMA/MMA functions**

In the main kernel, after Q load + `__syncthreads()`:

```cuda
if (is_dma_warp) {
    dma_warp_loop<BLOCK_KV, DIM>(
        K_base_ptr, V_base_ptr, K_smem, V_smem,
        seq_stride, max_kv_iter, lane_id);
} else {
    mma_warp_loop<BLOCK_Q, BLOCK_KV, DIM>(
        Q_smem, K_smem, V_smem, O_base,
        seq_stride, max_kv_iter, q_block_id, is_causal,
        warp_id - 1, lane_id);
}
```

- [ ] **Step 3: Verify compilation and register usage**

Run: `nvcc -arch=sm_120 -std=c++17 -O3 --resource-usage -c conf/fixtures/fa4/generated.cu -o /dev/null`
Expected: compiles, total registers ≤ 255, check for spill

- [ ] **Step 4: Commit MMA warp loop**

```bash
git add conf/fixtures/fa4/generated.cu
git commit -m "feat: implement MMA warp compute loop with barrier sync"
```

---

## Task 4: Correctness verification

**Files:**
- Modify: `conf/fixtures/fa4/generated.cu` (bug fixes only)

- [ ] **Step 1: Build the binary**

```bash
cd /home/centos/kernel_lab/.claude/worktrees/cuda-exec
/usr/local/cuda/bin/nvcc -arch=native -std=c++17 -O3 -lineinfo \
  -I cuda_exec/scripts \
  cuda_exec/scripts/eval_harness.cu conf/fixtures/fa4/generated.cu \
  -o /tmp/fa4_bin/fa4_ws.bin
```

- [ ] **Step 2: Run small correctness test**

```bash
CUDA_EXEC_PARAM_BATCH_SIZE=1 CUDA_EXEC_PARAM_SEQ_LEN=128 \
CUDA_EXEC_PARAM_NUM_HEADS=2 CUDA_EXEC_PARAM_HEAD_DIM=128 \
CUDA_EXEC_PARAM_CAUSAL=false CUDA_EXEC_PARAM_INPUT_SIZE=$((1*128*2*128)) \
CUDA_EXEC_HARNESS_NUM_INPUTS=3 CUDA_EXEC_HARNESS_NUM_OUTPUTS=1 \
CUDA_EXEC_NUM_WARMUPS=1 CUDA_EXEC_NUM_TRIALS=1 \
/tmp/fa4_bin/fa4_ws.bin
```

Expected: JSON output with non-zero `output.result` values. If all zeros or NaN: barrier sync bug.

- [ ] **Step 3: Cross-validate against PyTorch SDPA**

Write a quick Python script that runs both the generated binary and `F.scaled_dot_product_attention` on the same input, compares outputs. Use `torch.allclose(atol=0.05, rtol=0.05)` for BF16 tolerance.

- [ ] **Step 4: Test causal mask**

Same as step 3 but with `CUDA_EXEC_PARAM_CAUSAL=true`.

- [ ] **Step 5: Commit any fixes**

```bash
git add conf/fixtures/fa4/generated.cu
git commit -m "fix: correctness fixes for warp-specialized FA4 kernel"
```

---

## Task 5: Performance benchmark and tuning

**Files:**
- Modify: `conf/fixtures/fa4/generated.cu` (tuning only)

- [ ] **Step 1: Run full 8-config benchmark**

Use the existing `/tmp/bench_fa4_all.py` script with the new binary.
Target: ≥ 200 TFLOPS on noncausal long-sequence configs.

- [ ] **Step 2: NCU profile the warp-specialized kernel**

```bash
sudo --preserve-env /usr/local/cuda/bin/ncu --set detailed \
  --kernel-name regex:"flash_attention" --launch-count 1 \
  ... python script ...
```

Check: tensor pipe utilization should be > 80% (was 77.4%).
Check: both DMA and MMA warps should show activity.

- [ ] **Step 3: Tune barrier placement if needed**

If DMA warp is idle too long (K loads finish before MMA consumes previous tile):
- Consider having DMA warp load V simultaneously with K (pipeline V ahead)
- Consider having DMA warp do 2 K tiles ahead (triple buffer)

If MMA warps idle waiting for DMA:
- DMA warp is the bottleneck — consider using 2 DMA warps (3 MMA + 2 DMA)

- [ ] **Step 4: Check register distribution**

The `__noinline__` on `dma_warp_loop` should give the DMA function fewer registers. Verify with `--resource-usage`. If DMA function uses too many registers, add `__launch_bounds__` or `asm volatile("setmaxnreg...")`.

- [ ] **Step 5: Commit tuned version**

```bash
git add conf/fixtures/fa4/generated.cu
git commit -m "perf: tune warp-specialized FA4 kernel"
git push origin cuda-exec
```

---

## Task 6: Final merge

- [ ] **Step 1: Run full benchmark one more time to confirm**
- [ ] **Step 2: Merge to main**

```bash
cd /home/centos/kernel_lab
git checkout main && git pull origin main
git merge cuda-exec --no-ff -m "Merge branch 'cuda-exec': warp-specialized FA4 kernel"
git push origin main
```

---

## Verification Checklist

- [ ] All 8 configs produce correct output (validated against PyTorch SDPA)
- [ ] Causal mask works correctly
- [ ] No register spill (0 bytes spill stores/loads)
- [ ] Tensor pipe utilization > 80% (NCU)
- [ ] Performance ≥ 200 TFLOPS on noncausal b1-s32768
- [ ] Binary compiles with `nvcc -arch=sm_120 -std=c++17 -O3`
