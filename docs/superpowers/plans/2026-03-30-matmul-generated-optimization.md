# Generated Matmul Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the generated CUDA BF16 matmul kernel faster than the CuTe DSL reference (0.666ms at 4096×4096).

**Architecture:** Replace the scalar-gather B loading with `cp.async` bulk copy by pre-transposing B(K,N)→B_t(N,K) in `kernel_run()`. B_t is then contiguous along K, enabling the same efficient 16-byte `cp.async` path already used for A. Optionally add persistent tile scheduling if needed to beat the reference.

**Tech Stack:** CUDA C++, PTX inline asm (`cp.async`, `ldmatrix`, `mma.sync`), SM120 (RTX 5090)

---

## File Map

- **Modify:** `conf/fixtures/matmul/generated.cu` — the only file. All changes are here.

## Current Kernel Anatomy (context for all tasks)

The existing `generated.cu` has:
- `cp_async()`, `load_matrix_x4()`, `load_matrix_x2()`, `mma_m16n8k16()` — PTX helpers (lines 25-74)
- `gather_b_k8()` — scalar gather for B, 8× `__ldg` per call (lines 82-89)
- `load_b_tile_to_smem()` — writes B tile to SMEM via gather (lines 98-112)
- `mma_matmul_bf16()` — main GEMM kernel, 128×128 tiles, 3-stage pipeline (lines 121-235)
- `kernel_run()` — entry point, dispatches grid launch (lines 261-277)

**Key SMEM layout:** Both `As[N_STAGES*64][8]` and `Bs[N_STAGES*64][8]` use identical layouts. Each row holds 8 `uint4` = 128 bytes = 64 BF16 values. The store mapping (`storeRow`, `storeCol` with XOR swizzle) is shared for A and B.

**A loading pattern** (which we'll replicate for B):
```c
const uint4 *aGlobalAddress = globalTileA + (warpID * 8 + laneID / 4) * K8 + laneID % 4;
// ...
cp_async(aStorePtr[storeRow     ] + storeCol, aGlobalAddress + kStart);
cp_async(aStorePtr[storeRow + 32] + storeCol, aGlobalAddress + 64 * K8 + kStart);
```
Each of the 256 threads does 2× `cp.async` (16 bytes each), loading 2× 4KB = 8KB total for a 128×32 tile.

---

### Task 1: Add transpose kernel and wire it into kernel_run

**Files:**
- Modify: `conf/fixtures/matmul/generated.cu:237-277` (add transpose kernel + update kernel_run)

- [ ] **Step 1: Add the transpose kernel after the PTX helpers (before main GEMM kernel)**

Insert after line 112 (after `load_b_tile_to_smem`), before line 114 (before main kernel comment):

```c
/* -------------------------------------------------------------------------
 * Transpose kernel: B(K,N) row-major → B_t(N,K) row-major
 * 32×32 tiled, shared memory with +1 padding to avoid bank conflicts
 * ------------------------------------------------------------------------- */
__global__ void transpose_bf16(const __nv_bfloat16 *src, __nv_bfloat16 *dst,
                               int rows, int cols) {
    __shared__ __nv_bfloat16 tile[32][33]; /* +1 padding avoids bank conflicts */

    int bx = blockIdx.x * 32;
    int by = blockIdx.y * 32;

    /* Load 32×32 tile from src (rows × cols, row-major) */
    int x = bx + threadIdx.x;
    int y = by + threadIdx.y;
    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        if (y + j < rows && x < cols)
            tile[threadIdx.y + j][threadIdx.x] = src[(y + j) * cols + x];
    }
    __syncthreads();

    /* Store transposed: dst is cols × rows, row-major */
    x = by + threadIdx.x;  /* swapped block coordinates */
    y = bx + threadIdx.y;
    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        if (y + j < cols && x < rows)
            dst[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}
```

- [ ] **Step 2: Add static buffer and update kernel_run to transpose B**

Replace the `kernel_run` function (lines 261-277) with:

```c
static __nv_bfloat16 *s_Bt = nullptr;
static size_t s_Bt_size = 0;

extern "C" int kernel_run(__nv_bfloat16 **inputs,  int num_inputs,
                          __nv_bfloat16 **outputs, int num_outputs,
                          int n, cudaStream_t stream) {
    const __nv_bfloat16 *A = inputs[0];
    const __nv_bfloat16 *B = inputs[1];
    __nv_bfloat16       *C = outputs[0];

    ensure_shape(n);
    int M = s_M, N = s_N, K = s_K;

    /* Allocate / reuse transpose buffer for B_t(N,K) */
    size_t need = (size_t)N * K * sizeof(__nv_bfloat16);
    if (s_Bt == nullptr || s_Bt_size < need) {
        if (s_Bt) cudaFree(s_Bt);
        cudaMalloc(&s_Bt, need);
        s_Bt_size = need;
    }

    /* Transpose B(K,N) → B_t(N,K) */
    dim3 tBlk(32, 8);
    dim3 tGrid((N + 31) / 32, (K + 31) / 32);
    transpose_bf16<<<tGrid, tBlk, 0, stream>>>(B, s_Bt, K, N);

    /* Launch GEMM with transposed B */
    dim3 threads(16, 16);
    dim3 grid(N / 128, M / 128);
    mma_matmul_bf16<<<grid, threads, 0, stream>>>(A, s_Bt, C, M, N, K);

    return 0;
}
```

- [ ] **Step 3: Compile and verify it builds**

Run:
```bash
RUN_TAG="gen-opt-$(date +%Y%m%d%H%M%S)" && \
make -C cuda_exec/scripts compile KERNEL=matmul RUN_TAG="$RUN_TAG" \
  VENV_PYTHON=/home/centos/kernel_lab/cuda_exec/.venv/bin/python 2>&1 | tail -5
```
Expected: `--- compile [matmul]: done ---`

- [ ] **Step 4: Commit**

```bash
git add conf/fixtures/matmul/generated.cu
git commit -m "feat: add transpose kernel for B in generated matmul"
```

---

### Task 2: Replace gather-based B loading with cp.async

**Files:**
- Modify: `conf/fixtures/matmul/generated.cu:121-235` (main GEMM kernel)

Now that B is transposed to B_t(N,K) — contiguous along K — we replace the gather loading with cp.async, mirroring the A loading path.

- [ ] **Step 1: Update the kernel signature and add B global address computation**

In `mma_matmul_bf16`, the `B` parameter now points to B_t(N,K). Add a global address pointer for B, mirroring `aGlobalAddress`. After line 156 (`const uint4 *aGlobalAddress = ...`), add:

```c
    const uint4 *globalTileB = reinterpret_cast<const uint4 *>(B + blockColStart * K);
    const uint4 *bGlobalAddress = globalTileB + (warpID * 8 + laneID / 4) * K8 + laneID % 4;
```

- [ ] **Step 2: Replace B loading in the prelude (lines 161-175)**

Replace the prelude loop body to use cp.async for both A and B:

```c
    /* ---- PRELUDE: load first (N_STAGES - 1) tiles ---- */
    for (int nStage = 0; nStage < N_STAGES - 1; nStage++) {
        int kStart = nStage * 4;
        aStorePtr = As + 64 * nStage;
        bStorePtr = Bs + 64 * nStage;

        /* A: cp.async (contiguous along K) */
        cp_async(aStorePtr[storeRow     ] + storeCol, aGlobalAddress + kStart);
        cp_async(aStorePtr[storeRow + 32] + storeCol, aGlobalAddress + 64 * K8 + kStart);

        /* B: cp.async (B_t is now contiguous along K, same pattern as A) */
        cp_async(bStorePtr[storeRow     ] + storeCol, bGlobalAddress + kStart);
        cp_async(bStorePtr[storeRow + 32] + storeCol, bGlobalAddress + 64 * K8 + kStart);

        asm volatile("cp.async.commit_group;\n" ::);
    }
```

- [ ] **Step 3: Replace B loading in the main loop (lines 199-209)**

Replace the B prefetch section in the main loop with cp.async, and combine the A and B cp.async calls into one commit group:

```c
        /* Prefetch next tiles via cp.async */
        kStart = (kStart > kStartMax) ? kStartMax : kStart;
        cp_async(aStorePtr[storeRow     ] + storeCol, aGlobalAddress + kStart);
        cp_async(aStorePtr[storeRow + 32] + storeCol, aGlobalAddress + 64 * K8 + kStart);
        cp_async(bStorePtr[storeRow     ] + storeCol, bGlobalAddress + kStart);
        cp_async(bStorePtr[storeRow + 32] + storeCol, bGlobalAddress + 64 * K8 + kStart);
        asm volatile("cp.async.commit_group;\n" ::);
```

This replaces lines 199-209. Remove the `nextKBlock` variable and `load_b_tile_to_smem` call entirely.

- [ ] **Step 4: Remove dead code**

Delete the `gather_b_k8` function (lines 82-89) and `load_b_tile_to_smem` function (lines 98-112). These are no longer used.

- [ ] **Step 5: Compile, test correctness and performance**

```bash
rm -f /home/centos/.cuda_exec/.lock_cuda_0 && \
RUN_TAG="gen-opt-$(date +%Y%m%d%H%M%S)" && \
make -C cuda_exec/scripts compile evaluate KERNEL=matmul CONFIG=mat-4096x4096 \
  RUN_TAG="$RUN_TAG" VENV_PYTHON=/home/centos/kernel_lab/cuda_exec/.venv/bin/python 2>&1 | \
  grep -E 'status=|correctness|reference:|generated:|speedup'
```

Expected: correctness passes, generated latency drops significantly from 0.764ms.

- [ ] **Step 6: Commit**

```bash
git add conf/fixtures/matmul/generated.cu
git commit -m "perf: replace scatter-gather B loading with cp.async in generated matmul"
```

---

### Task 3: Evaluate and iterate if needed

**Files:**
- Modify: `conf/fixtures/matmul/generated.cu` (if further tuning is needed)

- [ ] **Step 1: Run full evaluation across sizes**

```bash
rm -f /home/centos/.cuda_exec/.lock_cuda_0 && \
RUN_TAG="gen-opt-$(date +%Y%m%d%H%M%S)" && \
make -C cuda_exec/scripts compile KERNEL=matmul RUN_TAG="$RUN_TAG" \
  VENV_PYTHON=/home/centos/kernel_lab/cuda_exec/.venv/bin/python 2>&1 | tail -3 && \
for cfg in mat-256x256 mat-1024x1024 mat-4096x4096 mat-8192x8192; do
  rm -f /home/centos/.cuda_exec/.lock_cuda_0
  make -C cuda_exec/scripts evaluate KERNEL=matmul CONFIG=$cfg RUN_TAG="$RUN_TAG" \
    VENV_PYTHON=/home/centos/kernel_lab/cuda_exec/.venv/bin/python 2>&1 | \
    grep -E 'status=|correctness|reference:|generated:|speedup'
done
```

Expected: generated faster than reference (speedup > 1.0x) at 4096×4096.

- [ ] **Step 2: If generated is still slower than reference, add persistent tile scheduling**

Add an atomic tile counter for persistent scheduling. Replace the grid launch in `kernel_run()`:

```c
/* In kernel_run, replace grid launch: */
int numTiles = (M / 128) * (N / 128);
int numSMs = 170;  /* RTX 5090 SM count */
int numCTAs = min(numTiles, numSMs);
dim3 threads(16, 16);
dim3 grid(numCTAs, 1);
/* Pass numTiles, N/128 as additional args for tile index computation */
```

And in the kernel, replace `blockIdx.x/y` with:
```c
/* Persistent loop: each CTA processes multiple tiles */
__shared__ int tileIdx;
if (threadID == 0) tileIdx = atomicAdd(&g_tile_counter, 1);
__syncthreads();
while (tileIdx < numTiles) {
    int tileRow = tileIdx / (N / 128);
    int tileCol = tileIdx % (N / 128);
    int blockRowStart = tileRow * 128;
    int blockColStart = tileCol * 128;
    /* ... existing GEMM body ... */
    if (threadID == 0) tileIdx = atomicAdd(&g_tile_counter, 1);
    __syncthreads();
}
```

- [ ] **Step 3: Re-evaluate and commit if improved**

Run the same evaluation as Step 1. If improved, commit:
```bash
git add conf/fixtures/matmul/generated.cu
git commit -m "perf: add persistent tile scheduling to generated matmul"
```
