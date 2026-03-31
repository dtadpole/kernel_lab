# FA4 Kernel Rewrite: Eliminate DMA Warp, Match FA4 CuTe DSL Architecture

## Problem

The FA4 kernel achieves 88-95% of FA4 CuTe DSL reference. Research into the FA4 CuTe DSL source code (`flash_fwd.py`, `FlashAttentionForwardSm80/Sm120`) reveals fundamental architectural differences explaining the gap.

## Key Findings from FA4 CuTe DSL Source

FA4 on SM120 uses `FlashAttentionForwardSm120` which subclasses `FlashAttentionForwardSm80`:
- **128 threads (4 warps)** — all threads do both loading and MMA
- **No dedicated DMA warp** — cp.async overlap handles data movement
- **tile_m=128, tile_n=64, head_dim=128** — same tiling as our kernel
- **num_stages=1** — single-buffered K/V
- **cp.async pipeline**: V loads overlap with QK MMA, K loads overlap with PV MMA
- **Layout transpose**: host transposes [B,S,H,D] → selects [S,D,H,B] strides

## Architecture Differences

| Feature | FA4 CuTe DSL | Our Kernel |
|---------|-------------|------------|
| Threads | 128 (4 warps, all compute+load) | 160 (32 DMA + 128 MMA) |
| DMA warp | None | Warp 0 dedicated |
| Pipeline sync | cp_async_wait_group + __syncthreads | Named barriers (bar.sync/bar.arrive) |
| K/V buffering | Single-buffered (num_stages=1) | Double-buffered |
| GEMM k-loop | Prefetch both A+B fragments | Prefetch Q only, K loaded inline |
| SMEM usage | Q + K + V (no double buffer) = 32+16+16=64KB | Q + K×2 + V×2 = 96KB |

## Design

### Thread Organization

Drop from 160 to **128 threads (4 warps)**. All threads participate in:
1. cp.async loading of Q, K, V tiles
2. mma.sync computation (QK and PV)
3. Softmax computation
4. Output stores

### Pipeline: cp.async overlap (no named barriers)

Follow FA4's `compute_one_n_block` pattern:

```
Prologue:
  ALL threads: cp.async load Q → SMEM, commit
  ALL threads: cp.async load K[last] → SMEM, commit (first KV tile)
  cp_async_wait_group(0), __syncthreads()

Per KV iteration (n_block from last to first):
  1. ALL threads: cp.async load V[n] → SMEM, commit
  2. QK GEMM: S = Q @ K[n]^T (ALL threads, K already in SMEM)
     - Prefetch both Q and K fragments in d-loop
  3. cp_async_wait_group(0), __syncthreads()  [wait for V]
  4. ALL threads: cp.async load K[n-1] → SMEM, commit
  5. Softmax + mask on S
  6. PV GEMM: O += P @ V[n] (ALL threads, V now in SMEM)
     - P in registers, V prefetched in d-loop
  7. __syncthreads()  [ensure K load complete before next iter]
```

Key: V loads (step 1) overlap with QK GEMM (step 2). K loads (step 4) overlap with softmax+PV GEMM (steps 5-6). No named barriers needed.

### SMEM Layout (single-buffered)

```
Q:  128 × 128 × 2 = 32,768 bytes (persistent)
K:  64 × 128 × 2  = 16,384 bytes (single buffer, reused each iter)
V:  64 × 128 × 2  = 16,384 bytes (single buffer, reused each iter)
Total: 65,536 bytes (64KB) — well within 99KB limit
```

Note: Q and V SMEM can overlap (Q loaded once, then V reuses the space after Q is consumed). FA4 does this when `Q_in_regs=True`. For simplicity, keep them separate first.

### GEMM Inner Loop (matching ampere_helpers.py `gemm()`)

```c
// Prefetch A[0] and B[0] from SMEM
ldmatrix A_cur from sQ[..., d=0]
ldmatrix B_cur from sK[..., d=0]

for d = 0 to DIM/MMA_K:
    // Prefetch A[d+1] and B[d+1] (if not last)
    if d+1 < DIM/MMA_K:
        ldmatrix A_next from sQ[..., d+1]
        ldmatrix B_next from sK[..., d+1]
    mma(A_cur, B_cur, acc)
    swap A_cur = A_next, B_cur = B_next
```

Both Q and K fragments are prefetched, overlapping ldmatrix latency with MMA execution.

### MMA Configuration

4 warps × m16n8k16: each warp handles a 16-row stripe of Q.
- WARP_Q = 128 / 4 = 32 rows per warp (same as current)
- With `permutation_mnk=(64, 16, 16)`: 4 warps collectively cover 64 M-rows with 16-wide K steps

### kernel_run Changes

Pass Q, K, V, O with **same [B,S,H,D] layout** (no host transpose — the kernel handles strided access like before). Changing layout would require modifying the harness contract.

Launch with **128 threads** instead of 160. Remove all named barrier constants.

## Expected Impact

- **+20% MMA throughput**: 128/128 threads compute vs 128/160 (was wasting 32 DMA threads)
- **Better cp.async overlap**: V loads during QK, K loads during PV — no DMA/MMA serialization
- **Less SMEM**: 64KB vs 96KB — room for future occupancy=2 or larger tiles
- **Simpler code**: no warp specialization, no named barriers

Target: noncausal **0.95x+**, causal **0.98x+** (matching or exceeding FA4 reference)

## Files Modified

- `conf/fixtures/fa4/generated.cu` — complete kernel rewrite

## Verification

1. Compile: check registers, spill, thread count
2. Smoke test: small config correctness
3. Benchmark all 8 configs
4. Compare Gen/FA4 ratios vs baseline (0.88-0.95x)
