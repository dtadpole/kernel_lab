# FA4 K Fragment Prefetching + Spill Reduction

## Problem

The FA4 kernel achieves 88-89% of FA4 CuTe DSL on non-causal and 94-95% on causal configs. NCU profiling shows SMEM loads at 19.84% of time (1.88B wavefronts) and 24 bytes/thread register spill (100M local load sectors).

The QK inner loop loads K fragments from SMEM without prefetching — each `ldmatrix` stalls until the SMEM data arrives before the MMA can proceed. Q and V already use prefetching (load next fragment while current MMA executes), but K does not.

## Baseline Performance

| Config | Gen/FA4 |
|--------|---------|
| Causal avg | 0.94x |
| Non-causal avg | 0.88x |

## Root Cause

### K fragment not prefetched in QK loop

Current QK inner loop pattern:
```
for mma_id_d = 0 to DIM/MMA_K:
    prefetch Q[d+1]
    for mma_id_kv = 0 to BLOCK_KV/MMA_N:
        K_frag = ldmatrix(K[kv, d])  ← BLOCKING: stalls until SMEM data arrives
        mma(Q, K_frag, S_local[kv])
    swap Q_cur = Q_next
```

Each K ldmatrix takes ~20-30 cycles. With BLOCK_KV/MMA_N = 8 and DIM/MMA_K = 8, that's 64 ldmatrix calls per mma_id_q, all blocking.

### 24-byte register spill

The compiler spills 3 registers (24 bytes) per thread. Over 4096 blocks × 160 threads × multiple KV iterations, this generates 100M L1 load sectors. While L1 hit rate is 99.98%, the spill/reload instructions consume issue slots.

## Design

### 1. K fragment prefetching in QK loop

Add K prefetching following the same pattern as V prefetching in the PV loop:

```
for mma_id_d = 0 to DIM/MMA_K:
    prefetch Q[d+1]
    K_cur = ldmatrix(K[0, d])           ← prefetch first K of this d-step
    for mma_id_kv = 0 to BLOCK_KV/MMA_N:
        if has_next_kv:
            K_next = ldmatrix(K[kv+1, d])   ← prefetch next K while MMA runs
        mma(Q, K_cur, S_local[kv])
        swap K_cur = K_next
```

This overlaps K ldmatrix latency with the MMA instruction for the current step. The MMA takes ~16 cycles, and ldmatrix takes ~20-30 cycles, so we overlap most of the SMEM latency.

**Register impact:** K_cur (2 regs) + K_next (2 regs) = 4 registers total for K fragments, same as current (K_frag is 2 regs). The "next" buffer adds 2 registers. At 255 regs with 24 bytes spill, adding 2 regs may push spill to ~32 bytes. This is acceptable since the L1 hit rate is 99.98%.

### 2. Spill reduction via `__restrict__` and recomputation

Add `__restrict__` to kernel pointer parameters:
```c
void flash_attention_kernel_ws(
    const nv_bfloat16 * __restrict__ Q,
    const nv_bfloat16 * __restrict__ K,
    const nv_bfloat16 * __restrict__ V,
    nv_bfloat16 * __restrict__ O, ...)
```

This tells the compiler Q, K, V, O don't alias, potentially enabling better register allocation and reducing spill.

Additionally, move `softmax_scale_log2` computation inside the MMA warp body (recompute from constant `DIM` instead of keeping it live across the entire function).

## Expected Impact

- K prefetching: 2-5% improvement (overlap ~64 ldmatrix stalls with MMA per mma_id_q)
- Spill reduction: 0-1% improvement (frees issue slots, slight reduction in local memory traffic)
- Combined: 3-6%, bringing noncausal to ~91-94% and causal to ~96-99%

## Risk

Low. The K prefetch pattern is identical to the V prefetch already in the PV loop. The `__restrict__` addition is a safe compiler hint. No algorithmic changes, no barrier changes, no SMEM layout changes.

## Files Modified

- `conf/fixtures/fa4/generated.cu` — QK inner loop (K prefetch), kernel signature (__restrict__), softmax_scale recomputation

## Verification

1. Compile: check register count, spill bytes
2. Benchmark all 8 configs
3. Compare Gen/FA4 ratios vs baseline
