# FA4 Inline Warp Specialization + DMA V-Overlap Optimization

## Problem

The warp-specialized FA4 kernel achieves 87.5% tensor pipe utilization but has 16.9M local memory spill requests from `__noinline__` function call ABI overhead. This accounts for ~5-8% of the remaining 12% gap vs FA4 CuTe DSL reference.

## Root Cause

`__noinline__` on `dma_warp_fn` and `mma_warp_fn` forces the compiler to save/restore registers at function call boundaries. Each call crosses the ABI: caller-saved registers are spilled to local memory (stack), then restored on return. With 255 registers active, this creates significant traffic to L1/local memory.

## Solution: Two Combined Optimizations

### 1. Inline DMA/MMA into kernel body (eliminate ABI spill)

Remove `__noinline__` function calls. Instead, place DMA and MMA logic directly in the kernel body under `if (warp_id == 0) { ... } else { ... }`. The compiler allocates registers for the combined function but can optimize the divergent paths.

Risk: compiler might allocate max(DMA_regs, MMA_regs) for all threads. Mitigations:
- `__launch_bounds__(160, 1)` — tells compiler occupancy=1 block/SM, allowing up to 255 regs
- Keep DMA logic minimal (loop + cp.async + barrier) — compiler should see few live registers in DMA path
- If spill increases, add `#pragma nv_diag_suppress` or try `--maxrregcount=255`

### 2. DMA V-overlap: load V[n] during QK MMA (pipeline tightening)

Current DMA timeline per iteration:
```
DMA: wait K_EMPTY → load K → signal K_FULL → wait V_EMPTY → load V → signal V_FULL
MMA:                                         wait K_FULL → QK MMA → softmax → wait V_FULL → PV MMA → signal K_EMPTY, V_EMPTY
```

DMA is idle during QK MMA + softmax. Better:
```
DMA: wait K_EMPTY → load K → signal K_FULL → load V (immediately, no wait) → signal V_FULL
MMA:                                         wait K_FULL → QK MMA → softmax → wait V_FULL → PV MMA → signal K_EMPTY, V_EMPTY
```

This works because V_smem is single-buffered and the previous V was consumed (PV MMA done) before MMA signals V_EMPTY. But the DMA warp doesn't wait for V_EMPTY anymore — it starts loading V right after signaling K_FULL. This is safe IF:
- V from previous iteration was already consumed (guaranteed: MMA's bar.arrive(V_EMPTY) happens before bar.sync(K_FULL) in the next iteration)
- Actually NO: in the next iteration, MMA does bar.sync(K_FULL) first, then QK MMA, THEN PV MMA, THEN bar.arrive(V_EMPTY). So V from the previous iteration hasn't been consumed when DMA starts loading V for the current iteration.

Fix: double-buffer V (2 × 64 × 128 × 2 = 32KB). Total SMEM: 32 + 32 + 32 = 96KB. SM120 has 99KB — tight but fits.

With double-buffered V:
```
DMA: load K[n] → signal K_FULL → load V[n] into V_smem[n%2] → signal V_FULL → wait K_EMPTY (next iter)
MMA: wait K_FULL → QK MMA → softmax → wait V_FULL → PV MMA from V_smem[n%2] → signal K_EMPTY, V_EMPTY
```

DMA never waits for V_EMPTY (double buffer eliminates the hazard). DMA loads V in parallel with MMA's QK computation.

## SMEM Budget

- Q: 128 × 128 × 2 = 32,768 bytes
- K: 2 × 64 × 128 × 2 = 32,768 bytes (double-buffered)
- V: 2 × 64 × 128 × 2 = 32,768 bytes (double-buffered, NEW)
- **Total: 98,304 bytes (96KB)** — fits in 99KB SM120 limit (1KB margin)

## Barrier Simplification

With V double-buffered, we can remove BAR_V_EMPTY entirely:
- BAR_K_FULL (1): DMA → MMA, K ready
- BAR_K_EMPTY (2): MMA → DMA, K consumed (still needed for K double-buffer)
- BAR_V_FULL (3): DMA → MMA, V ready

Three barriers instead of four.

## Expected Impact

- Eliminate 16.9M local memory spill requests → ~5% improvement
- V load overlaps with QK MMA → ~3% improvement (memory latency hidden)
- Combined: ~8% improvement, bringing Gen/FA4 from ~90% to ~95%+

## Files Modified

- `conf/fixtures/fa4/generated.cu` — kernel rewrite (inline + V double-buffer)

## Verification

- Must compile with 0 spill (or minimal spill < current 16.9M)
- Correctness: smoke test + relative benchmark
- NCU: tensor pipe > 87.5%, local mem spill near 0
