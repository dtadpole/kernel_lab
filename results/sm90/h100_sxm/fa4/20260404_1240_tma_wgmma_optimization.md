# FA4 TMA + WGMMA Optimization — SM90 H100 SXM

**Date:** 2026-04-04 12:40 PDT
**Host:** devvm8490 (h8_3), GPU 4, CUDA 12.8
**Branch:** worktree-FA4

## Summary

Replaced `cp.async.cg.shared.global` data loads with TMA (`cp.async.bulk.tensor.2d`)
hardware-accelerated tensor loads. TMA produces SWIZZLE_128B layout natively,
eliminating all software address swizzle computation (21% of baseline instructions
were IMAD/LOP3 for swizzle).

## Architecture Changes

| Aspect | Before (cp.async) | After (TMA) |
|--------|-------------------|-------------|
| Data load | `cp.async.cg.shared.global` (all 128 threads) | `cp.async.bulk.tensor.2d` (thread 0 only) |
| SMEM layout | Software B128 swizzle in `global_to_shared_B128()` | Hardware SWIZZLE_128B via TMA descriptor |
| Sync | `cp.async.commit_group` / `wait_group` | mbarrier (`init`, `arrive_expect_tx`, `try_wait.parity`) |
| DIM handling | Single 128-col tile with atom-based swizzle | Split-DIM: 2×64-col TMA loads per tile |
| Registers | 189 | 128 (-32%) |
| Spills | 0 | 0 |
| SMEM | 64 KB | 64 KB + 32B (mbarriers) |

### Key Implementation Details

- **Split-DIM**: DIM=128 split into 2×64 for SWIZZLE_128B (`boxDim[0] * 2B = 128B`)
- **TMA descriptors**: Created host-side via `cuTensorMapEncodeTiled`, one per (batch, head)
- **mbarrier phase tracking**: Double-buffered K uses separate mbarriers with phase toggling
- **PV GEMM**: Split into two m64n64k16 RS GEMMs (one per V half)
- **QK k-stepping**: 8 k-steps split as 4 from lo-half + 4 from hi-half

## Performance Results

Measured with eval_harness (cold-L2, fresh inputs per trial, 5 warmup + 10 timed runs):

| Config | Baseline (TFLOPS) | TMA (TFLOPS) | Speedup |
|--------|-------------------|--------------|---------|
| causal-b8-s4096 | 365.7 | 358.8 | 0.98x |
| causal-b4-s8192 | 392.9 | 407.4 | 1.04x |
| causal-b2-s16384 | 407.6 | 442.6 | 1.09x |
| noncausal-b8-s4096 | 427.5 | 446.9 | 1.05x |
| noncausal-b4-s8192 | 448.8 | 492.7 | 1.10x |
| noncausal-b2-s16384 | 448.9 | 514.1 | 1.15x |
| **Average** | **415.2** | **443.8** | **1.07x** |
| % of H100 peak | 42.0% | 44.8% | |

### Improvement Scales with Sequence Length

TMA benefit increases with problem size because:
1. Descriptor setup cost is amortized over more KV iterations
2. TMA eliminates per-element address computation in the inner loop
3. Thread 0 handles all loads → 127 threads fully dedicated to compute

## What Worked

1. **TMA hardware swizzle**: Eliminates 21% of instructions (IMAD/LOP3 address swizzle)
2. **Register reduction**: 189→128 regs (-32%) from removing address computation variables
3. **Single-thread TMA**: Frees 127 threads from data movement overhead
4. **mbarrier async**: Proper pipelining of V[i] and K[i+1] loads with QK+softmax compute

## What Didn't Work (Iterations)

1. Initial v_phase=1 caused mbarrier deadlock (fixed to v_phase=0)
2. Service cuDNN SDPA broken post-driver-update (bypassed, ran harness directly)

## Remaining Gap

- Current: 444 TFLOPS avg (44.8% of peak)
- CuTe DSL: ~700 TFLOPS (71% of peak)
- Gap: ~1.6x

### Root Causes of Remaining Gap

| Source | Impact | Fix |
|--------|--------|-----|
| 1 WG vs 2 WGs | ~1.1x | Add second MMA warp group |
| tile 64×64 vs 128×128 | ~1.1x | Increase tile sizes |
| Split-DIM PV overhead | ~1.05x | Use m64n128k16 with contiguous V layout |
| No inter-WG overlap | ~1.1x | Producer/consumer warp group pipelining |
