# FA4 Flash Attention Optimization Analysis (H100 SM90)

**Date**: 2026-04-05
**GPU**: NVIDIA H100 (h8_4), 800 TFLOPS BF16 peak (Meta R&R SKU @ 650W)
**Baseline**: FA4 CuTe DSL v4.0.0b7, torch 2.11+cu130, cuDNN 9.19.0

## Benchmark Results (cold-L2, 20 trials)

| Config | cuDNN TFLOPS | FA4 TFLOPS | Ratio |
|--------|-------------|-----------|-------|
| mha-causal-b8-s4096 | 558.4 | 558.3 | 1.00x |
| mha-causal-b4-s8192 | 594.7 | 581.2 | 0.98x |
| mha-causal-b2-s16384 | 584.8 | 595.0 | 1.02x |
| mha-noncausal-b8-s4096 | 637.3 | 666.9 | 1.05x |
| mha-noncausal-b4-s8192 | 612.0 | 609.6 | 1.00x |
| mha-noncausal-b2-s16384 | 588.2 | 657.5 | 1.12x |

**Peak utilization**: cuDNN 79.7%, FA4 83.4% (cold-L2)

## Profiling (ncu, B=1 S=4096 H=16 D=128 noncausal)

### Key Metrics

| Metric | Value |
|--------|-------|
| Duration (warm) | 276.8 us (725 TFLOPS = 90.6% peak) |
| Duration (cold-L2) | ~330 us (657.5 TFLOPS = 82.2% peak) |
| Compute SM Throughput | 75.6% |
| Tensor (FP) Pipeline | 75.6% (bottleneck) |
| Memory Throughput | 49.2% |
| DRAM Throughput | 8.7% |
| L2 Hit Rate | 83.9% |
| Registers/Thread | 168 |
| SMEM/Block | 161 KB |
| Occupancy | 13.9% (1 block/SM) |
| No Eligible Warps | 59.9% |

### Warp Stall Breakdown

| Stall Reason | Cycles/Inst | Share |
|---|---|---|
| Stall Wait (async TMA/GMMA) | 1.45 | 40% |
| Long Scoreboard (memory dep) | 0.81 | 22% |
| Dispatch Stall (pipe contention) | 0.60 | 16% |
| Barrier (named barrier sync) | 0.43 | 12% |
| GMMA (tensor core wait) | 0.34 | 9% |

### Architecture (both FA4 ref and generated kernel)

- 3 warp groups: 1 producer (TMA, 24 regs) + 2 consumers (wgmma, 240 regs)
- Tiles: BLOCK_Q=128, BLOCK_KV=128, DIM=128
- Pipeline: 2-stage double-buffered K/V via TMA + mbarrier
- QK GEMM: wgmma.m64n128k16 RS (tnspB=1, K transposed)
- PV GEMM: wgmma.m64n64k16 RS (tnspB=0, V non-transposed)
- Overlap: QK[n] issued concurrently with PV[n-1] (intra-WG)
- Softmax: Online, overlaps with PV tensor core execution

## Performance Gap Analysis

```
Peak (800 TFLOPS)
  90.6% — warm cache (725 TFLOPS)
    9.4% structural overhead:
      4% — Stall Wait: async pipeline bubbles (QK<->PV transition)
      2% — Long Scoreboard: memory dependency chains
      1.5% — Dispatch: FP32 pipe contention (unfused softmax ops)
      1% — Barrier: inter-WG synchronization
      0.9% — GMMA: tensor core completion waits

  82.2% — cold L2 (657.5 TFLOPS)
    additional 8.4% from L2 cache misses on K/V tiles
```

## Optimization Opportunities (Prioritized)

### 1. L2 Promotion for K/V TMA (cold: +5-10%)

- **Current**: TMA uses `L2_PROMOTION_NONE`
- **Fix**: `CU_TENSOR_MAP_L2_PROMOTION_L2_128B` for K and V
- **Impact**: Close 10% gap between warm (725 TF) and cold (657.5 TF)
- **Status**: Enum exists in CuTe DSL (`TensorMapL2PromoKind`) but not exposed in `make_tiled_tma_atom`. Needs CUTLASS source modification or driver-level TMA descriptor override.

### 2. FP32 Softmax Instruction Fusion (warm: +2-3%)

- **Current**: 17M non-fused FP32 ops (ncu flagged "up to 33% FP32 improvement")
- **Fix**: Fold `softmax_scale_log2` into `exp2` argument via FMA:
  - Remove: `S_acc[i] *= scale` (64 MULs)
  - Replace: `exp2(S_acc[i] - nm)` → `exp2(__fmaf_rn(S_acc[i], scale, -nm))`
  - Saves 62 MUL + 64 SUB per KV iteration per WG
- **Status**: Implemented in generated kernel (but blocked by PV bug)

### 3. Q Register Preloading During PV Overlap (warm: +2-4%)

- **Current**: 8x ldmatrix_x4 inside QK GEMM (~240 issue cycles)
- **Fix**: Pre-load Q fragments during PV overlap window where FPU is idle after softmax
- **Impact**: Fill ~200 cycles of dead time, reduce QK issue latency
- **Risk**: +16 registers for Q double-buffer (168→184, still fits)

### 4. Asymmetric Inter-WG Barrier (warm: +1-2%)

- **Current**: `bar.sync` (symmetric) — both WGs stall every iteration
- **Fix**: `bar.sync own + bar.arrive other` (CuTe DSL style)
- **Impact**: Reduce barrier stall from 0.43 to ~0.2 cycles/inst

### 5. TMA Store for O Writeback (warm: +0.5-1%)

- **Current**: Per-thread 32-bit stores to global memory
- **Fix**: Write O to SMEM (reuse Q buffer), then TMA store

## Generated Kernel Bugs Found

The hand-written SM90 generated kernel (`data/generated/sm90/fa4/generated.cu`) has two pre-existing PV GEMM bugs:

### Bug 1: PV k-step stride direction

```c
// CURRENT (wrong): advances within a row (DIM/column direction)
uint64_t vd = make_wgmma_desc(vl + ks * 32, 0);

// CORRECT: advances across rows (KV/row direction) for tnspB=0
uint64_t vd = make_wgmma_desc(vl + ks * 16 * HALF_ROW_STRIDE, 0);
```

For tnspB=0 (non-transposed V), K dimension maps to rows. Each k-step of K=16 should advance by 16 rows * 128 bytes/row = 2048 bytes, not 32 bytes.

### Bug 2: P register packing mismatch

The QK GEMM uses `m64n128k16` which maps S_acc registers with 8-column group spacing (N=128 / 16 groups = 8 per group). The PV GEMM uses `m64n64k16` RS which expects A registers with 4-element K-position spacing (K=16 / 4 lanes = 4 per lane).

`PACK_P_REGS` converts from QK output layout to PV input layout, but the register permutation is incorrect — it interleaves groups of 8 instead of packing groups of 4.

**Fix required**: Remap P registers from QK's n128 output mapping to PV's k16 RS input mapping. Requires understanding both wgmma register conventions and implementing the correct permutation.

## Files

- Benchmark script: `bench_sm90_fa4.py`
- Benchmark configs: `data/fixtures/sm90/fa4/configs.json`
- Benchmark results: `data/bench_fa4_latest.json`
- Profiler trace: `data/fa4_profile.json`
- ncu report: `data/fa4_ncu_report.ncu-rep`
- Generated kernel (buggy): `data/generated/sm90/fa4/generated.cu`
- Baseline (worktree): `.claude/worktrees/fa4/data/generated/sm90/fa4/generated.cu`
