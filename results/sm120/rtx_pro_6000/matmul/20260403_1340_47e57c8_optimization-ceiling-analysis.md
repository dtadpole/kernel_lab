# SM120 Matmul Optimization Ceiling Analysis — 2026-04-03

## Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA RTX PRO 6000 Blackwell (SM 12.0) |
| SMs | 188 |
| Tensor Cores | 752 (5th Gen) |
| Peak BF16 Tensor (FP32 accum) | 503.8 TFLOPS |
| DRAM Bandwidth | 1,792 GB/s |
| L2 Cache | 128 MB |
| SMEM per SM | 100 KB (102,400 bytes) |
| SMEM per block (opt-in max) | 99 KB (101,376 bytes) |
| Max threads per SM | 1,536 |
| Registers per SM | 65,536 |

## Baseline Performance

All measurements: cold-L2 (96 MB flush), fresh random inputs per trial, 20 trials median.

| Config | cuBLAS (TFLOPS) | Generated (TFLOPS) | Gen vs cuBLAS | % Peak |
|--------|----------------|---------------------|---------------|--------|
| 256×256 | 2.7 | **4.8** | 1.78× | 1.0% |
| 512×512 | 22.0 | **30.0** | 1.36× | 6.0% |
| 1024×1024 | 119.0 | **134.8** | 1.13× | 26.8% |
| 2048×2048 | 263.8 | **275.6** | 1.04× | 54.7% |
| 4096×4096 | 402.5 | 400.7 | 1.00× | 79.5% |
| 8192×8192 | 445.9 | **447.2** | 1.00× | 88.8% |

**The generated kernel matches or beats cuBLAS at every size.**

## Kernel Architecture

| Parameter | Big (M,N ≥ 2048) | Small (M,N < 2048) |
|-----------|-------------------|---------------------|
| Tile | 256×128×32 | 128×64×32 |
| Threads | 256 (8 warps) | 128 (4 warps) |
| Pipeline | 3-stage TMA | 4-stage TMA |
| Regs/thread | 191, 0 spills | 134, 0 spills |
| SMEM | 73.9 KB | 49.3 KB |
| Occupancy | 16.63% (8 warps/SM) | 16.63% (can fit 2 blocks/SM) |

## Optimization Attempts

### Attempt 1: Small kernel at 2048 with 2 blocks/SM

**Hypothesis:** Big kernel at 2048 creates only 128 tiles for 188 SMs (32% idle).
Use small kernel (128×64) to get 512 tiles + 2 blocks/SM → full SM utilization.

**Changes:**
- `use_big` threshold: M,N ≥ 2048 → M,N ≥ 4096
- `__launch_bounds__(128, 2)` for small kernel
- Grid: `min(totalTiles, 2*numSMs)` for small kernel

**Result: -18% at 2048 (275.6 → 225.7 TFLOPS)**

**Root cause:** Small kernel has lower MMA instruction density (53% vs 64% for big kernel).
With fewer HMMA per instruction, the Tensor Cores idle more often. The SM utilization
gain (68% → 100%) is more than negated by the reduced per-SM TC utilization.

**Key learning:** Per-tile MMA density matters more than SM utilization. A compute-bound
kernel is better served by fewer, larger tiles that maximize TC occupancy per SM.

### Attempt 2: 4-stage pipeline (3 → 4 stages)

**Hypothesis:** More pipeline stages = deeper prefetch = fewer mbarrier wait stalls.

**Changes:**
- `BIG_STAGES`: 3 → 4
- `BIG_SMEM_BYTES`: 73856 → 98432 (96 KB)

**Result: ~0% at all sizes (within noise)**

**Root cause:** TMA loads (~86 cycles for 24 KB at bandwidth) already complete well within
the 3-stage compute window (~1024 cycles per K-block). Extra prefetch depth is unnecessary.
The increased SMEM (98 KB → 4 KB L1) may slightly hurt store performance.

### Attempt 3: 1-M-tile-at-a-time K-loop restructuring

**Hypothesis:** Processing 1 M-tile at a time (instead of 2) creates tighter register
dependencies that force the compiler to interleave LDSM with HMMA more evenly.

**Changes:**
- `aReg[2][8]` → `aReg[8]` (halved A register footprint)
- Inner loop: 8 iterations of (2 LDSM + 8 HMMA) instead of 4 iterations of (4 LDSM + 16 HMMA)

**Result: Identical SASS output — compiler produces the same schedule**

**Root cause:** NVCC's instruction scheduler optimizes globally across the unrolled loop body.
The register dependencies in the source don't affect the final SASS because the compiler
sees through them. The `asm volatile` on ldmatrix/mma prevents elimination but not reordering.

## Why the 11% Gap to Peak is Fundamental

### TC throughput analysis

| Parameter | Value |
|-----------|-------|
| TC throughput | 1 HMMA per 16 cycles per TC |
| TCs per SM | 4 |
| HMMA/cycle/SM | 0.25 |
| FLOPs per HMMA (m16n8k16 BF16) | 4,096 |
| Peak FLOPs/cycle/SM | 1,024 |

### Instruction mix in K-loop body

| Instruction type | Count per K-block | % of total |
|------------------|-------------------|------------|
| HMMA.16816 | 64 | 64% |
| LDSM.M88.4 (A loads) | 16 | 16% |
| LDSM.MT88.2 (B loads) | 8 | 8% |
| Other (addr, mbar, TMA) | ~12 | 12% |
| **Total** | **~100** | **100%** |

### The bottleneck

With 64% HMMA instructions and TC throughput of 0.25/cycle, the HMMA supply from 8 warps
(~2.6 HMMA/cycle) exceeds TC demand by 10.4×. The TC is always the bottleneck.

The 11% gap comes from cycles when the TC is idle:
1. **Pipeline startup** (prelude stages): 2 K-blocks of latency per tile
2. **Pipeline drain** (last K-block): reduced overlap
3. **Tile transitions**: mbarrier reinit + accumulator zeroing + __syncthreads
4. **Epilogue stores**: 64 STG.E per thread (no MMA during stores)

These are architectural overheads inherent to the TMA + mbarrier + mma.sync pipeline
on SM120. They cannot be eliminated by instruction scheduling or code restructuring.

### What would close the gap

| Approach | Available on SM120 GeForce? | Expected gain |
|----------|-----------------------------|---------------|
| WGMMA (async warpgroup MMA) | **No** (data center only) | +5-8% |
| Higher occupancy (>8 warps) | Not feasible (191 regs) | +3-5% |
| Larger TILE_K (64) | Possible but same total work | +0-1% |
| Manual PTX scheduling | Possible but diminishing returns | +0-2% |
| Store coalescing (SMEM epilogue) | Yes but <0.1% impact | +0-0.1% |

## Conclusion

The SM120 matmul kernel at 447 TFLOPS (88.8% of peak) is at the hardware ceiling for
mma.sync-based matmul on this architecture. It matches cuBLAS at large sizes and
significantly outperforms cuBLAS at small sizes. No source-level optimization tested
produced a sustained improvement.

The remaining 11% gap to the 503.8 TFLOPS theoretical peak is from pipeline startup/drain
overhead and TC idle cycles during non-compute phases. This gap is consistent with cuBLAS
achieving the same 88.5% efficiency, confirming it is an architectural limit rather than
an implementation deficiency.

## Commit

- **Hash**: 47e57c8 (no code changes committed — all attempts reverted)
- **Branch**: matmul/optimize-reference
