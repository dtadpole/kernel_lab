# Matmul Reference (CuTe DSL) Optimization — atom_layout (4,2,1)

**Date:** 2026-04-03
**Status:** Implemented, awaiting SM120 verification
**Branch:** worktree-matmul

## Problem Statement

The CuTe DSL reference kernel reaches 80.7% of peak (406 TFLOPS) at 8192×8192
on RTX PRO 6000, while cuBLAS reaches 88.2% (444 TFLOPS) — a 9.3% gap.

## Data-Driven Root Cause Analysis

### NCU Profile Comparison (4096×4096, from 20260402 results)

| Metric | Reference | cuBLAS | Gap | Root Cause |
|--------|-----------|--------|-----|------------|
| **Occupancy** | **10.41%** | **16.65%** | **37% fewer warps** | **5 warps vs 8** |
| Active warps/SM | 4.99 | 7.99 | -37% | DMA warp + only 4 MMA warps |
| Memory throughput | 71.3% | 55.4% | +29% | More SMEM accesses per FLOP |
| Compute throughput | 84.2% | 86.7% | -3% | Less latency hiding |
| Registers/thread | 207 | 218 | -5% | 128 fp32 accumulators/thread |
| SMEM | 99.3 KB | 73.7 KB | +35% | TILE_K=64 vs 32 |

### Bottleneck Ranking

1. **Occupancy (37% gap)** — Structural: warp specialization with only 4 MMA warps
2. **Tile size (128×128 vs 256×128)** — Blocked by register spill with larger tiles
3. **Memory intensity (71% vs 55%)** — TILE_K=64 loads more data per stage
4. **Compute efficiency (84% vs 87%)** — Consequence of lower occupancy

### Constraints Checked

- TILE_K > 64: BLOCKED (mbarrier 3-TMA limit on SM120, documented)
- Tile 256×128 with 4 warps: BLOCKED (255 regs + spill, NCU verified)
- Tile 128×256 with 100KB SMEM: BLOCKED (only 1 mainloop stage = no pipeline)
- Occupancy > 1 block/SM: BLOCKED by SMEM (99KB > 50% of 100KB default)

## Optimization: Double MMA Warps

### Change

```python
# dense_gemm.py, Sm120GemmKernel.__init__()

# BEFORE
self.atom_layout = (2, 2, 1)     # 4 MMA warps
self.mma_register_requirement = 232

# AFTER
self.atom_layout = (4, 2, 1)     # 8 MMA warps
self.mma_register_requirement = 168
```

### How atom_layout Works

`atom_layout = (M, N, K)` distributes MMA warps across the tile dimensions:
- `num_mma_warps = M × N × K`
- `permutation_mnk = (M×16, N×8×2, K×16)` — MMA coverage per step
- Tile iterations: `tile_shape / permutation_mnk` per dimension

With (4,2,1):
- `permutation_mnk = (64, 32, 16)`
- Tile 128×128×64: M=128/64=2, N=128/32=4, K=64/16=4 iterations ✓
- 8 MMA warps + 1 DMA warp = 9 warps = 288 threads

### Expected Effects

| Metric | Before (2,2,1) | After (4,2,1) | Change |
|--------|----------------|---------------|--------|
| MMA warps | 4 | 8 | +100% |
| Total warps/block | 5 | 9 | +80% |
| Threads/block | 160 | 288 | +80% |
| Theoretical occupancy | 10.42% | 18.75% | +80% |
| Accumulators/thread | 128 fp32 | 64 fp32 | -50% |
| Estimated regs/thread | 207 | ~148 | -28% |
| SMEM | 99.3 KB | 99.3 KB | 0% |
| Mainloop stages | 2 | 2 | 0% |

### Why This is the Top Direction

1. **Directly addresses #1 bottleneck** (occupancy gap)
2. **Zero SMEM cost** — warp count doesn't affect shared memory
3. **Reduces register pressure** — more warps = fewer accumulators per thread
4. **Fully parametric** — CuTe DSL tiled_mma, ldmatrix, stmatrix, pipeline
   all adapt automatically via atom_layout
5. **No hardware constraint violations** — stays within SM120 limits

### What Didn't Change

- Tile shape: 128×128×64 (unchanged)
- TILE_K: 64 (unchanged)
- Pipeline stages: 2 mainloop + 8 epilogue (unchanged)
- Warp specialization architecture: still 1 DMA + N MMA warps
- TMA load/store operations (unchanged)
- Epilogue: stmatrix R2S + TMA S2G (unchanged)
- Persistent tile scheduling (unchanged)

## Expected Performance

### Optimistic (10% improvement)
| Config | Before (ms) | After (ms) | TFLOPS | %peak |
|--------|------------|-----------|--------|-------|
| 4096×4096 | 0.3436 | 0.309 | 444.8 | 88.3% |
| 8192×8192 | 2.7064 | 2.436 | 451.3 | 89.6% |

### Conservative (5% improvement)
| Config | Before (ms) | After (ms) | TFLOPS | %peak |
|--------|------------|-----------|--------|-------|
| 4096×4096 | 0.3436 | 0.326 | 421.5 | 83.7% |
| 8192×8192 | 2.7064 | 2.571 | 427.7 | 84.9% |

## Verification Plan

1. **Correctness**: `python benchmark.py --sizes 256 512 1024 2048 4096 8192`
   - All configs must PASS (allclose atol=1e-2, rtol=1e-2 vs cuBLAS)

2. **Performance**: Same benchmark outputs latency + TFLOPS + %peak

3. **NCU Profile**: `ncu --set detailed python cutedsl.py` with 4096×4096
   - Check: occupancy (expect ~18%), registers (expect ~148), SMEM (expect ~99KB)
   - Compare warp stall profile against previous run

4. **Regression**: Verify no regression at small sizes (256-1024)

## Future Directions (if this works)

1. **128×256×64 tile with 228KB SMEM**: Override `smem_capacity` to use full
   SM120 capacity, enabling 4 mainloop stages with the larger tile
2. **Occupancy=2**: If register pressure drops enough, try 2 blocks/SM
   (requires SMEM < 50KB per block — would need smaller tile)
3. **Better instruction scheduling**: Interleave ldmatrix with MMA in the
   K-loop inner body (requires CuTe DSL kernel code restructuring)

## Files Changed

- `data/fixtures/sm120/matmul/dense_gemm.py` — atom_layout (2,2,1)→(4,2,1), register hint 232→168
- `data/fixtures/sm120/matmul/cutedsl.py` — Docstring update
- `data/fixtures/sm120/matmul/benchmark.py` — New benchmark/verification script
