# SM90 Matmul — WMMA Baseline Progression

## Hardware
- NVIDIA H100 SXM5, GPU5
- CUDA 12.8, torch 2.11+cu128
- Peak BF16 dense: ~800 TFLOPS

## Optimization Steps (all pass correctness 6/6 ✓)

| Step | Technique | 8192×8192 TFLOPS | vs cuBLAS | Commit |
|------|-----------|------------------|-----------|--------|
| 0 | Naive 32×32 tiled, FP32 scalar | 9.1 | 1.2% | 4c45c61 |
| 1 | WMMA 64×64, 4 warps | 22.5 | 3.0% | a51b864 |
| 2 | WMMA 128×128, 8 warps, double buffer | 45.3 | 6.1% | ffd8c23 |
| 3 | cp.async vectorized loads, TILE_K=32 | 69.0 | 9.3% | d474cb3 |

cuBLAS reference: 744 TFLOPS at 8192×8192.

## Failed Attempts

### 256×128 tile, 16 warps — NO IMPROVEMENT (63 TF)
- Larger tile didn't help: more warps compete for SMEM bandwidth
- 16 warps × 32 threads = 512 threads — too many for the compute density

## Analysis

The WMMA API (nvcuda::wmma) is capped at ~70-100 TFLOPS on H100 because:
- WMMA uses `mma.sync` (16×16×16) — 4 warp-level instructions needed per warpgroup-level WGMMA
- Load/store overhead from fragment register shuffling
- Cannot overlap compute with async loads at instruction level

## Next Steps (not yet implemented)

To reach cuBLAS-level (700+ TFLOPS), need SM90's native instructions:
1. **WGMMA** (`wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16`) — warpgroup MMA
2. **TMA** (`cp.async.bulk.tensor`) — tensor memory accelerator for async loads
3. **mbarrier** pipeline — multi-stage K-loop with barrier-based synchronization
4. **Warp specialization** — producer warpgroup (loads) + consumer warpgroup (compute)
5. **Persistent scheduling** — grid-stride loop over tiles
