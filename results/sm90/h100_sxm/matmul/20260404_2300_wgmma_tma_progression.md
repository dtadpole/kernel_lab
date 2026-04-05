# SM90 Matmul — WGMMA + TMA Progression

## Hardware
- NVIDIA H100 SXM5, GPU5, 132 SMs
- CUDA 12.8, torch 2.11+cu128

## Optimization Steps (all pass correctness 6/6 ✓)

| Step | Technique | 8192×8192 TFLOPS | vs cuBLAS | Commit |
|------|-----------|------------------|-----------|--------|
| 0 | Naive 32×32 tiled, FP32 | 9.1 | 1.2% | 4c45c61 |
| 1 | WMMA 64×64 | 22.5 | 3.0% | a51b864 |
| 2 | WMMA 128×128, double buffer | 45.3 | 6.1% | ffd8c23 |
| 3 | cp.async vectorized, TILE_K=32 | 69.0 | 9.3% | d474cb3 |
| 4 | WGMMA m64n128k16 foundation | 21.1 | 2.8% | 7ab52a6 |
| 5 | WGMMA 128×128 | 33.0 | 4.4% | 5f33424 |
| 6 | cp.async vectorized A loads | 48.2 | 6.5% | 8728cef |
| 7 | Pre-transpose B + vectorized | 234.6 | 31.8% | de4c2d0 |
| 8 | **TMA loads (current best)** | **452.2** | **61.1%** | eff3a56 |

cuBLAS reference: ~742 TFLOPS.

## Failed Attempts (all regressed)

| Attempt | Result | Why |
|---------|--------|-----|
| 128×256 tile (1WG) | 51 TF | C7511: insufficient registers for WGMMA pipeline |
| TILE_K=128 | 159 TF | Swizzle broken for multi-line rows |
| No-swizzle loads | 155 TF | WGMMA expects swizzled data |
| 4-stage pipeline | 360 TF | 128KB SMEM limits to 1 CTA/SM |
| Persistent scheduling | 303 TF | Per-tile acc zeroing + sync overhead |

## Key Learnings

1. **128B swizzle is mandatory** — WGMMA with layout_type=1 requires swizzled SMEM
2. **TMA eliminates load overhead** — 235→452 TF (1.9×) by removing address computation
3. **SMEM budget matters** — 2-stage (64KB) allows higher occupancy than 4-stage (128KB)
4. **Persistent scheduling hurts** without warp specialization — per-tile overhead dominates

## Next Steps

To reach cuBLAS level (700+ TF), need:
1. **Warp specialization**: 3 warpgroups (1 producer + 2 consumers)
2. **Larger tile**: 128×256 with 2 consumer warpgroups (register budget split)
3. **Producer uses setmaxnreg.dec** to release registers, consumers use setmaxnreg.inc
