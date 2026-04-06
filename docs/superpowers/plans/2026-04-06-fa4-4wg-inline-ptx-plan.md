# FA4 4-WG via Inline PTX — Implementation Plan

## Breakthrough Discovery
Inline PTX `.reg .f32 r<N>` can declare registers beyond __launch_bounds__ limit.
After `setmaxnreg.inc 232`, inline PTX registers map to physical regs 0-231.
Verified: r0, r100, r190 all work correctly at `__launch_bounds__(512)` base=128.

## Architecture (matching AVO)
```
WG0 (128 threads): Producer — TMA loads, 24 regs (setmaxnreg.dec 24)
WG1 (128 threads): Consumer A — QK+softmax+PV, 192 regs (setmaxnreg.inc 232)
WG2 (128 threads): Consumer B — QK+softmax+PV, 192 regs (setmaxnreg.inc 232)
WG3 (128 threads): Correction — O rescale + TMA S2G, 24 regs (setmaxnreg.dec 24)
```

Budget: 24×128 + 232×128 + 232×128 + 24×128 = 65,536 ✓

## Key Technique
Consumer mainloop written in **inline PTX asm** blocks:
- Declare `.reg .f32 O<64>;` (O_acc)
- Declare `.reg .f32 S<64>;` (S_acc)
- Declare `.reg .u32 P<32>;` (P_packed)
- WGMMA instructions directly in PTX
- Softmax (exp2, fma, shfl) directly in PTX
- Total: ~192 PTX registers → ptxas maps to physical regs after setmaxnreg

This BYPASSES nvcc's register allocator. The compiler sees the asm block
as a black box and doesn't try to spill.

## Implementation Steps

### Phase 1: Prove concept with simple wgmma in PTX
- [x] Verified: inline PTX r<200> works after setmaxnreg.inc
- [ ] Test: wgmma.mma_async in inline PTX with r<192> outputs
- [ ] Test: full QK GEMM (8 wgmma calls) in inline PTX

### Phase 2: Write consumer mainloop in PTX
- [ ] QK GEMM: 8× wgmma m64n128k16 SS → O<64> output registers
- [ ] Softmax: exp2, fma, max reduction, shfl, sum reduction
- [ ] P pack: cvt.rn.bf16x2.f32
- [ ] PV GEMM: 8× wgmma m64n128k16 RS → S<64> output registers  
- [ ] O rescale: fmul
- [ ] Barrier protocol: mbarrier, named barriers

### Phase 3: Integrate into FA4 kernel
- [ ] CUDA kernel shell: mbarrier init, __syncthreads, setmaxnreg, WG routing
- [ ] Producer path: TMA loads (CUDA code, few regs)
- [ ] Consumer path: inline PTX mainloop
- [ ] Correction path: bar.sync + TMA S2G (CUDA code, few regs)
- [ ] Epilogue: O normalize + SMEM write + TMA store

### Phase 4: Benchmark and optimize
- [ ] Verify correctness on all 6 configs
- [ ] NCU profile: verify 0 spills, check warp cycles/instr
- [ ] Compare with 3-WG V3 baseline (548 TF)
- [ ] Optimize: QK/PV overlap, softmax scheduling

## Expected Performance
- 16 wgmma/iter (same as 3-WG, n128 QK+PV)
- 0 spills (192 regs > 168 needed)
- 16 active warps (vs 12 for 3-WG) → better scheduling
- Target: 550-650 TFLOPS (matching or exceeding FA4 DSL)

## Risk
- Inline PTX is hard to write and debug (~500 lines of PTX)
- wgmma register mapping in PTX might differ from CUDA
- ptxas might still add warpgroup.arrive barriers that interfere
