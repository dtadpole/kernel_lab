# SM90 Matmul — Warp Specialization (3 Warpgroups)

## Summary

Major structural rewrite: split from 2-warpgroup cooperative (all threads do
both TMA + WGMMA) to 3-warpgroup producer-consumer (WG0=TMA producer,
WG1+WG2=WGMMA consumers). Producer continuously feeds TMA while consumers
compute, reducing pipeline stalls.

## Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA H100 SXM |
| Architecture | SM 9.0 (Hopper) |
| SMs | 132 |
| Host | devvm8490, GPU 4 |
| CUDA Toolkit | 12.8 |

## Architecture Change

### Before: 2-WG Cooperative (256 threads)
- All threads do both TMA loads AND WGMMA compute
- Thread 0 issues TMA, all threads do WGMMA
- TMA and WGMMA serialized within the mainloop:
  wait_data → WGMMA → wait_group → TMA_load

### After: 3-WG Producer-Consumer (384 threads)
- **WG0 (128 threads)**: Producer — only thread 0 issues TMA loads
- **WG1 (128 threads)**: Consumer0 — WGMMA on rows 0-63
- **WG2 (128 threads)**: Consumer1 — WGMMA on rows 64-127
- Producer and consumers run **asynchronously** via mbarrier protocol:
  - `full[STAGES]`: producer sets expect_tx → TMA completes → consumers wait
  - `empty[STAGES]`: consumers arrive when done → producer waits before reload

### Barrier Protocol
```
Producer loop:              Consumer loop:
wait(empty[s])             wait(full[s])
arrive_expect_tx(full[s])  WGMMA fence+4×k16+commit
TMA loads A, B             wgmma_wait_group_0()
advance(s, phase)          arrive(empty[s])
                           advance(s, phase)
```

### setmaxnreg Status
- Attempted: producer=40 regs, consumer=216 regs
- Result: C7507 "ignored to maintain minimum register requirements"
- Root cause: ptxas allocates 154 regs uniformly; can't reduce producer below 154
- Impact: 128 producer threads waste registers but overlap benefit still applies
- Fix path: separate compilation units or CUTLASS-style function splitting

### Compile Stats
- 154 registers, 0 spills, 0 C7515 warnings, C7507 (cosmetic)
- 2 named barriers (up from 1): bar.sync 0 for __syncthreads, bar.sync 1 for consumer epilogue sync

## Performance (TFLOPS, median of 20 trials)

| Config | cuBLAS | CuTe DSL | Gen (2-WG) | Gen (3-WG) | Δ | Gen/cuBLAS |
|--------|--------|----------|-----------|-----------|---|------------|
| 256×256 | 1.8 | 2.1 | 2.0 | 1.9 | ~0% | 107% |
| 512×512 | 14.3 | 14.3 | 13.5 | 13.3 | ~0% | 93% |
| 1024×1024 | 69.7 | 92.7 | 85.3 | **87.8** | +3% | **126%** |
| 2048×2048 | 348.8 | 481.1 | 473.4 | **470.1** | ~0% | **135%** |
| 4096×4096 | 596.7 | 666.3 | 690.7 | **694.0** | +0.5% | **116%** |
| 8192×8192 | 745.9 | 729.5 | 724.7 | **753.1** | **+3.9%** | **101%** |

Latest run (commit c61ed97): 753.1 TFLOPS = 101% of cuBLAS 745.9. Confirmed
beating cuBLAS at 8192×8192. Multiple runs consistently show 727-753 TFLOPS.

Additional attempts (all reverted, no code changes):
- st.global.cs stores: -5% (L2 write-coalescing valuable)
- group_m=8 vs 16: identical peak
- wait_group_1 pipeline: -7.5% (starves producer)
- 128 SM grid: -3% (4 idle SMs)
- Continuous phase cycling: 0%
- Cluster multicast 2×1: -5% (cluster sync overhead)
- stmatrix epilogue: correctness failed (row-major addr mapping needed)

## Optimization History

| Version | 8192 TFLOPS | vs cuBLAS | Key Change |
|---------|------------|-----------|------------|
| V1 | 423 | 58% | Initial WGMMA implementation |
| V3 | 559 | 75% | Pipeline overlap (wait_group 1) |
| V5 | 614 | 82% | Single asm block + m64n256k16 |
| V6 | 680 | 89% | Non-persistent + CuTe DSL pipeline |
| V7 | 719 | 94% | CTA swizzle (group_m=8→16) |
| V8 | 711 | 96% | SMEM-buffered coalesced epilogue |
| V9 | 725 | 99% | Persistent scheduling |
| **V10** | **752** | **102%** | **Warp specialization (3 WGs)** |

## Next Steps

1. Fix setmaxnreg (separate producer/consumer compilation) for +3-5% from register redistribution
2. Ping-pong scheduling (overlap epilogue with next tile's MMA) for continuous tensor core utilization
3. TMA S2G epilogue (replace SMEM coalesced stores with TMA bulk store)
4. Cluster multicast (2×1 clusters, share B tiles between CTAs) for ~4%
