# SM90 Matmul — Optimization Ceiling Findings

## Summary

After reaching 752 TFLOPS (102% cuBLAS) with warp specialization, four
additional micro-optimizations were attempted. All failed. The kernel has
reached its ceiling for the current architecture (3-WG cooperative, no
clusters, no TMA S2G).

## Failed Attempts

### 1. Cache-streaming stores (`st.global.cs`) — REGRESSED -5%
- **Hypothesis**: Bypass L2 write-allocate for output stores to free L2
  capacity for A/B tiles (C output = 128MB >> 50MB L2)
- **Result**: 701 TFLOPS (94.8% cuBLAS), down from 752
- **Learning**: H100's L2 write-coalescing is MORE valuable than cache-pollution
  avoidance. The L2 write path aggregates stores before DRAM write — bypassing
  it increases raw DRAM traffic

### 2. CTA swizzle group_m = 8 vs 16 — NO CHANGE
- **Hypothesis**: Different swizzle group size might improve L2 hit rate with
  persistent scheduling
- **Result**: Identical peak (750.2 TFLOPS) for both values
- **Learning**: At 8192×8192, the persistent scheduling dominates L2 behavior.
  CTA swizzle group_m is a second-order effect when tiles are assigned
  persistently

### 3. wait_group_1 pipelined empty signals — REGRESSED -7.5%
- **Hypothesis**: Allow 1 WGMMA group in-flight while signaling the previous
  stage's empty barrier — more compute-memory overlap
- **Result**: 693 TFLOPS (93.1% cuBLAS)
- **Learning**: With warp specialization, the producer runs independently and
  needs IMMEDIATE empty signals to stay ahead. Deferring signals by 1 K-tile
  starves the producer. This is opposite to the 2-WG cooperative kernel where
  wait_group_1 helped because TMA and WGMMA shared the same thread

### 4. 128 SM persistent grid — REGRESSED -3%
- **Hypothesis**: Power-of-2 SM count enables cleaner tile group alignment
  for CTA swizzle
- **Result**: 701 TFLOPS (94.7% cuBLAS) at 8192
- **Learning**: 4 idle SMs = 3% compute capacity wasted. The cleaner tile
  alignment at 4096 (512/128 = 4 exact) doesn't compensate for the capacity
  loss at 8192 (2048 tiles, 4 SMs idle)

## Current Performance

| Config | cuBLAS | CuTe DSL | Generated | Gen/cuBLAS |
|--------|--------|----------|-----------|------------|
| 256 | 1.8 | 2.0 | 1.9 | 107% |
| 512 | 13.5 | 14.5 | 13.3 | 99% |
| 1024 | 70.6 | 94.0 | 87.3 | 124% |
| 2048 | 360.3 | 479.3 | 471.4 | 131% |
| 4096 | 611.7 | 672.6 | 692.6 | 113% |
| **8192** | **735.9** | **736.5** | **751.8** | **102%** |

## What Would Be Needed for 808+ TFLOPS (109% cuBLAS)

Based on Pranjal Shankhdhar's fast.cu progression:

1. **Fix setmaxnreg** (~3-5%): Requires separating producer and consumer code
   into distinct compilation units so ptxas can assign different register
   counts. Currently ptxas allocates 154 regs uniformly (C7507 warning).

2. **Cluster multicast 2×1** (~4%): Share B tiles between vertically adjacent
   CTAs via TMA multicast. Reduces global memory traffic for B by 50%.
   Requires `__cluster_dims__`, cluster-level mbarrier coordination, and
   cluster-aware CTA swizzle.

3. **TMA S2G epilogue** (~2%): Replace per-thread global stores with TMA bulk
   store from SMEM. Requires either column-major C output or 3D TMA encoding
   for row-major (max 2D box is 128 elements).

4. **Hilbert curve scheduling** (~1%): Space-filling curve tile traversal for
   better L2 locality than linear swizzle.

5. **Ping-pong scheduling** (~5-10%): Overlap Consumer0's epilogue with
   Consumer1's MMA on the next tile. Requires OrderedSequenceBarrier
   coordination between consumers.
