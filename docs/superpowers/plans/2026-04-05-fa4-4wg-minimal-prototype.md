# FA4 4-WG Minimal Prototype Plan

## Objective
Get a 4-WG (512 thread) FA kernel running correctly on SM90.

## Root Cause of Previous Failures
SM90 nvcc injects `warpgroup.arrive` for ALL WGs in a CTA with WGMMA.
WGs that don't execute WGMMA deadlock because they never reach barrier injection points.

## Solution: All 4 WGs Execute WGMMA (Same Consumer Code)

### Architecture
```
WG0 (128 threads): Producer + then join as Consumer for Q-rows 0-63
WG1 (128 threads): Consumer for Q-rows 0-63 (mirrors WG0's consumer work) 
WG2 (128 threads): Consumer for Q-rows 64-127
WG3 (128 threads): WG3 does the SAME as WG2 (duplicate, output discarded)
```

Wait — this wastes 25% compute. Better approach:

### Better: BLOCK_Q=128, 3 Consumers × m64 = Not Possible (3×64=192≠128)

### Actually Best: Don't Fight the Compiler

The simplest prototype that satisfies the compiler:
1. ALL 4 WGs execute the consumer path (including WGMMA)
2. WG0 does TMA loads FIRST, then switches to consumer role
3. Each WG handles 64 Q rows → total 256 Q rows → BLOCK_Q=256
4. Q SMEM = 256 × 128 × 2 = 64KB, K×2=64KB, V×2=64KB → 192KB < 232KB ✓

But BLOCK_Q=256 is a huge tile — might have causal mask inefficiency.

### Simplest Possible Prototype

BLOCK_Q=128, 2 real consumers (WG1, WG2), WG0 and WG3 are "shadow consumers" 
that execute the SAME consumer code but write O to a dummy SMEM region.

Steps:
1. All 4 WGs do __syncthreads() for mbarrier init
2. All 4 WGs do consumer code (including WGMMA → compiler barriers satisfied)
3. WG0: producer loads Q/K/V BEFORE entering consumer code
4. WG3: mirrors WG2's consumer work (same Q rows 64-127), O goes to dummy SMEM
5. Only WG1 and WG2 write the real O output

This wastes 50% compute (WG0 and WG3 do unnecessary WGMMA) but:
- Compiler barriers satisfied (all WGs do WGMMA) ✓
- No BLOCK_Q change needed ✓
- No SMEM layout change ✓
- Register budget: 128 regs, 0 spills (confirmed) ✓

### Implementation Steps
1. Copy V3 kernel
2. Change __launch_bounds__(384→512), TB_SIZE=512
3. Remove setmaxnreg (not needed with 128 regs, 0 spills)
4. WG0: do producer TMA loads, then FALL THROUGH to consumer code (cwg=0, same as WG1)
5. WG3: execute consumer code with cwg=2 (same as WG2) — write O to dummy
6. Only WG1's and WG2's O output is used (WG0 and WG3 write to Q_lo/Q_hi which gets overwritten)
7. Epilogue: only WG1 and WG2 write O to Q SMEM for TMA S2G

### After Prototype Works
- Profile: verify more eligible warps per scheduler
- If faster: optimize to reduce wasted compute (WG0/WG3 only do critical path WGMMA)
- If same/slower: the scheduler bottleneck theory was wrong, abandon 4-WG on SM90
