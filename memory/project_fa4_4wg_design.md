---
name: FA4 4-WG design — 3 consumers + 1 producer
description: 4-WG with BLOCK_Q=192 (3 consumer WGs × m64). All consumers same code path to satisfy compiler-injected warpgroup.arrive barriers.
type: project
---

## Why 4 WGs failed with WG3 on different code path
Compiler injects `warpgroup.arrive` barriers that ALL WGs must participate in.
If WG3 takes a different path (exit, spin-wait, or non-WGMMA code), it never reaches
the injected barrier points → DEADLOCK.

This is NOT setmaxnreg. Verified: deadlock persists even without setmaxnreg.
The simple 4-WG test (no WGMMA) works. WGMMA-specific barrier injection is the cause.

## Correct 4-WG Architecture: 1 Producer + 3 Consumers

512 threads = 4 WGs × 128:
- WG0: Producer (24 regs, TMA loads)
- WG1: Consumer for Q rows 0-63 (128 regs, no spills with IS_CAUSAL template)
- WG2: Consumer for Q rows 64-127
- WG3: Consumer for Q rows 128-191

BLOCK_Q = 192 (was 128). 3 consumers × m64 = 192 Q rows per block.

### SMEM Budget
- Q: 192 × 128 × 2 = 48KB (was 32KB)
- K×2: 64KB
- V×2: 64KB
- Total: 176KB + barriers < 232KB ✓

### Register Budget (no setmaxnreg needed!)
- 512 threads × 128 regs = 65536 ✓
- ptxas confirms: 128 regs, 0 spills with IS_CAUSAL template

### Changes from 3-WG
1. BLOCK_Q: 128 → 192
2. Q SMEM: 32KB → 48KB (3 × 16KB halves per Q_lo/Q_hi)
3. Each consumer WG offset: cwg * 8 * STRIDE (cwg = 0, 1, 2)
4. Block count: B*H*ceil(S/192) instead of B*H*ceil(S/128)
5. Causal max_kv_iter: adjust for BLOCK_Q=192
6. Inter-WG barriers: 3 consumer WGs instead of 2
7. Epilogue: 384 consumer threads write O to SMEM, then TMA S2G

### Key Insight
All consumers execute the SAME code path (including WGMMA). The compiler's
warpgroup.arrive injections are satisfied because ALL WGs reach the same
barrier points. No need for separate code paths.

### Risks
- BLOCK_Q=192 not a power of 2 — may cause Q tile boundary issues
- Fewer blocks per head (ceil(S/192) vs ceil(S/128))
- 3 consumer inter-WG barriers more complex than 2
- Q SMEM TMA box needs to be [64, 64] (3 loads per Q half instead of 2)
