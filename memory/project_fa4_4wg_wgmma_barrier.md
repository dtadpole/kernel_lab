---
name: FA4 4-WG — WGMMA works on SM90, barrier protocol is the real issue
description: 4-WG WGMMA confirmed working on SM90 (512 threads, all WGs execute WGMMA). FA4 deadlock is from barrier protocol, not hardware.
type: project
---

## CONFIRMED: 4-WG WGMMA Works on SM90 (2026-04-05)

Minimal test: 512 threads, 4 WGs, all execute wgmma_ss_m64n64k16.
Result: correct (all WGs produce 16.0). 58 regs, 1 barrier, 0 spills.

**SM90 hardware fully supports 4+ warp groups with WGMMA.**
Previous deadlocks were ALL from barrier protocol bugs in the FA4 kernel.

## FA4 4-WG Barrier Issues (all my bugs, not hardware)

1. **cwg=2 for WG3** → accessed out-of-bounds Q rows 128-191 (BLOCK_Q=128)
2. **EPILOGUE_BAR=1 conflict** with other_bar=1 (for cwg=2)
3. **setmaxnreg freed ≠ acquired** → deadlock (inc 240 > freed budget)
4. **3-consumer barrier count** → 2-bar asymmetric pattern doesn't prime correctly for 3 WGs
5. **Removing barrier code** → compiler dropped warpgroup.arrive injection (3 barriers instead of 16)

## Correct 4-WG Approach for FA4

Key constraints:
- ALL consumer WGs must execute the SAME WGMMA code path
- WG3 should shadow WG1 or WG2 (cwg mapping)
- Named barrier thread counts must account for 3 consumer WGs (384 threads)
- The 2-bar asymmetric arrive/sync pattern needs mma_init priming for BOTH bars
- Keep ALL barrier code in place (don't remove arrive lines — compiler needs them for warpgroup.arrive)

## Register Budget
With IS_CAUSAL template: 128 regs/thread, 0 spills, 16 barriers.
512 threads × 128 regs = 65536 ✓. No setmaxnreg needed.
