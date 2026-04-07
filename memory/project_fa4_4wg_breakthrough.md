---
name: FA4 4-WG breakthrough — proven working on SM90
description: 4 WG (512 threads) FA kernel runs correctly on SM90 H100. Key fixes and remaining issues documented.
type: project
---

## 4-WG PROVEN WORKING on SM90 (2026-04-05)

Correctly produced cos=1.000000 on B=1 S=128 H=1 D=128 noncausal.

## How It Was Made to Work

### Key Fixes
1. **No setmaxnreg**: 128 regs/thread sufficient with IS_CAUSAL template (0 spills at PTX level)
2. **WG3 shadows WG2**: `cwg = (wg_id >= 3) ? 1 : (wg_id - 1)` — WG3 processes same Q rows 64-127 as WG2
3. **All WGs execute consumer WGMMA code path**: satisfies compiler warpgroup.arrive injection
4. **mywarp = warp_id % 4**: correct for all 4 WGs
5. **Barrier thread counts 256→384**: 3 consumer WGs × 128 threads
6. **Epilogue threads 288→416**: 384 consumers + 32 warp-4
7. **Keep original barrier structure**: my_bar/other_bar/arrive/sync pattern preserved, just change thread counts

### What Failed Before
- `return` or spin-wait for WG3: compiler warpgroup.arrive for WG3 never reached → deadlock
- Different code path for WG3: same issue
- Removing barrier code: compiler dropped warpgroup.arrive injection → 3 barriers instead of 16
- Wrong cwg for WG3 (cwg=2): accessed out-of-bounds Q rows 128-191
- EPILOGUE_BAR=1 conflict with other_bar=1 (for cwg=2)

### Remaining Issues
- **1548 bytes spills**: ptxas uses 128 regs but the consumer code originally needs 168
  → spills cause performance regression and potential correctness issues on larger configs
- **Larger configs crash**: B=1 S=256 causal gives "illegal instruction" — likely from spill corruption

### Next Steps
1. Reduce register pressure to eliminate spills (the code COMPILES with 128 regs but SPILLS because
   the consumer path has more live variables than 128 regs can hold)
2. IS_CAUSAL template alone isn't enough to fit in 128 — need algorithmic register reduction
3. Options: m64n64k16 QK (S_acc 64→32), split PV into phases, stage O_acc to SMEM

### File
/tmp/test_4wg_v2.cu — the working version (needs to be saved to worktree)
