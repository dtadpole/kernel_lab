---
name: FA4 persistent kernel — blocked by ptxas producer/consumer lifetime analysis
description: Persistent kernel spills 300-400B because ptxas can't free producer regs for consumer wgmma without 'return'. All approaches tried, none achieves 0 spills at 384 threads.
type: project
---

## Definitive Finding (2026-04-06)

SM90 persistent FA4 kernel cannot achieve 0 spills with 384 threads.

### Root cause
Non-persistent: producer has `return` → ptxas knows producer regs are dead → consumer gets full budget → 168 regs, 0 spills, setmaxnreg works.

Persistent: producer reaches __syncthreads (no return) → ptxas keeps producer regs alive → consumer can't reuse them → 168 regs + 300-400B spills + C7512 (wgmma serialized).

### All approaches tried
| Approach | Regs | Spill | Performance | Issue |
|----------|------|-------|-------------|-------|
| Non-persistent V9 | 168 | 0B | 520-631 TF | **Optimal** |
| Persistent inline | 168 | 344B | 317 TF | No return → unified regs |
| Persistent noinline consumer | 168 | 348B | 365 TF | C7510: wgmma serialized across call |
| Persistent noinline producer | 168 | 308B | 335 TF | Consumer still spills |
| Persistent volatile SMEM params | 168 | 344B | - | Didn't help |
| Persistent atomic counter | 168 | 316B | - | Zero loop regs, still spills |
| Persistent no launch_bounds | 220 | 0B | ❌ corrupt | 220×384 > 65536 |
| BLOCK_Q=64 persistent (256 thr) | 207 | 0B | 397 TF | Half MMA throughput |
| Sequential tile scheduling | 168 | 344B | 222 TF | Load imbalance |

### cuDNN persistent kernel (from ncu)
168 regs, 0 spill, 1024B stack, 132 blocks.
cuDNN likely uses compiler-internal optimizations or a fundamentally different code structure that we cannot replicate with nvcc.

### The 7% gap on causal-b8-s4096
Confirmed from ncu: our kernel is 2.46ms vs cuDNN 2.52ms in warm L2.
The benchmark gap (0.93×) is entirely from cold-L2 behavior.
Persistent kernel is the correct fix but blocked by SM90 ptxas register analysis.
