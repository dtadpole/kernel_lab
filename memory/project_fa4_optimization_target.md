---
name: FA4 optimization target — 10%+ above cuDNN AND FA4 DSL
description: User set ambitious target. Current best is ~1.06× cuDNN, ~0.99× FA4 DSL. Need 1.10× both. Requires fundamentally better scheduling or algorithmic changes.
type: project
---

## Target
- Beat cuDNN by 10%+ on ALL configs
- Beat FA4 CuTe DSL by 10%+ on ALL configs

## Why this should be achievable
1. We use CUDA + inline PTX — full hardware access
2. We have source documentation + PTX ISA + profiling tools
3. Others (AVO on Blackwell) achieved +10% over FA4 DSL

## Current state (V22b)
- vs cuDNN: 0.94-1.10× (avg ~1.03×) — need +7% more
- vs FA4 DSL: 0.99-1.07× (avg ~1.02×) — need +8% more

## Profile gap analysis (noncausal-b8-s4096)
| Metric | Ours | FA4 DSL | Gap |
|--------|------|---------|-----|
| Tensor Core | 81.3% | 83.1% | -1.8% |
| Wait stall | 1.71 | 1.46 | +17% |
| Barrier stall | 0.56 | 0.38 | +47% |

## Remaining optimization directions to explore
1. Deeper wgmma pipeline (wait_group depth > 1 for PV)
2. Reorder softmax phases to increase ILP
3. Non-blocking fence variants
4. Bitmask causal masking
5. CUTLASS OrderedSequenceBarrier pattern
6. Inline PTX for critical softmax sections
7. Different Q/K/V double-buffering depths
