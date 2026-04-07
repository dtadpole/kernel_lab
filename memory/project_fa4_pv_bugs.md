---
name: FA4 generated kernel PV GEMM bugs
description: SM90 generated.cu has two PV GEMM bugs — wrong k-step stride and P register packing mismatch
type: project
---

The hand-written SM90 FA4 kernel (`.claude/worktrees/fa4/data/generated/sm90/fa4/generated.cu`) has two PV GEMM bugs that produce incorrect output (cos_sim ≈ 0 vs reference):

1. **PV k-step stride**: Uses `ks * 32` (column advance for tnspB=1) but should use `ks * 16 * HALF_ROW_STRIDE = ks * 2048` (row advance for tnspB=0). For tnspB=0, K maps to rows not columns.

2. **P register packing**: `PACK_P_REGS` converts QK m64n128k16 output (16 groups × 8 N-positions) to PV m64n64k16 RS input, but the permutation is wrong. QK groups have 8-column spacing; PV RS expects 4-element K-position blocks per lane.

**Why:** The QK and PV GEMMs use different wgmma shapes (m64n128k16 vs m64n64k16) with different operand modes (tnspB=1 vs tnspB=0), creating a register layout impedance mismatch.

**How to apply:** When fixing the generated kernel or writing new CUDA wgmma kernels, always verify the register mapping between QK output and PV input matches. The CuTe DSL reference handles this correctly via CUTLASS abstractions.
