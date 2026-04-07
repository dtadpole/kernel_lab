---
name: Blackwell vs Hopper register file — same 64K
description: Both SM90 and SM120 have 65536 registers per SM. 4-WG difference is toolchain, not hardware.
type: project
---

## Confirmed (NVIDIA official docs)
Blackwell Tuning Guide: "The register file size is 64K 32-bit registers per SM."
H100 (SM90): cuDeviceGetAttribute reports 65536 regs per SM.

**Same register file size: 65,536 registers per SM.**

## Why Blackwell 4-WG works but SM90 doesn't
AVO paper's 4-WG (512 threads): 192×256 + 80×128 + 48×128 = 65,536 ✓ (fits)

On SM90 with CUDA 12:
- `__launch_bounds__(512, 1)` → compiler allocates 128 regs/thread
- Consumer needs 168 → 40 regs spilled to local memory
- setmaxnreg.inc at runtime doesn't fix compile-time spills

On Blackwell with CUDA 13+:
- Compiler likely supports per-WG register budget at compile time
- OR: setmaxnreg is better integrated into code generation
- OR: Blackwell's AVO kernel uses fewer regs per MMA warp (different algorithm)

**Conclusion: 4-WG on SM90 blocked by nvcc toolchain, not hardware.**

## How to verify
Try compiling with a newer CUDA toolkit that supports SM120.
If `__launch_bounds__(512, 1)` + setmaxnreg produces 0 spills on SM120, the toolchain theory is confirmed.
