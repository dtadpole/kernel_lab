# Matmul Kernel: SMEM Bank Conflict Fix + MMA/ldmatrix Interleaving

**Date:** 2026-03-31
**Status:** Implementation-ready
**Branch:** worktree-matmul

## Problem

NCU profile at 4096×4096 shows:
- **8,650,752 excess SMEM wavefronts** (11% above ideal 67,633,152)
- **Estimated 10.26% speedup** from eliminating bank conflicts
- Top warp stalls: math_pipe_throttle (41%), wait (30%), mio_throttle (14%)

Current kernel reaches 344–357 TFLOPS (68–71% of 503.8 peak). Target: >380 TFLOPS (>75%).

## Root Cause Analysis

### Bank Conflicts
SMEM has 32 banks of 4 bytes each. Each uint4 (16 bytes) spans 4 consecutive banks. With 8 uint4 columns per row = 128 bytes = one full bank cycle, two accesses to the same column but different rows always hit the same 4 banks.

The cp.async store swizzle `storeCol = (laneID%8) ^ (laneID/8)` maps all 32 lanes to 8 columns, with 4 lanes per column. These 4 lanes access different rows but the same column → **4-way bank conflict per cp.async store**.

For K=64 with 128 SMEM rows and 8 cp.async per stage per thread, the total excess from cp.async bank conflicts across 16.7M ldmatrix operations contributes to the 8.6M excess wavefronts.

### Pipeline Bubble
The current main loop loads ALL A and B registers before computing ANY MMAs. This creates a bubble where tensor cores idle during the ~20-cycle ldmatrix latency window.

## Design

### Fix 1: SMEM Padding for Bank Conflict Elimination

Add 1 uint4 padding per row (9 columns instead of 8). This shifts bank alignment by 4 banks per row, breaking the column-aligned conflict pattern.

```
Before: row R at byte offset R * 128 → bank = col * 4 (independent of row)
After:  row R at byte offset R * 144 → bank = (col * 4 + R * 4) % 32 (row-dependent)
```

With 9 columns: two accesses to the same column but rows R and S hit banks `(col*4 + R*4)%32` and `(col*4 + S*4)%32`. These differ when `(R-S)*4 % 32 ≠ 0`, i.e., when `R-S` is not a multiple of 8. Since storeRow = threadID/8 (0..31), consecutive storeRow values differ by 1, giving different banks.

**SMEM cost:** 9/8 × 96 KB = 108 KB. But GPU max is 100 KB per SM.

**Alternative: 128B XOR swizzle (no padding).** Instead of adding columns, XOR the byte offset with row-dependent bits:

```c
// Instead of direct [row][col] access, use swizzled column:
int swizzled_col = col ^ ((row >> 1) & 3);
```

This permutes columns based on row index, distributing bank accesses across different banks for different rows. No extra SMEM needed.

**Chosen approach:** 128B XOR swizzle (no SMEM increase).

### Fix 2: MMA/ldmatrix Interleaving

Restructure the inner loop to interleave A ldmatrix loads with MMA:

```
Current:  load A[0..3], load B[0..3], cp.async, MMA[0..3][0..3]
Proposed: load A[0], load B[0..3], cp.async, {MMA(0,*) + load A[1]}, {MMA(1,*) + load A[2]}, ...
```

B is loaded entirely first (needed for all MMA columns). Then A is loaded one m-tile at a time, interleaved with MMA for the previous m-tile. This hides the ~20-cycle ldmatrix.x4 latency behind MMA execution.

## Implementation Plan

1. Add `SMEM_STRIDE` constant (9 or keep 8 with XOR swizzle)
2. Modify store addresses: `As[row * SMEM_STRIDE + swizzled_col]` or keep 2D with XOR
3. Modify load addresses: same swizzle applied to loadColA/loadColB
4. Restructure MMA loop for A-interleaving
5. Compile, test correctness, benchmark

## Expected Impact

- Bank conflict fix: ~5-10% improvement (NCU estimate: 10.26%)
- Instruction interleaving: ~2-5% improvement (if compiler wasn't already doing it)
- Combined target: 370-400 TFLOPS at 4096/8192 (73-79% peak)
