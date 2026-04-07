---
name: wgmma descriptor format and SMEM layout requirements
description: SM90 wgmma requires interleaved SMEM layout (not row-major); descriptor bitfield layout; RS vs SS variant syntax differences
type: project
---

wgmma B operand from SMEM descriptor requires data in **canonical interleaved layout**, NOT simple row-major or swizzled row-major. CUTLASS canonical layout interleaves 8 N-elements with 8 K-elements within each 128-byte block.

**Why:** Row-major cp.async loads produce contiguous rows, but wgmma hardware expects interleaved columns. Verified: V=ones (uniform) passes but V=arange (column-varying) fails with column offset.

**How to apply:**
- To use wgmma, data must be loaded via TMA (which produces correct interleaved layout) or explicitly rearranged in SMEM after cp.async load
- GmmaDescriptor bitfield: [0:14) start_addr>>4, [16:30) leading>>4, [32:46) stride>>4, [49:52) base_offset, [62:64) layout_type (0=INTERLEAVE, 1=SW128B, 2=SW64B, 3=SW32B)
- RS variant (A=regs, B=SMEM): 4 trailing params: `p, scaleA, scaleB, tnspB`
- SS variant (both SMEM): 5 trailing params: `p, scaleA, scaleB, tnspA, tnspB`
- Compile with `-gencode arch=compute_90a,code=sm_90a` (NOT just `-arch=sm_90`)
- Despite wrong results, wgmma showed 25-40% speedup (249-312 vs 200-229 TFLOPS) due to async tensor core execution
