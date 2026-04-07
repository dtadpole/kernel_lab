---
name: tma-swizzle-constraints
description: cuTensorMapEncodeTiled SWIZZLE_128B requires boxDim[0]*elemSize==128; for bf16 DIM=128 split into 2x64 loads
type: project
---

cuTensorMapEncodeTiled with SWIZZLE_128B has a strict constraint: `boxDim[0] * elementSize` must equal exactly 128 bytes. For bf16 (2 bytes), this means `boxDim[0] = 64`.

For head_dim=128 (DIM=128), each Q/K/V tile must be loaded as 2 TMA calls of box=[64, BLOCK]:
- lo half: coord0 = head_id * DIM + 0
- hi half: coord0 = head_id * DIM + 64
- arrive_expect_tx total = 2 * HALF_TILE_BYTES per mbarrier

**Why:** This is an architectural constraint of the TMA hardware. SWIZZLE_128B operates on exactly 128-byte-wide rows.

**How to apply:** When implementing TMA for any kernel with innermost dimension > 64 bf16 elements, split into multiple loads. The wgmma descriptor should use layout_type=1 (SWIZZLE_128B) with LBO=128 bytes, SBO=1024 bytes per half-tile.
