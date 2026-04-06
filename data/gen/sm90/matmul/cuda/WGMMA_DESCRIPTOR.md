# WGMMA Descriptor Format (SM90, from PTX ISA)

## Bit Layout

| Bits | Field |
|------|-------|
| 13–0 | Start address: `(addr & 0x3FFFF) >> 4` |
| 29–16 | Leading Dimension Byte Offset (LBO >> 4) |
| 45–32 | Stride Dimension Byte Offset (SBO >> 4) |
| 51–49 | Base offset (swizzle alignment) |
| 63–62 | Swizzle: 0=none, 1=128B, 2=64B, 3=32B |

**Bit 14 is part of LBO, NOT swizzle. Bit 62 is swizzle, NOT "smem flag".**

## BF16 Parameters (T = 8)

### K-Major, 128B swizzle
- A descriptor (M rows × K cols, K contiguous):
  - LBO = 8 * TILE_K * sizeof(bf16) = 8 * 64 * 2 = 1024 → encoded = 64
  - SBO = 0 (not used for single-core-matrix) → set to reasonable value
  
  Actually for K-major: canonical `((8,m),(T,2k)):((8T,SBO),(1,T))`
  - SBO = stride between 8-row groups = 8 * K_per_group * sizeof(bf16)
  - For TILE_K=64: K is contiguous, SBO = 8 * 64 * 2 = 1024 → 64

### MN-Major, 128B swizzle (for B without transpose)
- B descriptor (loaded as K×N_sub with N contiguous):
  - LBO = 256 * sizeof(bf16) = 512 → encoded = 32
  - SBO = 512 * sizeof(bf16) = 1024 → encoded = 64
  - Swizzle mode = 1 (128B)

## Previous make_desc BUG

The old `make_desc` was wrong:
```cpp
// OLD (WRONG):
desc |= (uint64_t)((addr >> 4) & 0x3FFF);        // bits 0-13: OK
desc |= (uint64_t)(1) << 16;                       // bit 16: WRONG (sets LBO bit, not "base offset")
desc |= (uint64_t)(stride_16B & 0x3FFF) << 32;    // bits 32-45: puts stride in SBO field
desc |= (uint64_t)(1) << 62;                       // bit 62: WRONG (sets swizzle=01=128B, not "smem flag")
```

The kernel "worked" because:
- bit 62 = 1 → swizzle mode = 01 = 128B (accidentally correct for 128B TMA swizzle!)
- LBO field (bits 29-16) = 0x0001 (bit 16 set) = 1 → LBO = 1 * 16 = 16 bytes
- SBO field (bits 45-32) = stride_16B → SBO = stride_16B * 16

For K-major with TILE_K=64: SBO = 64 * 16 = 1024 bytes = 8 * 64 * 2. This happens
to be the correct SBO for K-major! And LBO = 16 bytes = 1 unit (the minimum, which
works for K-major where LBO is often "1" or "assumed").

So the old descriptor accidentally produced correct values for K-major 128B swizzle.
But for MN-major it would need different LBO/SBO values.
```

## Correct make_desc for both layouts

```cpp
// K-major (A matrix, K contiguous):
uint64_t make_desc_k_major(void* smem, int tile_k) {
    uint32_t addr = __cvta_generic_to_shared(smem);
    uint64_t desc = 0;
    desc |= (uint64_t)((addr >> 4) & 0x3FFF);           // bits 0-13
    desc |= (uint64_t)(1 & 0x3FFF) << 16;               // bits 16-29: LBO = 1 (16 bytes)
    int sbo_16B = (8 * tile_k * 2) >> 4;
    desc |= (uint64_t)(sbo_16B & 0x3FFF) << 32;         // bits 32-45: SBO
    desc |= (uint64_t)(1) << 62;                          // bits 62-63: swizzle = 128B
    return desc;
}

// MN-major (B matrix, N contiguous):
uint64_t make_desc_mn_major(void* smem) {
    uint32_t addr = __cvta_generic_to_shared(smem);
    uint64_t desc = 0;
    desc |= (uint64_t)((addr >> 4) & 0x3FFF);           // bits 0-13
    int lbo_16B = (256 * 2) >> 4;  // = 32
    desc |= (uint64_t)(lbo_16B & 0x3FFF) << 16;         // bits 16-29: LBO
    int sbo_16B = (512 * 2) >> 4;  // = 64
    desc |= (uint64_t)(sbo_16B & 0x3FFF) << 32;         // bits 32-45: SBO
    desc |= (uint64_t)(1) << 62;                          // bits 62-63: swizzle = 128B
    return desc;
}
```
