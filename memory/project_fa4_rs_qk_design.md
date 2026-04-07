---
name: FA4 RS-mode QK design
description: Replace SS-mode QK GEMM with RS-mode (ldmatrix Q→registers) to reduce SMEM bandwidth contention
type: project
---

## Why RS-mode QK
Current SS-mode: both Q and K read from SMEM via descriptors → compete for SMEM port.
RS-mode: Q loaded to registers via ldmatrix, only K uses SMEM descriptor → halves SMEM port pressure.
FA4 CuTe DSL uses RS-mode QK — this may explain their 4-8% advantage.

## Previous attempt (generated.cu)
Had wrong P register packing for PV GEMM → IMA crash.
Root cause was packing S_acc groups incorrectly for wgmma RS A operand layout.
The QK RS-mode (ldmatrix) part was likely correct — the bug was in PV, not QK.

## RS-mode QK implementation
For m64n128k16 RS mode:
- A operand (Q): 4 uint32_t registers per thread = 8 bf16 values
- B operand (K): SMEM descriptor
- Need ldmatrix.sync.aligned.m8n8.x4 to load Q from SMEM into registers

Q SMEM is swizzled (SWIZZLE_128B). ldmatrix address computation must account for swizzle:
- row = warp_in_wg * 16 + (lane_id % 16)
- col_group = k_step * 2 + (lane_id / 16)
- physical_col = col_group XOR (row & 7)  (SWIZZLE_128B pattern)
- smem_addr = Q_base + row * 128 + physical_col * 16

Key: the Q descriptor (make_wgmma_desc) is replaced by ldmatrix + register A operand.
K descriptor stays the same.

## Result (2026-04-05): RS-QK is 2-5% SLOWER than SS-QK
ldmatrix.x4 also uses SMEM port — no bandwidth savings vs SS.
Extra instruction overhead (64 ldmatrix + address computations).
SS mode lets wgmma hardware read SMEM internally, bypassing instruction pipeline.
FA4 CuTe DSL's advantage comes from CUTLASS template compiler optimization, not RS vs SS.

V3 SS-QK on main branch remains the best SM90 kernel.
