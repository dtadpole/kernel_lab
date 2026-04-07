---
name: FA4 QK/PV intra-WG overlap
description: FA4 kernel optimization — overlapping QK and PV WGMMA groups within same consumer warp group using wait_group depth
type: project
---

The main performance gap between our FA4 CUDA kernel and CuTe DSL was the **intra-warpgroup QK/PV overlap** pattern.

**CuTe DSL architecture** (`flash_fwd_sm90.py`):
- `intra_wg_overlap=True` for hdim≤128
- Prologue (`first_half_block_overlap`): QK only, no barriers
- Mainloop (`mma_one_n_block_intrawg_overlap`): issues QK[n+1] + PV[n] concurrently, softmax overlaps PV via `wgmma.wait_group(1)`
- Epilogue (`last_half_block_overlap`): PV only
- O rescaling deferred AFTER PV for hdim≤128 (`rescale_O_before_gemm=False`)

**Why:** The config for hdim=128 is `FwdConfig(128, 128, mma_pv_is_rs=True, intra_wg_overlap=True)` — 2 consumer WGs, 240 consumer regs, 24 producer regs, num_stages=2.

**How to apply:** When optimizing WGMMA-based kernels, always check if consecutive GEMM stages can be overlapped via `wait_group` depth control. The key requirement is separate accumulator registers (S_acc vs P_packed vs O_acc).
