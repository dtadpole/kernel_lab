---
name: flash-attn and FA4 CuTe DSL installation
description: Two-part install — flash-attn 2.7.x (CUDA kernels) + flash_attn.cute (FA4 pure Python JIT) with env workaround
type: project
---

Flash Attention on SM90 H100 requires two separate installs:

**Part 1: flash-attn 2.7.x** (FA2/FA3 compiled CUDA kernels)
- `pip install packaging wheel && pip install flash-attn==2.7.4.post1 --no-build-isolation`
- Needs `CUDA_HOME=/usr/local/cuda-12.9` on devvms (12.8 has no headers)
- flash-attn 2.8.x has ABI issues with PyTorch 2.6.0 — stick with 2.7.x
- Performance: ~325-360 TFLOPS on H100

**Part 2: FA4 CuTe DSL** (pure Python, JIT-compiled via cutlass-dsl)
- Not on PyPI as a standalone package. Install by copying from flash-attention main branch:
  ```
  git clone --depth 1 --sparse https://github.com/Dao-AILab/flash-attention.git /tmp/fa4
  cd /tmp/fa4 && git sparse-checkout set flash_attn/cute
  cp -r /tmp/fa4/flash_attn/cute $VENV/lib/python3.12/site-packages/flash_attn/cute
  ```
- Dependencies: `pip install 'quack-kernels>=0.3.3' 'apache-tvm-ffi>=0.1.5,<0.2' 'torch-c-dlpack-ext'`
- **Required env var:** `TVM_FFI_DISABLE_TORCH_C_DLPACK=1` (torch-c-dlpack-ext has CXX11 ABI mismatch with PyTorch 2.6.0)
- Performance: **549-679 TFLOPS** on H100 — ~2x faster than FA2

**Why this matters:** Without FA4, cutedsl.py falls back to FA2 or SDPA, losing half the performance. FA4 uses SM90-native warp-specialized kernels via CuTe DSL JIT.

**How to apply:** Always install both parts. Set `TVM_FFI_DISABLE_TORCH_C_DLPACK=1` in the systemd service unit and benchmark scripts.
