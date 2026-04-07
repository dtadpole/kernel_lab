---
name: FA4 CuTe DSL environment setup
description: Environment variables and package versions required to run FA4 CuTe DSL (flash_attn.cute) on devvm8491 H100
type: project
---

FA4 CuTe DSL (`flash_attn.cute.flash_attn_func`) requires specific environment setup on devvm8491 (h8_4).

## Required Environment Variable

```bash
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
```

**Why:** The installed `torch_c_dlpack_ext==0.1.5` has an ABI mismatch with PyTorch 2.6.0+cu124. The `libtorch_c_dlpack_addon_torch26-cuda.so` references `_ZNK3c106Device3strB5cxx11Ev` which doesn't exist in the installed torch build. Setting `TVM_FFI_DISABLE_TORCH_C_DLPACK=1` bypasses this extension without affecting FA4 functionality.

## Working Package Versions (devvm8491)

- Python: 3.12 (from `.cuda_exec_service/.venv`)
- PyTorch: 2.6.0+cu124
- flash_attn: 2.7.4.post1 (includes flash_attn.cute for FA4 CuTe DSL)
- nvidia-cutlass-dsl: 4.4.2
- nvidia-cutlass-dsl-libs-base: 4.4.2
- apache-tvm-ffi: 0.1.9
- torch_c_dlpack_ext: 0.1.5 (disabled via env var)

## Import Pattern

```python
# CORRECT: FA4 CuTe DSL (549-658 TFLOPS on H100)
from flash_attn.cute import flash_attn_func

# WRONG: This imports FA2/3 (322-356 TFLOPS — much slower)
from flash_attn import flash_attn_func
```

## How to Apply

- Always set `TVM_FFI_DISABLE_TORCH_C_DLPACK=1` before importing `flash_attn.cute`
- Use the `.cuda_exec_service/.venv` Python environment on devvm8491
- CUDA_HOME should be `/usr/local/cuda-12.9` for compilation
