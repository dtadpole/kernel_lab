# cuda_exec Scripts Contract

## Input Generation

**The harness generates all inputs. Implementations do NOT generate inputs.**

This is the fundamental principle. All implementations (reference `.py`, generated `.cu`,
vendor baseline) receive the same inputs from the same source — `eval_support.generate_inputs()`.

```
eval_support.generate_inputs(config, device)
        │
        ├──→ Reference .py Model.forward(*inputs)
        ├──→ Generated .cu kernel_run(inputs, outputs, n, stream)
        └──→ Vendor baseline .py Model.forward(*inputs)
```

### Why

- **Correctness comparison requires identical inputs.** If each implementation generates
  its own inputs (even with the same seed), floating-point order differences or library
  version differences can produce different tensors.
- **Single source of truth.** One function, one set of tensors, shared across all sides.

### Rules

1. **`generate_inputs(config, device)`** in `eval_support.py` is the single source of
   truth for input generation. It dispatches based on `config["family"]` and `config["shape"]`.

2. **Reference `.py` files** must export `class Model(nn.Module)` and `get_init_inputs()`.
   They must NOT generate inputs — the harness calls `Model.forward(*inputs)` where
   `inputs` comes from `generate_inputs()`.

3. **Generated `.cu` files** must export `kernel_run(inputs, outputs, n, stream)`.
   The eval_harness binary calls `generate_inputs()` equivalent in C, then passes
   pointers to `kernel_run()`.

4. **`get_inputs(config)` in reference files is DEPRECATED** — it exists for standalone
   testing but is NOT used during trial. The harness always uses `generate_inputs()`.

## Input PRNG — Normalization

The C harness and Python trial script must use **identical PRNG** to generate inputs.
This was a real bug: mismatched formulas caused all correctness checks to fail.

### The formula (single source of truth)

```c
// eval_harness.cu — fill_random_bf16
unsigned int h = (unsigned int)idx ^ seed;
h = (h ^ 61u) ^ (h >> 16);
h += (h << 3);
h ^= (h >> 4);
h *= 0x27d4eb2du;
h ^= (h >> 15);
float val = (float)(h & 0xFFFFu) / 65536.0f - 0.5f;
buf[idx] = __float2bfloat16(val);
```

```python
# trial.py — _harness_fill_random_bf16
f = (h & 0xFFFF).astype(np.float32) / 65536.0 - 0.5
```

### Key details

- **Range: [-0.5, 0.5)** — kept small to avoid FP overflow in matmul accumulation
- **Hash: Wang's 32-bit integer hash** — `idx ^ seed` then 5-step mixing
- **Normalization: `(h & 0xFFFF) / 65536.0 - 0.5`** — NOT `2*(h & 0xFFFF)/65535 - 1`
- **Seeds: 1, 2, 3, ...** per input tensor (simple, not 0xCAFE or 0xC0DE)

### Correctness pass

The harness runs the kernel once more after all timed trials with deterministic inputs:
- Allocates **fresh** buffers (new pointers)
- Fills with `fill_random_bf16(seed = j + 1)` for input tensor `j`
- The Python `_verify_correctness` reproduces this exact PRNG
- Both sides get identical inputs → outputs are comparable

### If you change the PRNG

Change it in **both** places simultaneously:
1. `eval_harness.cu` → `fill_random_bf16()`
2. `trial.py` → `_harness_fill_random_bf16()`

Verify with a small n (e.g., 4) by printing values from both sides.

## Correctness Comparison

The trial script (`trial.py`) compares outputs element-by-element:

1. Run reference: `ref_output = Model(*inputs)` where inputs from `generate_inputs()`
2. Run generated: kernel binary produces output from the same inputs
3. Compare: `allclose(ref_output, gen_output, atol=1e-2, rtol=1e-2)`

Both sides must receive identical input tensors with identical memory layout.

## kernel_run Contract

```c
extern "C" int kernel_run(
    __nv_bfloat16** inputs,   // array of input tensor pointers
    int num_inputs,           // number of input tensors
    __nv_bfloat16** outputs,  // array of output tensor pointers
    int num_outputs,          // number of output tensors
    int n,                    // total number of elements
    cudaStream_t stream       // CUDA stream
);
```

- Return 0 on success, non-zero on error
- The harness allocates and fills input tensors before calling `kernel_run`
- The harness allocates output tensors before calling `kernel_run`
- `kernel_run` writes results into the pre-allocated output tensors
