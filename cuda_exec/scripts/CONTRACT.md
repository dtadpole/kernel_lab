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
