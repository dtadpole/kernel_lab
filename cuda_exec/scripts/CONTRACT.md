# cuda_exec Scripts Contract

## Input Generation

**The harness generates all inputs. Implementations do NOT.**

Both harnesses (C `eval_harness.cu` and Python `trial.py`) must produce
identical inputs for the correctness pass. Same PRNG, same seeds, same formula.

## PRNG

Wang's 32-bit hash. Normalization: `(h & 0xFFFF) / 65536.0 - 0.5` → range [-0.5, 0.5).
Seeds: `1, 2, 3, ...` per input tensor.

```c
// C (eval_harness.cu)
float val = (float)(h & 0xFFFFu) / 65536.0f - 0.5f;
```

```python
# Python (trial.py)
f = (h & 0xFFFF).astype(np.float32) / 65536.0 - 0.5
```

If you change the PRNG, change both simultaneously. Verify with n=4.

## kernel_run

```c
extern "C" int kernel_run(
    __nv_bfloat16** inputs, int num_inputs,
    __nv_bfloat16** outputs, int num_outputs,
    int n, cudaStream_t stream);
```

- Return 0 on success
- Harness allocates and fills inputs/outputs before calling
- `kernel_run` writes results into pre-allocated output buffers

## Reference .py

- Export `class Model(nn.Module)` and `get_init_inputs()`
- Do NOT generate inputs — harness provides them via `Model.forward(*inputs)`

## Correctness

After timed trials, the harness runs one correctness pass with fresh buffers
filled by the PRNG (seeds 1, 2, ...). Python reproduces the same PRNG and
compares outputs element-by-element via `allclose(atol=1e-2, rtol=1e-2)`.
