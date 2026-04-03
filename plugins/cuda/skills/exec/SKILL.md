---
name: exec
description: Compile, evaluate, and profile CUDA kernels using the remote cuda_exec service
user-invocable: true
argument-hint: <action> [options]
---

# CUDA Kernel Execution

Compile, evaluate, and profile CUDA kernels via the cuda_exec remote service.

## Tools

- **health** — Check if the target cuda_exec service is responding
- **compile** — Compile CUDA source to binary/PTX/SASS
- **evaluate** — Correctness + performance testing against runtime configs
- **profile** — NCU profiling (generated or reference side)
- **execute** — Ad-hoc command execution (e.g. query device info, toolkit versions)

## Workflow

1. **Health check**: Verify the target service is reachable before sending work
2. **Compile**: Call `compile` with metadata, reference_files, and generated_files
3. **Evaluate**: Call `evaluate` with the same metadata and configs to test correctness + performance
4. **Profile** (optional): Call `profile` to get NCU hardware metrics
5. **Iterate**: Modify source code → increment `metadata.turn` → compile again

## Kernel Types

The exec workflow supports multiple kernel implementation approaches:

- **Generated CUDA** — Hand-written `.cu` kernel following the `kernel_run` contract
- **CuTe DSL reference** — Python reference using `cutlass.cute` JIT compilation
- **cuDNN** — Native cuDNN kernel implementation (planned)

All types share the same compile → evaluate → profile pipeline. The reference
implementation provides the ground truth for correctness comparison.

## Rules

- Compile exactly once per turn before evaluate or profile
- New source code requires a new turn (increment metadata.turn)
- Old turns are immutable — never recompile on a previous turn number
- One compile fans out to many evaluate/profile calls with different configs

## Metadata

Every tool requires a `metadata` dict:
```json
{
  "run_tag": "optim_001",
  "version": "v1",
  "direction_id": 7,
  "direction_slug": "vector-add",
  "turn": 1
}
```

## File Inputs

### compile

```json
{
  "reference_files": {"cutedsl.py": "<content>"},
  "generated_files": {"generated.cu": "<content>"}
}
```

- `reference_files`: Python reference source (Model class, get_inputs, get_init_inputs)
- `generated_files`: Must contain exactly one `.cu` file with `kernel_run` entry point

### evaluate / profile

```json
{
  "configs": {
    "vec1d-n65536": {"shape": [65536], "rank": 1, "input_size": 65536, "shape_kind": "1d"},
    "tensor2d-1024x1024": {"shape": [1024, 1024], "rank": 2, "input_size": 1048576, "shape_kind": "2d"}
  }
}
```

Configs are kernel-specific. Each config_slug is a stable identifier across evaluate/profile calls.

## Error Handling

- **Compile failure**: Check `all_ok` field. Use `/cuda:inspect` with `get_compile_data` field="tool_outputs" to see nvcc/ptxas errors.
- **Evaluate failure**: Check per-config `status` and `correctness.passed`. Use `/cuda:inspect` to fetch full logs.
- **Profile failure**: NCU profiling requires elevated permissions. Check `status` per config.

## Available Fixtures

Test workloads — reference/configs in `data/fixtures/`, generated code in `data/generated/{arch}/`:

| Fixture | Description | Configs |
|---------|-------------|---------|
| `vecadd` | BF16 vector addition | 7 configs (1D/2D/3D shapes) |
| `matmul` | Matrix multiplication (CuTe reference) | varies |
| `fa4` | Flash Attention v4 | varies |

## Reviewing Results

Use `/cuda:inspect` to review compile, evaluate, and profile results from past turns.
