---
name: exec
description: Compile, evaluate, and profile CUDA kernels locally or on remote GPU hosts
user-invocable: true
argument-hint: <local|remote> <action> [options]
---

# CUDA Kernel Execution

Compile, evaluate, and profile CUDA kernels locally or on remote GPU hosts via the cuda_exec service.

## Target (Required)

Every action tool call requires an explicit `target` parameter. No default — you must always specify where to run.

### Local execution
```json
{"mode": "local", "gpu_index": 0}
```
Runs directly on the local machine. `gpu_index` selects which GPU (maps to `CUDA_VISIBLE_DEVICES`).

### Remote execution
```json
{"mode": "remote", "host": "_one"}
```
Proxies to the cuda_exec HTTP service on the named host. Host names are resolved from `conf/hosts/default.yaml`.

### Skill invocation
- `/cuda:exec local 0 compile ...` → `target={"mode": "local", "gpu_index": 0}`
- `/cuda:exec remote _one compile ...` → `target={"mode": "remote", "host": "_one"}`

### Available remote hosts
| Host | GPU | Description |
|------|-----|-------------|
| `_one` | RTX PRO 6000 Blackwell | 98GB |
| `_two` | RTX PRO 6000 Blackwell | 98GB |
| `h8_3` | 8x NVIDIA H100 | Meta devvm |
| `h8_4` | 8x NVIDIA H100 | Meta devvm |

## Tools

- **health** — Check if the target cuda_exec service is responding (remote only)
- **compile** — Compile CUDA source to binary/PTX/SASS
- **evaluate** — Correctness + performance testing against runtime configs
- **profile** — NCU profiling (generated or reference side)
- **execute** — Ad-hoc command execution (e.g. query device info, toolkit versions)

## Workflow

1. **Target selection**: Determine local vs remote based on user instruction
2. **Health check** (remote only): Verify the target service is reachable
3. **Compile**: Call `compile` with target, metadata, reference_files, and generated_files
4. **Evaluate**: Call `evaluate` with the same target and metadata, plus configs
5. **Profile** (optional): Call `profile` to get NCU hardware metrics
6. **Iterate**: Modify source code → increment `metadata.turn` → compile again

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
- Use the same target for all calls within a turn

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
- **Local GPU not found**: Check `gpu_index` matches available GPUs (`nvidia-smi`).
- **Remote host unreachable**: Check host is running with `/cuda:service health <host>`.

## Available Fixtures

Test workloads — reference/configs in `data/fixtures/`, generated code in `data/generated/{arch}/`:

| Fixture | Description | Configs |
|---------|-------------|---------|
| `vecadd` | BF16 vector addition | 7 configs (1D/2D/3D shapes) |
| `matmul` | Matrix multiplication (CuTe reference) | varies |
| `fa4` | Flash Attention v4 | varies |

## Reviewing Results

Use `/cuda:inspect` to review compile, evaluate, and profile results from past turns.
