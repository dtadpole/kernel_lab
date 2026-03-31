---
name: exec
description: Compile, evaluate, and profile CUDA kernels using the remote cuda_exec service
user-invocable: true
argument-hint: <action> [options]
---

# CUDA Toolkit Execution Service

Compile, evaluate, and profile CUDA kernels using the cuda-toolkit-exec MCP tools.

## Available Tools

### Action Tools (proxy to cuda_exec HTTP API)
- **compile** — Compile CUDA source to binary/PTX/SASS
- **evaluate** — Correctness + performance testing against configs
- **profile** — NCU profiling (generated or reference side)
- **execute** — Ad-hoc command execution
- **read_file** — On-demand file reading from turn directories

### Data Retrieval Tools (read from local data store)
- **get_compile_data** — Structured compile results (ptx, sass, resource_usage, tool_outputs)
- **get_evaluate_data** — Structured correctness/performance with config filtering
- **get_profile_data** — NCU summary with config filtering
- **get_data_point** — Raw uncompacted request/response fallback

## Workflow

1. **Compile first**: Call `compile` with metadata, reference_files, and generated_files
2. **Evaluate**: Call `evaluate` with the same metadata and configs to test correctness + performance
3. **Profile** (optional): Call `profile` to get NCU hardware metrics
4. **Iterate**: Modify source code → increment `metadata.turn` → compile again

## Workflow Rules

- Compile exactly once per turn before evaluate or profile
- New source code requires a new turn (increment metadata.turn)
- Old turns are immutable — never recompile on a previous turn number
- One compile fans out to many evaluate/profile calls with different configs

## Metadata Format

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
