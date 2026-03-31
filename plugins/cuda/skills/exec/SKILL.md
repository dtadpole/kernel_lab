---
name: exec
description: Compile, evaluate, and profile CUDA kernels using the remote cuda_exec service
user-invocable: true
argument-hint: <action> [options]
---

# CUDA Kernel Execution

Compile, evaluate, and profile CUDA kernels via the cuda_exec remote service.

## Tools

- **compile** — Compile CUDA source to binary/PTX/SASS
- **evaluate** — Correctness + performance testing against runtime configs
- **profile** — NCU profiling (generated or reference side)
- **execute** — Ad-hoc command execution (e.g. query device info, toolkit versions)

## Workflow

1. **Compile**: Call `compile` with metadata, reference_files, and generated_files
2. **Evaluate**: Call `evaluate` with the same metadata and configs to test correctness + performance
3. **Profile** (optional): Call `profile` to get NCU hardware metrics
4. **Iterate**: Modify source code → increment `metadata.turn` → compile again

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

## Reviewing results

Use `/cuda:inspect` to review compile, evaluate, and profile results from past turns.
