---
name: inspect
description: Review compile, evaluate, and profile results from past CUDA kernel optimization runs
user-invocable: true
argument-hint: <run_tag> [turn]
---

# Inspect Optimization Results

Review and analyze results from past CUDA kernel optimization runs by reading the data store directly.

## Data Store Location

Results are stored in `~/.cuda_exec/` (the runtime data directory), organized by turn:

```
~/.cuda_exec/
  turn_1/
    compile.attempt_001.request.json
    compile.attempt_001.response.json
    evaluate.attempt_001.request.json
    evaluate.attempt_001.response.json
    profile.attempt_001.request.json
    profile.attempt_001.response.json
  turn_2/
    ...
```

## Workflow

1. Identify the run to inspect via `$ARGUMENTS` (run_tag and optionally turn number)
2. List available data points:
   ```bash
   ls ~/.cuda_exec/turn_*/
   ```
3. Read compile results for PTX, SASS, register/shared-memory usage:
   ```bash
   cat ~/.cuda_exec/turn_1/compile.attempt_001.response.json | python3 -m json.tool
   ```
4. Read evaluate results for correctness and performance across configs:
   ```bash
   cat ~/.cuda_exec/turn_1/evaluate.attempt_001.response.json | python3 -m json.tool
   ```
5. Read profile results for NCU hardware metrics:
   ```bash
   cat ~/.cuda_exec/turn_1/profile.attempt_001.response.json | python3 -m json.tool
   ```

## Key Fields in Responses

### Compile Response
- `all_ok` — whether compilation succeeded
- `artifacts.ptx.content` — PTX assembly text
- `artifacts.sass.content` — SASS disassembly text
- `artifacts.resource_usage.content` — register/shared-memory usage
- `tool_outputs` — stdout/stderr from each compile stage

### Evaluate Response
- `all_ok` — true only if every config passed correctness
- `configs.{slug}.correctness` — {passed, max_abs_error, mean_abs_error, ...}
- `configs.{slug}.performance` — {latency_ms: {min, median, max, mean}, runs, comparison}

### Profile Response
- `all_ok` — true if profiling succeeded for all configs
- `configs.{slug}.summary` — NCU metrics summary

## Common Patterns

### Compare performance across turns
Read `evaluate.attempt_001.response.json` from each turn to see how latency evolved.

### Deep-dive a failing config
1. Read evaluate response, find the failing config slug
2. Look for detailed logs in the turn directory

### Examine generated assembly
Read compile response and extract `artifacts.ptx.content` or `artifacts.sass.content`.
