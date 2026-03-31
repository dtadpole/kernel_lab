---
name: inspect
description: Review compile, evaluate, and profile results from past CUDA kernel optimization runs
user-invocable: true
argument-hint: <run_tag> [turn]
---

# Inspect Optimization Results

Review and analyze results from past CUDA kernel optimization runs.

## Tools

- **get_compile_data** — Structured compile results (ptx, sass, resource_usage, tool_outputs)
- **get_evaluate_data** — Structured correctness/performance with config filtering
- **get_profile_data** — NCU summary with config filtering
- **get_data_point** — Raw uncompacted request/response fallback
- **read_file** — On-demand file reading from turn directories (artifacts, logs, state)

## Workflow

1. Identify the run to inspect via `$ARGUMENTS` (run_tag and optionally turn number)
2. Use `get_compile_data` to review PTX, SASS, register/shared-memory usage
3. Use `get_evaluate_data` to check correctness and performance across configs
4. Use `get_profile_data` to examine NCU hardware metrics
5. Use `read_file` to fetch full details (logs, artifacts) that were truncated in responses
6. Use `get_data_point` as a fallback to see raw uncompacted request/response

## Metadata

All tools require a `metadata` dict identifying the run and turn:
```json
{
  "run_tag": "optim_001",
  "version": "v1",
  "direction_id": 7,
  "direction_slug": "vector-add",
  "turn": 1
}
```

## Common Patterns

### Compare performance across turns
Call `get_evaluate_data` for each turn to see how latency and correctness evolved.

### Deep-dive a failing config
1. `get_evaluate_data` with `config_slug` filter to isolate the failure
2. `read_file` to fetch the full log: `logs/evaluate.attempt_001.config_<slug>.log`

### Examine generated assembly
1. `get_compile_data` with `field="ptx"` or `field="sass"`
2. `get_compile_data` with `field="resource_usage"` to check register pressure
