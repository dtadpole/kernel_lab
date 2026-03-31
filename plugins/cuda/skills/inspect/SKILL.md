---
name: inspect
description: Review compile, evaluate, and profile results from past optimization runs
user-invocable: true
argument-hint: <run_tag> [turn]
---

# TODO: Inspect Skill

Review and analyze results from past CUDA kernel optimization runs.

## Planned Workflow

1. List available runs or accept a specific run_tag
2. Show compile results (PTX, SASS, resource usage)
3. Show evaluate results (correctness, performance across configs)
4. Show profile results (NCU summaries)
5. Compare results across turns to track optimization progress

## MCP Tools Used

- `get_compile_data` — structured compile results
- `get_evaluate_data` — structured evaluate results with config filtering
- `get_profile_data` — NCU summary with config filtering
- `get_data_point` — raw request/response fallback

## Status: NOT IMPLEMENTED
