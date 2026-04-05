---
name: exec
description: Compile, trial, and profile CUDA kernels
user-invocable: true
argument-hint: <action> [--gpu N] [options]
---

# CUDA Kernel Execution

Compile, trial, and profile CUDA kernels by calling `cuda_exec` handlers directly via Python CLI.

## GPU Selection

When the user specifies `--gpu N`, set `CUDA_VISIBLE_DEVICES=N` on the command.
If no GPU is specified, check `CLAUDE.md` or `AGENTS.md` for the assigned GPU indices for the current host.

**GPU is session-sticky**: once a GPU index is set by ANY ik skill (ik:exec, ik:bench, ik:optimize), ALL subsequent ik skill invocations in the same session MUST use that same GPU — unless the user explicitly provides a new `--gpu` value to override it.

## Actions

All commands run from the project root:

```bash
cd /home/zhenc/kernel_lab
```

### Compile

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -c "
from cuda_exec.tasks import compile_endpoint
from cuda_exec.models import CompileRequest
import json

req = CompileRequest(
    metadata={'run_tag': 'optim_001', 'version': 'v1', 'direction_id': 7, 'direction_slug': 'vector-add', 'turn': 1},
    reference_files={'cutedsl.py': open('data/fixtures/sm90/vecadd/cutedsl.py').read()},
    generated_files={'generated.cu': open('data/generated/sm90/vecadd/generated.cu').read()},
)
resp = compile_endpoint(req)
print(json.dumps(resp.model_dump(), indent=2, default=str))
"
```

Returns: `all_ok`, `artifacts` (ptx, sass, resource_usage), `tool_outputs` (nvcc/ptxas stderr).

### Trial

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -c "
from cuda_exec.tasks import trial_endpoint
from cuda_exec.models import TrialRequest
import json

req = TrialRequest(
    metadata={'run_tag': 'optim_001', 'version': 'v1', 'direction_id': 7, 'direction_slug': 'vector-add', 'turn': 1},
    configs=json.load(open('data/fixtures/sm90/vecadd/configs.json')),
)
resp = trial_endpoint(req)
print(json.dumps(resp.model_dump(), indent=2, default=str))
"
```

Returns: `all_ok`, `configs` with per-config `status`, `correctness`, `performance`.

### Profile

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -c "
from cuda_exec.tasks import profile_endpoint
from cuda_exec.models import ProfileRequest
import json

req = ProfileRequest(
    metadata={'run_tag': 'optim_001', 'version': 'v1', 'direction_id': 7, 'direction_slug': 'vector-add', 'turn': 1},
    configs={'vec1d-n65536': {'shape': [65536], 'rank': 1, 'input_size': 65536, 'shape_kind': '1d'}},
    side='generated',
)
resp = profile_endpoint(req)
print(json.dumps(resp.model_dump(), indent=2, default=str))
"
```

Returns: `all_ok`, `configs` with per-config `status`, `summary` (NCU metrics).

## Workflow

1. **Compile** once per turn with reference + generated source files
2. **Trial** against selected configs — check correctness and latency
3. **Profile** selectively (1-2 configs) — NCU hardware metrics

## Rules

- Compile exactly once per turn before trial or profile
- New source code requires a new turn (increment `metadata.turn`)
- Old turns are immutable — never recompile on a previous turn number
- One compile fans out to many trial/profile calls with different configs

## Metadata

Every action requires a `metadata` dict:
```json
{
  "run_tag": "optim_001",
  "version": "v1",
  "direction_id": 7,
  "direction_slug": "vector-add",
  "turn": 1
}
```

## Available Fixtures

| Fixture | Description | Path |
|---------|-------------|------|
| `vecadd` | BF16 vector addition | `data/fixtures/{arch}/vecadd/` |
| `matmul` | Matrix multiplication | `data/fixtures/{arch}/matmul/` |
| `fa4` | Flash Attention v4 | `data/fixtures/{arch}/fa4/` |
