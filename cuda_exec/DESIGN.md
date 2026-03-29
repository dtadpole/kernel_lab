# cuda_exec Design

This file is the source of truth for `cuda_exec` conventions.

The design goal is simple:

- keep agent inputs small
- keep runtime layout clean
- make workflow rules explicit
- separate scratch files from kept results
- support many runtime configs per compile

---

## 1. Core mental model

Use these four concepts:

1. **workspace = inputs + scratch**
2. **artifacts = kept results**
3. **logs = process output**
4. **state = workflow record**

### `workspace`

`workspace` is the working area for the current turn.

It contains:

- staged inputs
- scratch/intermediate files
- the initial cwd for launched processes

### `artifacts`

`artifacts` are files worth keeping.

Examples:

- compiled binaries
- profiler reports
- explicit result files worth preserving

### `logs`

`logs` store process output:

- combined logs
- stdout captures
- stderr captures

### `state`

`state` stores workflow records:

- stage manifests
- runtime config records
- per-config results
- references to kept artifacts

---

## 2. Turn-root layout

Turn root:

```text
~/.cuda_exec/<run_tag>/<version>/<direction_id>_<direction_slug>/turn_<turn>/
```

On this machine:

```text
/home/centos/.cuda_exec/<run_tag>/<version>/<direction_id>_<direction_slug>/turn_<turn>/
```

Implemented top-level directories:

```text
turn_<turn>/
  workspace/
  artifacts/
  logs/
  state/
```

Common subpaths inside `workspace/`:

```text
workspace/
  inputs/
    reference/
    generated/
```

`turn_root` means the whole per-turn directory.
`workspace` means only the working directory under that turn root.

---

## 3. Workflow convention

Workflow order:

1. `compile`
2. `evaluate` (optional)
3. `profile` (optional)
4. `execute` for special/tooling cases only

Rules:

- `compile` must run first for a turn
- `compile` may run only once per turn
- do not reuse the same turn to upload a different file set for compile
- if new reference/generated inputs arrive after compile, start a new turn
- `evaluate` and `profile` require compile state from the same turn
- old turns are immutable

---

## 4. Compile input convention

`compile` now takes inline file maps instead of file lists.

Both of these request fields are:

- `Dict[relative_path, content]`

Specifically:

- `reference_files: Dict[str, str]`
- `generated_files: Dict[str, str]`

Rules:

- `reference_files` must be non-empty
- `generated_files` must be non-empty
- keys must be relative paths
- keys may include folder names
- keys must not be absolute paths
- keys must not contain `.` or `..` path traversal segments
- values are file contents
- `generated_files` must contain exactly one `.cu` file
- `generated_files` may also include multiple headers or inline helper files
- `reference_files` may include `.cu` files, but does not require one
- if `generated_files` contains zero or multiple `.cu` files, reject the compile request
- when rejecting multiple generated `.cu` files, the caller-facing guidance should recommend using a generator

Conceptual example:

```json
{
  "metadata": { "...": "..." },
  "generated_files": {
    "kernels/generated.cu": "extern \"C\" __global__ void ..."
  },
  "reference_files": {
    "reference/reference.cu": "extern \"C\" __global__ void ..."
  }
}
```

The service writes these inputs under:

```text
workspace/inputs/reference/<relative_path>
workspace/inputs/generated/<relative_path>
```

---

## 5. Code-level compile vs config-level evaluate/profile

### Compile is code-level

`compile` builds the code artifact for the turn.

- compile happens once per turn
- compile should not vary across runtime configs
- code should be general enough to support all configs for that turn

### Evaluate and profile are config-level

`evaluate` and `profile` run the compiled artifact against one or more runtime configs.

That means one compile can fan out into many configs.

For `evaluate`, the current runtime shape is comparison-first:

- load exactly one reference Python module from `workspace/inputs/reference/`
- require the module contract `Model`, `get_inputs(config)`, and `get_init_inputs()`
- run the reference side through that module contract
- the current vector-add reference fixture now genuinely launches a CuTe DSL kernel from a `@cute.jit` host launcher, rather than falling back to eager `x + y`
- run the generated side through the compiled primary artifact
- persist one kept comparison artifact per config under `artifacts/evaluate.attempt_###.config_<slug>.comparison.json`
- return per-config `correctness`, `performance`, `artifacts`, and `logs`

Example config fields:

- transformer layer count
- embedding size
- number of heads
- whether causal masking is enabled

FA4-style example:

- 4 causal configs
- 4 non-causal configs
- one compile artifact
- many evaluate/profile runs

---

## 6. Runtime config convention

`evaluate` and `profile` accept slug-keyed config maps:

- `configs: Dict[config_slug, Dict[str, Any]]`

The config slug is the stable identity for a config across both request and response.
The config body itself is intentionally flexible and kernel-specific.

In other words:

- the service owns the transport shape of config
- the kernel owns the semantic shape of config

Conceptual example:

```json
{
  "metadata": { "...": "..." },
  "configs": {
    "tensor2d-1024x1024": {
      "shape": [1024, 1024],
      "rank": 2,
      "input_size": 1048576
    },
    "attention-b1-s4096-h32-d128": {
      "batch_size": 1,
      "seq_len": 4096,
      "num_heads": 32,
      "head_dim": 128,
      "causal": true
    }
  }
}
```

For each config, the service writes a config record under `state/configs/` and exports runtime
information through environment variables such as:

- `CUDA_EXEC_CONFIG_ID` (set to the config slug)
- `CUDA_EXEC_CONFIG_PATH`
- `CUDA_EXEC_CONFIG_JSON`
- `CUDA_EXEC_PARAM_<KEY>` for top-level config values; structured values are JSON-encoded when needed
- `CUDA_EXEC_EXTRA_<KEY>` for values nested under a config `extra` object (still supported, but top-level config fields are the preferred shape)

---

## 7. Attempt convention

Stage outputs use uniform attempt naming.

Examples:

```text
state/compile.attempt_001.json
logs/compile.attempt_001.log
logs/compile.attempt_001.stdout
logs/compile.attempt_001.stderr

state/evaluate.attempt_001.json
state/profile.attempt_001.json
```

For config-specific stage runs, logs and kept artifacts carry both the attempt and config identity.

Examples:

```text
logs/evaluate.attempt_001.config_fa4_causal_l12_e4096_h32.log
artifacts/evaluate.attempt_001.config_fa4_causal_l12_e4096_h32.comparison.json
logs/profile.attempt_001.config_fa4_causal_l12_e4096_h32.log
artifacts/profile.attempt_001.config_fa4_causal_l12_e4096_h32.ncu-rep
```

Even though compile runs only once per turn, it still uses `attempt_001` for naming uniformity.

---

## 8. Stage outputs

### Compile

Kept results:

- runnable binary in `artifacts/`
- generated PTX in `artifacts/`
- ptxas-produced CUBIN in `artifacts/`
- resource-usage text report in `artifacts/`
- SASS dump from `nvdisasm` in `artifacts/`

Process output:

- `logs/compile.attempt_001.nvcc-ptx.stdout`
- `logs/compile.attempt_001.nvcc-ptx.stderr`
- `logs/compile.attempt_001.ptxas.stdout`
- `logs/compile.attempt_001.ptxas.stderr`
- `logs/compile.attempt_001.resource-usage.stdout`
- `logs/compile.attempt_001.resource-usage.stderr`

Workflow record:

- `state/compile.attempt_001.json`

### Evaluate

For each config:

- config record under `state/configs/`
- config-specific logs under `logs/`

Workflow record:

- `state/evaluate.attempt_###.json`

### Profile

For each config:

- config record under `state/configs/`
- config-specific logs under `logs/`
- kept NCU report under `artifacts/`

Workflow record:

- `state/profile.attempt_###.json`

### Execute

Process output:

- `logs/execute.attempt_###.log`
- `logs/execute.attempt_###.stdout`
- `logs/execute.attempt_###.stderr`

`execute` does **not** write a stage state file.
It is intentionally treated as a tool-style execution path, not a workflow-record stage.

If `execute` generates meaningful kept results, those files should be written explicitly to `artifacts/`.
Scratch/intermediate files can remain in `workspace/`.

---

## 9. Response convention

Public API responses should stay stage-specific and minimal.

`state` is internal-first. It is kept for compile/evaluate/profile bookkeeping and inspection,
but it should not be part of the default public response.

### Relative-path rule

For public request and response payloads:

- use relative paths only
- include folder names inside the relative path when needed
- do not use absolute paths

### File-return rule

For public responses, files are returned as dictionaries keyed by relative path.

Conceptually:

```json
{
  "relative/path/to/file": {
    "content": "...",
    "encoding": "utf8",
    "truncated": false
  }
}
```

A small refinement is kept for practicality:

- text files use `encoding = "utf8"`
- binary files use `encoding = "base64"`

This keeps the external shape simple while still supporting binary artifacts such as compiled binaries
and profiler reports.

### Compile response

Return only:

- `metadata`
- `all_ok`
- `attempt`
- `artifacts`
- `tool_outputs`

`artifacts` should be a structured object containing the stage-relevant kept outputs when present, especially:

- `binary`
- `ptx`
- `cubin`
- `resource_usage`
- `sass.nvdisasm`

These compile artifacts may be returned as path-only file references when inline content is not required.

`tool_outputs` should be a structured object containing the inline stdout/stderr payloads needed by callers, especially:

- `nvcc_ptx.stdout` / `nvcc_ptx.stderr`
- `ptxas.stdout` / `ptxas.stderr`
- `resource_usage.stdout` / `resource_usage.stderr`
- `nvdisasm.stdout` / `nvdisasm.stderr`

### Evaluate response

Return only:

- `metadata`
- `all_ok`
- `attempt`
- `configs: Dict[config_slug, evaluate_config_output]`

Each evaluate config output contains:

- `status`
- `correctness`
- `performance`
- `logs: Dict[relative_path, file_payload]`

`correctness` should carry structured quality information such as:

- pass/fail
- max / mean absolute error
- absolute variance
- max / mean relative error
- relative variance

`performance` should carry structured timing information such as:

- min / median / max / mean latency
- run count

### Profile response

Return only:

- `metadata`
- `all_ok`
- `attempt`
- `configs: Dict[config_slug, profile_config_output]`

Each profile config output contains:

- `status`
- `summary`
- `artifacts: Dict[relative_path, file_payload]`
- `logs: Dict[relative_path, file_payload]`

`summary` should carry structured profile-level information such as:

- min / median / max / mean latency
- run count
- additional metadata when available

### Execute response

Return only:

- `metadata`
- `all_ok`
- `attempt`
- `logs: Dict[relative_path, file_payload]`

### What should not be exposed in the default public response

Do not expose generic heavy internal structures by default, such as:

- internal `state` paths
- full internal `artifacts[]` catalogs
- generic `files[]`
- internal nested runtime bookkeeping objects

The public response should present only the stage-relevant artifacts and logs in a direct relative-path keyed form.

---

## 10. CWD convention

The service guarantees that the **initial cwd** for launched processes is:

```text
<turn_root>/workspace/
```

That is the service guarantee.

The service does not guarantee that an invoked program will remain inside that directory if the
program itself changes cwd, writes to absolute paths, or spawns child processes with different
path behavior.

### Test isolation and retention note

For integration tests or manual validation runs, the runtime root may be redirected by setting:

- `CUDA_EXEC_ROOT=<temporary-directory>`

Current automated integration-test harness behavior:

- place temporary run roots under `~/temp/`
- create one top-level temp directory per integration-suite invocation
- prefix the run directory name with `YYYY-MM-DD-HH-MM-`
- then use a kebab-case slug plus PID in the run directory name
- if multiple service processes are started during the suite, reuse that same top-level run directory and namespace per-service logs/runtime roots inside it
- use the repo-local Python environment at `cuda_exec/.venv`
- invoke `cuda_exec/scripts/prune_temp_runs.py` before starting the temporary uvicorn service
- terminate the subprocess on teardown
- preserve the temporary run directory, service log, and runtime root for later inspection

If the harness is later moved to fully temporary uv-managed environments, prefer `<run-dir>/.venv`.

Retention helper for preserved runs:

- the helper script is `cuda_exec/scripts/prune_temp_runs.py`
- its default behavior deletes preserved run directories older than 7 days
- `--dry-run` is supported
- keep rules: skip directories whose name contains `keep`, or that contain keep-marker files such as `KEEP`

This keeps recent integration trajectories available for inspection without committing them into Git.

---

## 11. Caller-facing simplicity rules

To keep agent behavior simple:

- do not ask the agent to choose artifact ids in V0
- do not ask the agent to choose returned file sets in V0
- let compile take code inputs
- let evaluate/profile take runtime configs
- let the service own layout, logging, attempts, and workflow rules

---

## 12. JSON examples

### Compile request

```json
{
  "metadata": {
    "run_tag": "agent_a",
    "version": "v1",
    "direction_id": 7,
    "direction_slug": "vector-add",
    "turn": 3
  },
  "timeout_seconds": 180,
  "reference_files": {
    "reference/reference.cu": "extern \"C\" __global__ void reference() {}"
  },
  "generated_files": {
    "kernels/generated.cu": "extern \"C\" __global__ void generated() {}"
  }
}
```

### Compile response

```json
{
  "metadata": {
    "run_tag": "agent_a",
    "version": "v1",
    "direction_id": 7,
    "direction_slug": "vector-add",
    "turn": 3
  },
  "all_ok": true,
  "attempt": 1,
  "artifacts": {
    "artifacts/compile.attempt_001.generated.bin": {
      "content": "<base64>",
      "encoding": "base64",
      "truncated": false
    }
  },
  "logs": {
    "logs/compile.attempt_001.log": {
      "content": "command: ...",
      "encoding": "utf8",
      "truncated": false
    },
    "logs/compile.attempt_001.stdout": {
      "content": "...",
      "encoding": "utf8",
      "truncated": false
    },
    "logs/compile.attempt_001.stderr": {
      "content": "...",
      "encoding": "utf8",
      "truncated": false
    }
  }
}
```

### File read request

```json
{
  "metadata": {
    "run_tag": "agent_a",
    "version": "v1",
    "direction_id": 7,
    "direction_slug": "vector-add",
    "turn": 3
  },
  "path": "artifacts/compile.attempt_001.vector_add_inline_ptx.ptx",
  "max_bytes": 65536
}
```

Response example:

```json
{
  "metadata": {
    "run_tag": "agent_a",
    "version": "v1",
    "direction_id": 7,
    "direction_slug": "vector-add",
    "turn": 3
  },
  "file": {
    "path": "artifacts/compile.attempt_001.vector_add_inline_ptx.ptx",
    "inline": true,
    "content": ".version 8.8\n.target sm_120\n...",
    "encoding": "utf8",
    "truncated": false
  }
}
```

Rules:

- use `POST /files/read`
- `path` must be a relative path under the resolved turn root
- `path` must stay within one of: `artifacts/`, `logs/`, `state/`
- do not allow absolute paths or `..` traversal
- return the requested file as an inline `FilePayload`
- if `max_bytes` is provided, return at most that many bytes in `content`
- when `max_bytes` causes truncation, return `truncated = true`
- if the file does not exist for that turn/path, return `404`

### Reference Python contract

Current contract: reference execution is Python-only and module-based.

A reference Python file must export:

- `class Model(torch.nn.Module)`
- `get_inputs(config: dict) -> list`
- `get_init_inputs(config: dict) -> list` (or equivalent zero-arg init list when config is unused)

`cuda_exec` owns the evaluator/profiler runtime. Reference files do not need to implement
comparison logic themselves. They only need to expose a runnable module interface compatible
with the evaluator, similar to the `kbEval` expectation.

### Evaluate request

Evaluate is always a comparison between `reference` and `generated` for the same config set.

```json
{
  "metadata": {
    "run_tag": "agent_a",
    "version": "v1",
    "direction_id": 7,
    "direction_slug": "vector-add",
    "turn": 4
  },
  "timeout_seconds": 180,
  "configs": {
    "shape-1d-1048576": {
      "shape_kind": "1d",
      "rank": 1,
      "input_size": 1048576,
      "shape": [1048576]
    }
  }
}
```

Response shape per config:

- `status`
- `reference`
  - `correctness`
  - `performance`
  - `logs`
- `generated`
  - `correctness`
  - `performance`
  - `logs`
- `comparison`
  - `correctness_match`
  - `performance`
    - `reference_median_ms`
    - `generated_median_ms`
    - `delta_ms`
    - `speedup`

### Profile request

Profile supports an explicit mode:

- `reference_only`
- `generated_only`
- `dual`

```json
{
  "metadata": {
    "run_tag": "agent_a",
    "version": "v1",
    "direction_id": 7,
    "direction_slug": "vector-add",
    "turn": 4
  },
  "timeout_seconds": 180,
  "mode": "dual",
  "configs": {
    "shape-1d-1048576": {
      "shape_kind": "1d",
      "rank": 1,
      "input_size": 1048576,
      "shape": [1048576]
    }
  }
}
```

Response shape per config:

- `status`
- `reference` (present in `reference_only` and `dual`)
  - `summary`
  - `artifacts`
  - `logs`
- `generated` (present in `generated_only` and `dual`)
  - `summary`
  - `artifacts`
  - `logs`
- `comparison` (present in `dual`)
  - `reference_median_ms`
  - `generated_median_ms`
  - `delta_ms`
  - `speedup`

### Evaluate request

```json
{
  "metadata": {
    "run_tag": "agent_a",
    "version": "v1",
    "direction_id": 7,
    "direction_slug": "vector-add",
    "turn": 3
  },
  "timeout_seconds": 180,
  "configs": {
    "tensor2d-1024x1024": {
      "shape": [1024, 1024],
      "rank": 2,
      "input_size": 1048576
    },
    "tensor3d-64x64x64": {
      "shape": [64, 64, 64],
      "rank": 3,
      "input_size": 262144
    }
  }
}
```

### Evaluate response

```json
{
  "metadata": {
    "run_tag": "agent_a",
    "version": "v1",
    "direction_id": 7,
    "direction_slug": "vector-add",
    "turn": 3
  },
  "all_ok": true,
  "attempt": 1,
  "configs": {
    "tensor2d-1024x1024": {
      "status": "ok",
      "correctness": {
        "metadata": {},
        "passed": true,
        "max_abs_error": 1.2e-6,
        "mean_abs_error": 3.1e-8,
        "abs_variance": 2.5e-14,
        "max_rel_error": 2.4e-5,
        "mean_rel_error": 8.7e-7,
        "rel_variance": 1.3e-11
      },
      "performance": {
        "metadata": {},
        "latency_ms": {
          "min": 0.82,
          "median": 0.89,
          "max": 0.97,
          "mean": 0.89
        },
        "runs": 100
      },
      "logs": {
        "logs/evaluate.attempt_001.config_tensor2d-1024x1024.stdout": {
          "content": "...",
          "encoding": "utf8",
          "truncated": false
        }
      }
    }
  }
}
```

### Profile request

```json
{
  "metadata": {
    "run_tag": "agent_a",
    "version": "v1",
    "direction_id": 7,
    "direction_slug": "vector-add",
    "turn": 3
  },
  "timeout_seconds": 180,
  "configs": {
    "tensor2d-1024x1024": {
      "shape": [1024, 1024],
      "rank": 2,
      "input_size": 1048576
    }
  }
}
```

### Profile response

```json
{
  "metadata": {
    "run_tag": "agent_a",
    "version": "v1",
    "direction_id": 7,
    "direction_slug": "vector-add",
    "turn": 3
  },
  "all_ok": true,
  "attempt": 1,
  "configs": {
    "tensor2d-1024x1024": {
      "status": "ok",
      "summary": {
        "metadata": {},
        "latency_ms": {
          "min": 0.80,
          "median": 0.87,
          "max": 0.95,
          "mean": 0.87
        },
        "runs": 100
      },
      "artifacts": {
        "artifacts/profile.attempt_001.config_tensor2d-1024x1024.ncu-rep": {
          "content": "<base64>",
          "encoding": "base64",
          "truncated": false
        }
      },
      "logs": {
        "logs/profile.attempt_001.config_tensor2d-1024x1024.log": {
          "content": "...",
          "encoding": "utf8",
          "truncated": false
        }
      }
    }
  }
}
```

### Execute request

```json
{
  "metadata": {
    "run_tag": "agent_a",
    "version": "v1",
    "direction_id": 7,
    "direction_slug": "vector-add",
    "turn": 3
  },
  "timeout_seconds": 180,
  "command": ["/usr/local/cuda/bin/nvcc", "--version"],
  "env": {}
}
```

### Execute response

```json
{
  "metadata": {
    "run_tag": "agent_a",
    "version": "v1",
    "direction_id": 7,
    "direction_slug": "vector-add",
    "turn": 3
  },
  "all_ok": true,
  "attempt": 1,
  "logs": {
    "logs/execute.attempt_001.stdout": {
      "content": "Cuda compilation tools, release ...",
      "encoding": "utf8",
      "truncated": false
    },
    "logs/execute.attempt_001.stderr": {
      "content": "",
      "encoding": "utf8",
      "truncated": false
    }
  }
}
```

## 13. Documentation split

- `DESIGN.md` = detailed source of truth
- `README.md` = short entrypoint
- `AGENTS.md` = repo-level instructions only
