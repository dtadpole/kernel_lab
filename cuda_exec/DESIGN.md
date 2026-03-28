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
    original/
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
- `evaluate` and `profile` require compile state from the same turn
- if new code inputs arrive after compile, start a new turn
- old turns are immutable

---

## 4. Compile input convention

`compile` now takes inline file maps instead of file lists.

Both of these request fields are:

- `Dict[relative_path, content]`

Specifically:

- `original_files: Dict[str, str]`
- `generated_files: Dict[str, str]`

Rules:

- keys must be relative paths
- keys may include folder names
- keys must not be absolute paths
- keys must not contain `.` or `..` path traversal segments
- values are file contents

Conceptual example:

```json
{
  "metadata": { "...": "..." },
  "generated_files": {
    "kernels/candidate.cu": "extern \"C\" __global__ void ..."
  },
  "original_files": {
    "reference/baseline.cu": "extern \"C\" __global__ void ..."
  }
}
```

The service writes these inputs under:

```text
workspace/inputs/original/<relative_path>
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
- `CUDA_EXEC_PARAM_<KEY>` for simple top-level scalar config values
- `CUDA_EXEC_EXTRA_<KEY>` for values nested under a config `extra` object

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
logs/profile.attempt_001.config_fa4_causal_l12_e4096_h32.log
artifacts/profile.attempt_001.config_fa4_causal_l12_e4096_h32.ncu-rep
```

Even though compile runs only once per turn, it still uses `attempt_001` for naming uniformity.

---

## 8. Stage outputs

### Compile

Kept results:

- compiled binary in `artifacts/`

Process output:

- `logs/compile.attempt_001.log`
- `logs/compile.attempt_001.stdout`
- `logs/compile.attempt_001.stderr`

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
- `artifacts: Dict[relative_path, file_payload]`
- `logs: Dict[relative_path, file_payload]`

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

Preferred layout for preserved runs:

- place run roots under `~/temp/`
- prefix the run directory name with `YYYY-MM-DD-HH-MM-`
- then use a kebab-case slug plus PID in the run directory name
- when a temporary uv-managed environment is created there, prefer `<run-dir>/.venv`

Retention policy direction:

- do **not** immediately delete the run environment or intermediate outputs after a successful run
- preserve request/response logs, service logs, runtime root contents, and temporary `.venv` for later inspection
- handle cleanup through a separate retention process, for example deleting runs older than 7 days
- the helper script is `cuda_exec/scripts/prune_temp_runs.py`
- its default behavior deletes preserved run directories older than 7 days
- `--dry-run` is supported
- keep rules: skip directories whose name contains `keep`, or that contain keep-marker files such as `KEEP`

This keeps recent integration trajectories available without committing them into Git.

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
    "direction_slug": "fa4",
    "turn": 3
  },
  "timeout_seconds": 180,
  "original_files": {
    "reference/baseline.cu": "extern \"C\" __global__ void baseline() {}"
  },
  "generated_files": {
    "kernels/candidate.cu": "extern \"C\" __global__ void candidate() {}"
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
    "direction_slug": "fa4",
    "turn": 3
  },
  "all_ok": true,
  "attempt": 1,
  "artifacts": {
    "artifacts/compile.attempt_001.candidate.bin": {
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
    "direction_slug": "fa4",
    "turn": 3
  },
  "all_ok": true,
  "attempt": 1,
  "configs": {
    "fa4-causal-l12-e4096-h32": {
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
        "logs/evaluate.attempt_001.config_fa4-causal-l12-e4096-h32.stdout": {
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
    "direction_slug": "fa4",
    "turn": 3
  },
  "all_ok": true,
  "attempt": 1,
  "configs": {
    "fa4-causal-l12-e4096-h32": {
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
        "artifacts/profile.attempt_001.config_fa4-causal-l12-e4096-h32.ncu-rep": {
          "content": "<base64>",
          "encoding": "base64",
          "truncated": false
        }
      },
      "logs": {
        "logs/profile.attempt_001.config_fa4-causal-l12-e4096-h32.log": {
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
    "direction_slug": "fa4",
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
    "direction_slug": "fa4",
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
