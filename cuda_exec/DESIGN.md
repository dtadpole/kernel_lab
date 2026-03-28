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

## 4. Code-level compile vs config-level evaluate/profile

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

## 5. Runtime config convention

`evaluate` and `profile` accept `configs[]`.
Each config includes at least:

- `config_id`

Optional standard fields:

- `num_layers`
- `embedding_size`
- `num_heads`
- `causal`
- `extra`

Conceptual example:

```json
{
  "metadata": { "...": "..." },
  "configs": [
    {
      "config_id": "fa4_causal_l12_e4096_h32",
      "num_layers": 12,
      "embedding_size": 4096,
      "num_heads": 32,
      "causal": true
    },
    {
      "config_id": "fa4_noncausal_l12_e4096_h32",
      "num_layers": 12,
      "embedding_size": 4096,
      "num_heads": 32,
      "causal": false
    }
  ]
}
```

For each config, the service writes a config record under `state/configs/` and exports runtime
information through environment variables such as:

- `CUDA_EXEC_CONFIG_ID`
- `CUDA_EXEC_CONFIG_PATH`
- `CUDA_EXEC_CONFIG_JSON`
- `CUDA_EXEC_NUM_LAYERS`
- `CUDA_EXEC_EMBEDDING_SIZE`
- `CUDA_EXEC_NUM_HEADS`
- `CUDA_EXEC_CAUSAL`

---

## 6. Attempt convention

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

## 7. Stage outputs

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

## 8. Response convention

Public API responses should stay stage-specific and minimal.
The response is a **summary**, not a mirror of the full runtime directory.

`state` is internal-first. It is kept for compile/evaluate/profile bookkeeping and inspection,
but it should not be part of the default public response.

### Compile response

Return only:

- `metadata`
- `ok`
- `attempt`
- `binary_path`
- `log_path`
- `stdout_path`
- `stderr_path`

### Evaluate response

Return only:

- `metadata`
- `ok`
- `attempt`
- `results[]`

Each evaluate result contains:

- `config_id`
- `ok`
- `log_path`
- `stdout_path`
- `stderr_path`

### Profile response

Return only:

- `metadata`
- `ok`
- `attempt`
- `results[]`

Each profile result contains:

- `config_id`
- `ok`
- `report_path`
- `log_path`
- `stdout_path`
- `stderr_path`

### Execute response

Return only:

- `metadata`
- `ok`
- `attempt`
- `log_path`
- `stdout_path`
- `stderr_path`

### What should not be exposed in the default public response

Do not expose generic heavy response fields by default, such as:

- full `files[]`
- full `artifacts[]`
- nested `config_results[]` with inline stdout/stderr
- large inline file contents

The detailed runtime information already exists on disk under `workspace/`, `artifacts/`, `logs/`, and `state/`.
The public response should only point to the important paths.

---

## 9. CWD convention

The service guarantees that the **initial cwd** for launched processes is:

```text
<turn_root>/workspace/
```

That is the service guarantee.

The service does not guarantee that an invoked program will remain inside that directory if the
program itself changes cwd, writes to absolute paths, or spawns child processes with different
path behavior.

---

## 10. Caller-facing simplicity rules

To keep agent behavior simple:

- do not ask the agent to choose artifact ids in V0
- do not ask the agent to choose returned file sets in V0
- let compile take code inputs
- let evaluate/profile take runtime configs
- let the service own layout, logging, attempts, and workflow rules

---

## 11. Documentation split

- `DESIGN.md` = detailed source of truth
- `README.md` = short entrypoint
- `AGENTS.md` = repo-level instructions only
