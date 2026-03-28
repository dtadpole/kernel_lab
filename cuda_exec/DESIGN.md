# cuda_exec V0 Design

This document defines the **early-version convention** for `cuda_exec`.

The main goal is to keep the agent-facing API as small and predictable as possible.
The service is intentionally **convention-driven**. If a behavior can be fixed by
convention, it should not be pushed onto the agent as an extra input field.

## Design goals

1. Keep agent inputs minimal.
2. Make workflow ordering explicit and enforceable.
3. Make turns immutable.
4. Persist all runtime outputs locally on the machine.
5. Keep the local directory layout stable and easy to inspect.

---

## 1. Workflow convention

`cuda_exec` uses a fixed workflow:

1. `compile`
2. `evaluate` (optional)
3. `profile` (optional)
4. `execute` for special/tooling cases only

### Workflow rules

- `compile` must happen **first** within a turn.
- `compile` may run **only once** within a turn.
- `evaluate` and `profile` are only valid **after** `compile` has already run in the same turn.
- If `evaluate` or `profile` is called before `compile`, the service must return a workflow error telling the caller to run `compile` first.
- If a caller wants to retry with different inputs, it must start a **new turn**.
- Old turns are **immutable**. A turn is not edited in place.
- In V0, new uploaded files after `compile` always imply a **new turn**.

### Why this is the convention

This keeps the API simple for agents:

- the agent does not choose a target artifact
- the agent does not choose return-file sets
- the agent only chooses the workflow step to run

---

## 2. Agent-facing request model

### `POST /compile`

Required:

- `metadata`
- `original_files`
- `generated_files`

Optional:

- `timeout_seconds`

### `POST /evaluate`

Required:

- `metadata`

Optional:

- `timeout_seconds`

### `POST /profile`

Required:

- `metadata`

Optional:

- `timeout_seconds`

### `POST /execute`

Required:

- `metadata`
- `command`

Optional:

- `env`
- `timeout_seconds`

### Intentionally omitted in V0

The following are intentionally **not** agent inputs in V0:

- `target_artifact_id`
- `return_files`

These are fixed by convention instead of being chosen by the caller.

---

## 3. Compile artifact convention

In V0, `compile` exposes exactly **one public compile artifact**:

- `compile:primary_binary`

That is the executable produced by the hardened compile flow for the turn.

`evaluate` and `profile` both consume this artifact by convention.
The agent does not specify which artifact to use.

Other files such as logs, state manifests, and profile reports exist, but they are treated as
**convention-defined stage outputs**, not agent-selectable compile inputs.

---

## 4. Turn and directory convention

The runtime root is:

```text
~/.cuda_exec/<run_tag>/<version>/<direction_id>_<direction_slug>/turn_<turn>/
```

On this machine:

```text
/home/centos/.cuda_exec/<run_tag>/<version>/<direction_id>_<direction_slug>/turn_<turn>/
```

Inside each turn root, `cuda_exec` creates a fixed layout:

```text
turn_<turn>/
  workspace/
    original/
    generated/
  outputs/
  logs/
  profiles/
  state/
  tmp/
```

### Meaning of each directory

- `workspace/`: execution cwd and staged source files
- `workspace/original/`: copied original inputs
- `workspace/generated/`: copied generated inputs
- `outputs/`: primary runtime outputs from compile
- `logs/`: local persisted logs for each stage
- `profiles/`: profiler outputs
- `state/`: machine-readable workflow state manifests
- `tmp/`: scratch files

### Important naming note

- `turn_root` means the whole per-turn directory.
- `workspace` means only the execution working directory under that turn.

`workspace` should **not** be used as the name for the whole turn root.

---

## 5. Local logging convention

For every stage that runs locally on the machine, `cuda_exec` must persist runtime output under
that turn root.

This applies to:

- `compile`
- `evaluate`
- `profile`
- `execute`

Each stage writes:

- a combined log file
- a stdout file
- a stderr file

### Stage log files

#### Compile

```text
logs/compile.log
logs/compile.stdout
logs/compile.stderr
```

#### Evaluate

```text
logs/evaluate.log
logs/evaluate.stdout
logs/evaluate.stderr
```

#### Profile

```text
logs/profile.log
logs/profile.stdout
logs/profile.stderr
```

#### Execute

```text
logs/execute.log
logs/execute.stdout
logs/execute.stderr
```

This logging convention exists even if stdout/stderr are also returned in the HTTP response.
The local copies are the source of truth for later inspection.

---

## 6. State-file convention

Each workflow stage writes a state file under `state/`.

### Compile

```text
state/compile.json
```

Contains at least:

- request metadata
- workflow invariants
- compile status
- selected source
- primary artifact convention

### Evaluate

```text
state/evaluate.json
```

Contains at least:

- request metadata
- workflow invariants
- evaluate status
- compile input used

### Profile

```text
state/profile.json
```

Contains at least:

- request metadata
- workflow invariants
- profile status
- compile input used
- profiler identity (`ncu` in V0)

---

## 7. Response convention

All command-style responses return:

- `metadata`
- `ok`
- `kind`
- `command`
- `turn_root`
- `workspace_path`
- `returncode`
- `duration_seconds`
- `output.stdout`
- `output.stderr`
- `artifacts[]`
- `files[]`

### Why `artifacts[]` still exists in V0

`artifacts[]` is still useful as an output description, even though the agent does not choose
artifact ids as input parameters.

In V0, the important part is:

- the agent does **not** pass artifact-selection fields
- the service may still describe its outputs using a simple artifact convention

---

## 8. Fixed returned files by stage

Returned files are fixed by convention in V0.
The caller does not choose them.

### Compile returns by convention

- `outputs/<stem>`
- `logs/compile.log`
- `logs/compile.stdout`
- `logs/compile.stderr`
- `state/compile.json`

### Evaluate returns by convention

- `logs/evaluate.log`
- `logs/evaluate.stdout`
- `logs/evaluate.stderr`
- `state/evaluate.json`

### Profile returns by convention

- `profiles/<stem>-ncu.ncu-rep`
- `logs/profile.log`
- `logs/profile.stdout`
- `logs/profile.stderr`
- `state/profile.json`

### Execute returns by convention

- `logs/execute.log`
- `logs/execute.stdout`
- `logs/execute.stderr`

---

## 9. UI / control-plane display convention

The UI should present the workflow and returned outputs explicitly.

### Workflow banner

The UI should make these rules visible:

- compile first
- compile once per turn
- new files require a new turn
- old turns are immutable

### Per-stage output display

For each stage, the UI should show:

- stage status
- return code
- stdout
- stderr
- returned files
- important stage outputs

### Stage-specific highlights

#### Compile
Show:

- primary binary path
- compile log paths
- compile state path

#### Evaluate
Show:

- input binary used
- evaluate log paths
- evaluate state path

#### Profile
Show:

- input binary used
- profiler report path
- profile log paths
- profile state path

#### Execute
Show:

- command actually run
- execute log paths

---

## 10. Runtime output layout example

Example turn root:

```text
/home/centos/.cuda_exec/agent_a/v1/7_convention/turn_3/
  workspace/
    original/
      baseline.cu
    generated/
      candidate.cu
  outputs/
    candidate
  logs/
    compile.log
    compile.stdout
    compile.stderr
    evaluate.log
    evaluate.stdout
    evaluate.stderr
    profile.log
    profile.stdout
    profile.stderr
  profiles/
    candidate-ncu.ncu-rep
  state/
    compile.json
    evaluate.json
    profile.json
  tmp/
```

This is the reference layout that the control plane and docs should describe.
