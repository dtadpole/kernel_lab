# cuda_exec Design

This file is the **single source of truth** for `cuda_exec` conventions.

`README.md` should stay short. Detailed behavior, workflow rules, naming rules, and
runtime layout all live here.

The design goal is to keep the agent-facing API and the agent mental model as small
as possible.

---

## 1. Core mental model

Use only these four concepts when thinking about runtime files:

1. **workspace = inputs + scratch**
2. **artifacts = kept results**
3. **logs = process output**
4. **state = workflow record**

That is the intended mental model for agents and for the UI.

### What each means

#### `workspace`

`workspace` is the execution working area.
It is **not** "input only".

It contains:

- staged input files
- scratch/intermediate files created during execution
- the current cwd for launched processes

Simple rule:

- if a file is only needed to make this run work, it can stay in `workspace`

#### `artifacts`

`artifacts` means **kept results**.
These are files worth preserving, showing in the UI, or reusing later.

Examples:

- compiled binaries
- `.ptx` / `.cubin` outputs worth keeping
- profiler reports
- any explicit result file the service wants to retain

Simple rule:

- if a file should be kept after the run and treated as a result, it belongs in `artifacts`

#### `logs`

`logs` means **process output**.

This includes:

- combined log files
- stdout captures
- stderr captures

#### `state`

`state` means **workflow record**.

This includes:

- metadata for the turn
- workflow rules/invariants
- stage status
- selected inputs
- config lists
- per-config results
- references to kept outputs

---

## 2. Current on-disk layout vs conceptual layout

The **conceptual** layout should be thought of as:

```text
turn_<turn>/
  workspace/
  artifacts/
  logs/
  state/
```

That is the recommended mental model.

### Current implementation note

The current code still materializes kept results in multiple directories:

- `outputs/`
- `profiles/`

and may also create:

- `tmp/`

For agent reasoning and UI explanation, treat:

- `outputs/` + `profiles/` as the **artifact bucket**
- `tmp/` as an implementation detail, not a core concept

So even before the code is simplified further, the agent mental model should remain:

- `workspace`
- `artifacts`
- `logs`
- `state`

---

## 3. Turn root convention

The turn root is:

```text
~/.cuda_exec/<run_tag>/<version>/<direction_id>_<direction_slug>/turn_<turn>/
```

On this machine:

```text
/home/centos/.cuda_exec/<run_tag>/<version>/<direction_id>_<direction_slug>/turn_<turn>/
```

`turn_root` means the whole per-turn directory.

`workspace` means only the working directory under that turn root.

---

## 4. Workflow convention

The workflow is fixed:

1. `compile`
2. `evaluate` (optional)
3. `profile` (optional)
4. `execute` for special/tooling cases only

### Workflow rules

- `compile` must happen first within a turn
- `compile` may run only once within a turn
- `evaluate` and `profile` require compile state from the same turn
- if `evaluate` or `profile` is called before `compile`, the service returns a workflow error
- if new files are uploaded after `compile`, the caller must start a new turn
- old turns are immutable

The point is to keep the caller simple:

- the agent chooses the workflow step
- the service owns the workflow convention

---

## 5. Compile vs runtime config

This is an important design rule.

### Compile is code-level

`compile` is about building the code artifact for the turn.

In the intended design:

- compile runs once per turn
- compile does **not** vary across model/runtime configs
- the compiled code should be generic enough to support all supported configs for that turn

### Evaluate / profile are config-level

`evaluate` and `profile` run the compiled artifact against one or more **runtime configs**.

This means:

- a single compile may be followed by many evaluate runs
- a single compile may be followed by many profile runs
- these runs differ by config, not by source code

Examples of config fields:

- transformer layer count
- embedding size
- number of heads
- whether causal masking is enabled

For FA4-style usage, one compiled artifact may be evaluated/profiled against 8 configs,
for example:

- 4 causal configs
- 4 non-causal configs

That is normal and should be represented explicitly in the design.

---

## 6. Config convention

`config` is a first-class runtime concept for `evaluate` and `profile`.

### Design intent

- code stays the same
- config changes
- `evaluate` and `profile` consume configs
- config does not imply recompilation unless the source code itself changes

### Recommended request shape

`compile` should continue to take code inputs only.

`evaluate` and `profile` should accept a list of runtime configs, conceptually:

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

The important rule is not the exact schema shape. The important rule is:

- compile is code-level
- evaluate/profile are config-level

### Why this helps

This keeps recompilation logic simple:

- one code artifact
- many runtime configs
- many evaluate/profile results

That matches the FA4-style use case and avoids unnecessary compile churn.

---

## 7. Attempt convention

Repeated stage calls should not overwrite one another.

The recommended naming convention is:

- `compile.attempt_001.*`
- `evaluate.attempt_001.*`
- `evaluate.attempt_002.*`
- `profile.attempt_001.*`
- `execute.attempt_001.*`

Even though compile only runs once per turn, using `attempt_001` there too keeps naming uniform.

### Why attempts exist

- avoid overwriting local files
- keep history for repeated `evaluate`, `profile`, and `execute`
- make UI ordering simpler
- keep file naming uniform across stages

### Timestamp guidance

Timestamps are still useful, but they should usually live in `state` rather than being the primary filename key.

Recommended pattern:

- filenames use `attempt_###`
- state records `started_at` / `finished_at`

---

## 8. Config results and file layout

A stage attempt may contain multiple configs.

The simple conceptual model is:

- one stage attempt
- many config results under that attempt

### Recommended state behavior

`state/<stage>.attempt_###.json` should record:

- metadata
- attempt number
- timestamps
- config list
- per-config result summaries
- references to kept outputs

### Recommended artifact behavior

If a config produces a kept result, its filename should carry both stage attempt and config identity.

Examples:

```text
artifacts/profile.attempt_001.config_fa4_causal_l12_e4096_h32.ncu-rep
artifacts/profile.attempt_001.config_fa4_noncausal_l12_e4096_h32.ncu-rep
```

If a config produces only transient data, that data can remain in `workspace`.

---

## 9. Local process output convention

For every stage that launches a local process, the service persists process output under `logs`.

This applies to:

- `compile`
- `evaluate`
- `profile`
- `execute`

Recommended files per stage attempt:

```text
logs/<stage>.attempt_###.log
logs/<stage>.attempt_###.stdout
logs/<stage>.attempt_###.stderr
```

These files exist even if stdout/stderr are also returned in the HTTP response.

### `execute` specifically

`execute` is the most generic stage, so its contract should stay simple:

- process output always goes to `logs`
- meaningful kept result files should be written explicitly to `artifacts`
- scratch/intermediate files can remain in `workspace`

---

## 10. CWD convention

When the service launches a process, the **initial cwd** is the current turn's `workspace`.

That is the service guarantee.

More precisely:

- the service resolves the current turn root from metadata
- it sets `cwd = <turn_root>/workspace/` when launching the process

The service does **not** guarantee that the invoked program will remain inside that directory if the program itself:

- changes cwd internally
- writes to absolute paths
- spawns child processes with different path behavior

So the correct guarantee is:

- initial cwd is the current turn's `workspace`

---

## 11. Caller-facing simplicity rules

To keep agent behavior simple:

- do not ask the agent to choose artifact ids in V0
- do not ask the agent to choose returned file sets in V0
- let the agent provide code at compile time
- let the agent provide runtime configs at evaluate/profile time
- let the service own file placement, logging, and workflow rules

---

## 12. Document structure

To reduce duplication:

- `DESIGN.md` is the detailed design source of truth
- `README.md` should be short and point here
- `AGENTS.md` should contain only repo-level instructions and high-level conventions
