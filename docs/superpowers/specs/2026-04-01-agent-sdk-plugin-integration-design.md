# Agent SDK Plugin Integration Design

> **Goal:** Integrate CUDA and KB plugins plus superpowers workflow into the
> Claude Agent SDK runtime, with clean separation between remote MCP tools,
> local CLI tools, and on-demand skill loading.
>
> **Architecture:** MCP for remote CUDA service calls; CLI (via Bash) for local
> data inspection and KB documentation search; superpowers SKILL.md files loaded
> on-demand via Read at phase transitions.

---

## 1. Components

### 1.1 Skill Loader (`cuda_agent/skills.py` — new)

Utility module for loading skill content from two sources:

- **Plugin skills** — read from `plugins/<plugin>/skills/<skill>/SKILL.md`
  relative to repo root.
- **Superpowers skills** — discover install path from
  `~/.claude/plugins/installed_plugins.json`, return absolute SKILL.md path
  for the agent to `Read` on demand.

```python
def load_plugin_skill(plugin: str, skill: str) -> str
def superpowers_skill_path(skill: str) -> str
```

### 1.2 Inspect CLI (`cuda_agent/inspect_cli.py` — new)

Standalone CLI that reads from the local data store written by the CUDA MCP
server's action tools. Replaces the MCP `get_*_data` tools for Agent SDK use.

**Subcommands:**

| Command    | Reads from                              | Key options              |
|------------|-----------------------------------------|--------------------------|
| `compile`  | `turn_<T>/compile.attempt_NNN.response` | `--field` (ptx/sass/resource_usage/tool_outputs/all) |
| `evaluate` | `turn_<T>/evaluate.attempt_NNN.response`| `--config` (slug filter) |
| `profile`  | `turn_<T>/profile.attempt_NNN.response` | `--config` (slug filter) |
| `raw`      | `turn_<T>/<stage>.attempt_NNN.*`        | `--stage`, `--attempt`, `--side` (request/response) |

**Common options:** `--data-dir` (path to run directory), `--turn` (turn number),
`--attempt` (defaults to latest).

**Output:** JSON to stdout. The agent sees it as Bash tool output.

**Invocation:**
```bash
python -m cuda_agent.inspect_cli compile \
    --data-dir ~/.cuda_agent/R/V/D_S --turn 1 --field ptx
```

### 1.3 System Prompt (`cuda_agent/prompts.py` — rewrite)

Built dynamically from task metadata and skill content:

```
┌─ Section 1: Agent role and identity
├─ Section 2: Three-phase workflow (superpowers skill paths)
├─ Section 3: Platform adaptation (Claude Code → Agent SDK)
├─ Section 4: cuda:exec skill content (embedded, from SKILL.md)
├─ Section 5: Inspect CLI usage (generated, with data-dir for this task)
└─ Section 6: kb:docs skill content (embedded, from SKILL.md)
```

**Section 2 detail — three-phase workflow:**

The agent works in three phases. At the start of each phase it uses Read to
load the corresponding superpowers SKILL.md and follows its instructions.

- Phase 1 (Analysis & Strategy): `brainstorming/SKILL.md`
- Phase 2 (Optimization Plan): `writing-plans/SKILL.md`
- Phase 3 (Iterative Execution): `executing-plans/SKILL.md`

Since the agent runs autonomously (no human in the loop), the system prompt
includes adaptation notes:
- Make decisions yourself — do not wait for user input.
- Skip interactive-only concepts (visual companion, AskUserQuestion).
- Skip sub-skills that don't apply (TDD, finishing-branch, git worktrees).

**Section 3 detail — platform adaptation:**

| Claude Code concept   | Agent SDK equivalent                  |
|-----------------------|---------------------------------------|
| `Skill` tool          | `Read` tool on SKILL.md file          |
| `TodoWrite`           | Track progress in conversation text   |
| `EnterPlanMode`       | Plan directly in conversation         |
| `AskUserQuestion`     | Not available; decide autonomously    |
| Visual companion      | Not available; use text               |

### 1.4 Agent Configuration (`cuda_agent/agent.py` — modify)

```python
ClaudeAgentOptions(
    model=cfg.agent.model,
    system_prompt=build_system_prompt(task),
    max_turns=max_turns,
    permission_mode=cfg.agent.permission_mode,
    allowed_tools=["Read", "Write", "Edit", "Glob", "Grep", "Bash", "Agent"],
    hooks={
        "PreToolUse": [
            HookMatcher(matcher="Edit|Write", hooks=[restrict_edit_path]),
            HookMatcher(matcher="Bash", hooks=[deny_cuda_toolkit]),
        ],
        "PostToolUse": [
            HookMatcher(matcher=None, hooks=[log_tool_use]),
        ],
    },
    mcp_servers={
        "cuda": {
            "command": sys.executable,
            "args": [cuda_mcp_server_path],
            "env": exec_env,
        },
    },
)
```

Changes from current code:
- Add `allowed_tools` (currently absent = all allowed).
- Add Edit/Write path restriction hook.
- `system_prompt` from `build_system_prompt(task)` instead of static constant.
- No `setting_sources` — system prompt is fully self-contained.

### 1.5 Edit Path Restriction Hook

PreToolUse hook on `Edit|Write` that denies writes outside the task's fixture
directory. Uses `os.path.realpath()` to prevent symlink/`..` escape.

The allowed directory is derived from the task: `conf/fixtures/<direction_slug>/`
relative to repo root. Configurable via `cfg.agent.edit_allowed_dir` override.

---

## 2. What Does NOT Change

| Component                     | Reason                                          |
|-------------------------------|-------------------------------------------------|
| `plugins/cuda/mcp_server.py`  | Unchanged — keeps all 9 tools for Claude Code compatibility |
| `plugins/cuda/skills/`        | Read as-is by skill loader                      |
| `plugins/kb/skills/`          | Read as-is by skill loader                      |
| `plugins/kb/` (no MCP server) | KB stays CLI-only, accessed via Bash             |
| `cuda_agent/task.py`          | Task dataclass unchanged                        |
| `cuda_agent/config.py`        | Config loader unchanged                         |
| `cuda_agent/cli.py`           | CLI entry unchanged                             |
| `cuda_agent/blocked_tools.json` | Blocklist unchanged                            |

---

## 3. Tool Delivery Summary

| Tool                          | Delivery   | Source                              |
|-------------------------------|------------|-------------------------------------|
| compile, evaluate, profile    | MCP        | `plugins/cuda/mcp_server.py` (remote HTTP) |
| execute, read_file            | MCP        | `plugins/cuda/mcp_server.py` (remote HTTP) |
| inspect compile/evaluate/etc  | CLI (Bash) | `cuda_agent/inspect_cli.py` (local JSON)   |
| doc find/read/browse          | CLI (Bash) | `python -m doc_retrieval` (local index)    |
| Read, Write, Edit, Glob, etc  | Built-in   | Agent SDK                           |

---

## 4. Skill Loading Summary

| Skill             | When loaded            | How loaded                      |
|-------------------|------------------------|---------------------------------|
| `cuda:exec`       | Always (system prompt) | Embedded via `load_plugin_skill` |
| `cuda:inspect`    | Always (system prompt) | Generated CLI docs with data-dir |
| `kb:docs`         | Always (system prompt) | Embedded via `load_plugin_skill` |
| `brainstorming`   | Phase 1 start          | Agent calls `Read` on SKILL.md   |
| `writing-plans`   | Phase 2 start          | Agent calls `Read` on SKILL.md   |
| `executing-plans` | Phase 3 start          | Agent calls `Read` on SKILL.md   |

---

## 5. New Files

```
cuda_agent/
├── skills.py          # ~40 lines: load_plugin_skill(), superpowers_skill_path()
├── inspect_cli.py     # ~150 lines: argparse CLI, JSON file reading, field extraction
```

## 6. Modified Files

```
cuda_agent/
├── agent.py           # Add allowed_tools, edit restriction hook, use build_system_prompt()
├── prompts.py         # Rewrite: build_system_prompt() with skill embedding + phase workflow
```
