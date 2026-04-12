# Plan: Solver Stubbornness Enhancement

**Goal**: Solver commits to optimization directions, decomposes when stuck, and receives contextual Steward guidance at key operation points.
**Architecture**: Three-layer (L1 prompt, L2 Steward, L3 Workshop enforcement) + Direction system + wave-first journal
**Tech Stack**: Python 3.12, claude_agent_sdk, asyncio
**Design**: `docs/plans/2026-04-11-solver-stubbornness-design.md`

---

## Step 1: Solver prompt — persistence + decomposition

**File**: `conf/agent/prompts/solver.md`

### 1a. Add persistence principles to Key Principles section (after line 9)

```
- **Stay the course** — compilation errors mean bugs, not wrong direction. Fix the bugs.
- **Big architecture changes take time** — 5-10 iterations to get right.
  Do NOT judge them by the first 2 attempts. Decompose and persist.
- If stuck 15+ minutes on a sub-problem, call ask_supervisor for global perspective.
```

### 1b. Replace "After 4 consecutive failures" block in Phase 7 (line 399-401)

Replace:
```
### After 4 consecutive failures:
Try a fundamentally different architecture (e.g., switch from 1-WG to
warp-specialization, or change tile sizes, or restructure the pipeline).
```

With the DECOMPOSE methodology from the design doc (4-step breakdown).

### 1c. Add Direction section after Phase 7 (before "## Correctness First")

Add the `## Direction` section with `set_direction` usage instructions from design doc.

### 1d. Verify: read the modified file, check no broken formatting

---

## Step 2: Steward prompt — methodologist role

**File**: `conf/agent/prompts/steward.md`

### 2a. Add "Core Role" section after "## Who You Are" (line 9)

```
## Core Role: Methodologist with Global Perspective

You see the FULL trajectory. Solver sees only its current problem.
You do NOT write code. You tell Solver:
- What to focus on
- When to decompose
- When to persist
- When to step back
```

### 2b. Verify: read the modified file

---

## Step 3: Direction review response prompt (new file)

**File**: `conf/agent/response_prompts/direction_review.md`

### 3a. Write the prompt

Content from design doc — evaluates specificity, evidence quality, opportunity realism. Responds APPROVED / REVISE / REJECTED.

### 3b. Verify: file exists and is well-formed

---

## Step 4: Update progress_check prompt

**File**: `conf/agent/response_prompts/progress_check.md`

### 4a. Rewrite with direction awareness

Add direction context variables (`{direction_name}`, `{direction_description}`, `{direction_opportunity}`) and the 4 check patterns from design doc (drift, stuck loop, tunnel vision, healthy progress). Keep the existing good parts (tool call activity check, plan before code).

### 4b. Verify: read and check variable references match what Workshop will provide

---

## Step 5: Storage restructure — wave-first

**File**: `agents/storage.py`

### 5a. Modify WaveStorage.__init__ path construction

Current: `config.journal_path / agent_name / task_slug / wave_id`
New: `config.journal_path / wave_id`

Remove `task_slug` from path. Remove `agent_name` from top-level path.

### 5b. Add solver/steward/directions subdirectory creation

In `__init__`, after `self.wave_dir.mkdir()`:
```python
(self.wave_dir / "solver").mkdir(exist_ok=True)
(self.wave_dir / "steward").mkdir(exist_ok=True)
(self.wave_dir / "directions").mkdir(exist_ok=True)
```

### 5c. Update file paths — move transcript/events/logs under solver/

Current paths like `self.wave_dir / "transcript.md"` →
`self.wave_dir / "solver" / "transcript.md"`

Update: `events_path`, `transcript_path`, `log_stdin`, `log_stdout`, `log_stderr`.

### 5d. Rename heartbeat → heartbeat.json

`self.wave_dir / "heartbeat"` → `self.wave_dir / "heartbeat.json"`

### 5e. Add directions path property

```python
@property
def directions_path(self) -> Path:
    return self.wave_dir / "directions"
```

### 5f. Add method to create Steward sub-storage

```python
def steward_storage(self, scenario: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = self.wave_dir / "steward" / f"{scenario}_{ts}"
    path.mkdir(parents=True, exist_ok=True)
    return path
```

### 5g. Verify: instantiate WaveStorage, check directory tree created correctly

---

## Step 6: Direction data model

**File**: `agents/direction.py` (new)

### 6a. Write Direction class

```python
import json
from pathlib import Path

def write_direction(directions_dir: Path, seq: int, direction: dict) -> Path:
    name = direction.get("name", "unknown")
    path = directions_dir / f"{seq:03d}_{name}.json"
    path.write_text(json.dumps(direction, indent=2, ensure_ascii=False))
    return path

def read_active_direction(directions_dir: Path) -> dict | None:
    if not directions_dir.exists():
        return None
    files = sorted(directions_dir.glob("*.json"))
    if not files:
        return None
    last = json.loads(files[-1].read_text())
    return last

def next_seq(directions_dir: Path) -> int:
    if not directions_dir.exists():
        return 1
    return len(list(directions_dir.glob("*.json"))) + 1
```

### 6b. Verify: test read/write roundtrip in Python REPL

---

## Step 7: Steward — add review_direction method

**File**: `agents/steward.py`

### 7a. Add review_direction method

```python
async def review_direction(
    self,
    direction: dict,
    transcript_path: str,
    is_change: bool = False,
    current_direction: dict | None = None,
) -> StewardResponse:
    context = {
        "direction_json": json.dumps(direction, indent=2),
        "is_change": str(is_change),
        "current_direction_json": json.dumps(current_direction, indent=2) if current_direction else "None",
        "transcript_path": transcript_path,
    }
    verdict = await self.router.respond("direction_review", context)
    return _to_steward_response(verdict)
```

### 7b. Add review_direction_alignment method (for diffusion)

```python
async def review_direction_alignment(
    self,
    direction: dict,
    trigger_type: str,
    trigger_result: str,
    transcript_path: str,
) -> StewardResponse:
    context = {
        "direction_json": json.dumps(direction, indent=2),
        "trigger_type": trigger_type,
        "trigger_result": trigger_result[:2000],
        "transcript_path": transcript_path,
    }
    verdict = await self.router.respond("direction_alignment", context)
    return _to_steward_response(verdict)
```

### 7c. Add direction_alignment response prompt

**File**: `conf/agent/response_prompts/direction_alignment.md`

Write the diffusion prompt from design doc.

### 7d. Update response_router.py — add context templates for new scenarios

Add `"direction_review"` and `"direction_alignment"` to `CONTEXT_TEMPLATES` and `SCENARIO_MAX_TURNS`.

### 7e. Verify: check steward.py and response_router.py have no import errors

---

## Step 8: Runner — add set_direction MCP tool

**File**: `agents/runner.py`

### 8a. Add set_direction to _build_mcp_tools

Find the MCP tools section (where `ask_supervisor`, `request_formal_bench` are defined). Add:

```python
@tool
def set_direction(direction_json: str) -> str:
    """Set your optimization direction after brainstorming.
    Pass a JSON string with: name, description, opportunity, evidence, ideas."""
    ...  # delegates to handler
```

### 8b. Add handler callback plumbing

Add `on_set_direction` to the EventHandler protocol or handle it like `on_ask`.

### 8c. Verify: tool appears in MCP server tool list

---

## Step 9: Workshop — direction state + set_direction handler

**File**: `agents/workshop.py`

### 9a. Add direction state to WorkshopState

```python
current_direction: dict | None = None
```

### 9b. Add direction handling in on_ask (or new handler)

Implement `_handle_set_direction` from design doc:
- All calls → Steward review
- APPROVED → write file + set state
- REVISE/REJECTED → return guidance, don't persist

### 9c. Load direction on wave start

On new wave, check previous wave's `directions/` for active direction. If found, inherit (copy file with new seq).

### 9d. Verify: manual test — call set_direction, check file written

---

## Step 10: Workshop — pre-tool direction gate

**File**: `agents/runner.py` (in `_build_hooks` / `on_pre_tool_use`)

### 10a. Add direction check to pre_tool_use hook

In the existing `on_pre_tool_use` function, after `_check_tool_rules`:

```python
# Direction gate: no writes without approved direction
if tool_name in ("Write", "Edit") and not self._has_active_direction():
    path = tool_input.get("file_path", "")
    if path.endswith((".cu", ".cuh", ".py")):
        return {"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "You must set_direction first."
        }}
if tool_name == "Bash" and not self._has_active_direction():
    cmd = tool_input.get("command", "")
    if _is_write_command(cmd):
        return {"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "You must set_direction first."
        }}
```

### 10b. Add `_has_active_direction` and `_is_write_command` helpers

### 10c. Verify: manually trigger Write without direction, check it's blocked

---

## Step 11: Workshop — activate progress_check

**File**: `agents/workshop.py`

### 11a. Change progress_check handler (line 766-769)

Replace:
```python
elif event.alert_type == "progress_check":
    print(f"[Workshop] Progress check at {elapsed} — heartbeat OK")
    return "continue"
```

With actual Steward call + direction context injection.

### 11b. Pass direction context to Steward check_progress

Update the `check_progress` call to include direction data in the context.

### 11c. Verify: check that progress_check actually triggers Steward

---

## Step 12: Workshop — diffusion trigger detection

**File**: `agents/workshop.py`

### 12a. Add _detect_trigger method

Pattern-match on tool_name/tool_input to detect compile, trial, profile, code_write.

### 12b. Add _steward_direction_diffusion async method

Fire Steward `review_direction_alignment`, inject response into Solver via `client.query()`.

### 12c. Wire into post_tool_use hook

In the existing `on_post_tool_use`, after logging, call `_detect_trigger` and fire async task if trigger detected.

### 12d. Add rate limiting (_too_recent check, 2 min minimum)

### 12e. Verify: manually trigger a compile, check Steward review fires

---

## Step 13: End-to-end verification

### 13a. Start Workshop with a test task

```bash
cd /home/zhenc/kernel_lab
.venv/bin/python -m agents.launcher --kernel matmul --gpu 4
```

### 13b. Verify the full flow

1. Solver starts → no direction → Write blocked
2. Solver calls set_direction → Steward reviews → APPROVED → file written
3. Solver writes code → allowed
4. Solver compiles → diffusion review fires → guidance injected
5. Progress check → Steward checks direction alignment
6. Solver tries to change direction → Steward reviews → DENIED

---

## Task Dependencies

| Group | Steps | Can Parallelize | Files Touched |
|-------|-------|-----------------|---------------|
| 1 | Steps 1, 2, 3, 4 | Yes (all independent prompt files) | solver.md, steward.md, response_prompts/ |
| 2 | Steps 5, 6 | Yes (independent) | storage.py, direction.py (new) |
| 3 | Step 7 | No (depends on Group 1 prompts + Group 2 direction model) | steward.py, response_router.py, response_prompts/ |
| 4 | Step 8 | No (depends on Group 2) | runner.py |
| 5 | Steps 9, 10, 11, 12 | Partially (9 first, then 10-12 in parallel) | workshop.py, runner.py |
| 6 | Step 13 | No (depends on all above) | — |
