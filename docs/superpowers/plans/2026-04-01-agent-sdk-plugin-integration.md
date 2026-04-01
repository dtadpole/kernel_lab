# Agent SDK Plugin Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate CUDA and KB plugins plus superpowers workflow into the Agent SDK runtime.

**Architecture:** MCP for remote CUDA service calls; CLI (via Bash) for local data inspection and KB doc search; superpowers SKILL.md files loaded on-demand via Read at phase transitions. Edit/Write restricted to fixture directory via PreToolUse hook.

**Tech Stack:** Python 3.12, claude-agent-sdk, argparse, json, pathlib

**Spec:** `docs/superpowers/specs/2026-04-01-agent-sdk-plugin-integration-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `cuda_agent/skills.py` | Create | Load plugin SKILL.md content; discover superpowers install paths |
| `cuda_agent/inspect_cli.py` | Create | CLI for reading local data store (compile/evaluate/profile/raw) |
| `cuda_agent/prompts.py` | Rewrite | Build system prompt from skills + phases + platform adaptation |
| `cuda_agent/agent.py` | Modify | Add allowed_tools, edit restriction hook, use new build_system_prompt |
| `cuda_agent/tests/__init__.py` | Create | Test package marker |
| `cuda_agent/tests/conftest.py` | Create | Shared test fixtures (sample data store, tmp dirs) |
| `cuda_agent/tests/test_skills.py` | Create | Tests for skills.py |
| `cuda_agent/tests/test_inspect_cli.py` | Create | Tests for inspect_cli.py |
| `cuda_agent/tests/test_prompts.py` | Create | Tests for prompts.py |
| `cuda_agent/tests/test_hooks.py` | Create | Tests for edit path restriction hook |

---

### Task 1: Skill Loader (`cuda_agent/skills.py`)

**Files:**
- Create: `cuda_agent/skills.py`
- Create: `cuda_agent/tests/__init__.py`
- Create: `cuda_agent/tests/test_skills.py`

- [ ] **Step 1: Create test package**

Create `cuda_agent/tests/__init__.py` (empty file).

- [ ] **Step 2: Write failing tests for skills.py**

```python
# cuda_agent/tests/test_skills.py
"""Tests for cuda_agent.skills — skill file loader."""

from pathlib import Path

import pytest

from cuda_agent.skills import load_plugin_skill, superpowers_base_dir, superpowers_skill_path

_REPO_ROOT = Path(__file__).resolve().parents[2]


class TestLoadPluginSkill:
    def test_loads_cuda_exec_skill(self):
        content = load_plugin_skill("cuda", "exec")
        assert "name: exec" in content
        assert "compile" in content.lower()

    def test_loads_cuda_inspect_skill(self):
        content = load_plugin_skill("cuda", "inspect")
        assert "name: inspect" in content

    def test_loads_kb_docs_skill(self):
        content = load_plugin_skill("kb", "docs")
        assert "name: docs" in content
        assert "doc_retrieval" in content

    def test_nonexistent_skill_raises(self):
        with pytest.raises(FileNotFoundError):
            load_plugin_skill("cuda", "nonexistent")


class TestSuperpowersPath:
    def test_base_dir_exists(self):
        base = superpowers_base_dir()
        assert base.is_dir()

    def test_skill_path_returns_existing_file(self):
        path = superpowers_skill_path("brainstorming")
        assert Path(path).is_file()

    def test_skill_path_has_skill_md_suffix(self):
        path = superpowers_skill_path("writing-plans")
        assert path.endswith("SKILL.md")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/centos/kernel_lab && python -m pytest cuda_agent/tests/test_skills.py -v`
Expected: `ModuleNotFoundError: No module named 'cuda_agent.skills'`

- [ ] **Step 4: Implement skills.py**

```python
# cuda_agent/skills.py
"""Load skill content from plugin directories and superpowers."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PLUGIN_DIR = _REPO_ROOT / "plugins"

_SUPERPOWERS_MARKER = "superpowers@claude-plugins-official"
_INSTALLED_PLUGINS = Path.home() / ".claude" / "plugins" / "installed_plugins.json"


def load_plugin_skill(plugin: str, skill: str) -> str:
    """Read a plugin SKILL.md and return its full text content."""
    path = _PLUGIN_DIR / plugin / "skills" / skill / "SKILL.md"
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def superpowers_base_dir() -> Path:
    """Discover the superpowers install path from installed_plugins.json."""
    data = json.loads(_INSTALLED_PLUGINS.read_text(encoding="utf-8"))
    entries = data["plugins"].get(_SUPERPOWERS_MARKER)
    if not entries:
        raise FileNotFoundError(
            f"superpowers plugin not found in {_INSTALLED_PLUGINS}"
        )
    return Path(entries[0]["installPath"])


def superpowers_skill_path(skill: str) -> str:
    """Return the absolute path to a superpowers SKILL.md."""
    return str(superpowers_base_dir() / "skills" / skill / "SKILL.md")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/centos/kernel_lab && python -m pytest cuda_agent/tests/test_skills.py -v`
Expected: all 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add cuda_agent/skills.py cuda_agent/tests/__init__.py cuda_agent/tests/test_skills.py
git commit -m "feat(agent): add skill loader for plugin and superpowers SKILL.md files"
```

---

### Task 2: Inspect CLI (`cuda_agent/inspect_cli.py`)

**Files:**
- Create: `cuda_agent/tests/conftest.py`
- Create: `cuda_agent/tests/test_inspect_cli.py`
- Create: `cuda_agent/inspect_cli.py`

- [ ] **Step 1: Create shared test fixtures**

```python
# cuda_agent/tests/conftest.py
"""Shared test fixtures for cuda_agent tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def data_store(tmp_path: Path) -> Path:
    """Create a sample data store with compile/evaluate/profile responses."""
    turn_dir = tmp_path / "turn_1"
    turn_dir.mkdir()

    # Compile response
    compile_resp = {
        "all_ok": True,
        "attempt": 1,
        "artifacts": {
            "ptx": {"content": ".version 8.0\n.target sm_90\n.entry kernel() {}"},
            "sass": {"content": "MOV R0, R1;\nIADD R2, R0, R1;"},
            "resource_usage": {"content": "REG:32 SMEM:1024 STACK:0"},
            "binary": {"encoding": "base64", "content": "AQID..."},
        },
        "tool_outputs": {
            "nvcc": {"content": "nvcc: success"},
            "ptxas": {"content": "ptxas info: 32 registers"},
        },
    }
    (turn_dir / "compile.attempt_001.response.json").write_text(
        json.dumps(compile_resp, indent=2)
    )
    (turn_dir / "compile.attempt_001.request.json").write_text(
        json.dumps({"metadata": {"turn": 1}}, indent=2)
    )

    # Evaluate response
    eval_resp = {
        "all_ok": True,
        "configs": {
            "vec1d-n65536": {
                "status": "ok",
                "correctness": {"passed": True, "max_abs_error": 0.0},
                "performance": {"latency_ms": {"median": 0.05}},
            },
            "tensor2d-1024x1024": {
                "status": "ok",
                "correctness": {"passed": True, "max_abs_error": 1e-6},
                "performance": {"latency_ms": {"median": 0.12}},
            },
        },
    }
    (turn_dir / "evaluate.attempt_001.response.json").write_text(
        json.dumps(eval_resp, indent=2)
    )

    # Profile response
    profile_resp = {
        "all_ok": True,
        "configs": {
            "vec1d-n65536": {
                "status": "ok",
                "summary": {"side": "generated", "ncu_profiled": True},
            },
        },
    }
    (turn_dir / "profile.attempt_001.response.json").write_text(
        json.dumps(profile_resp, indent=2)
    )

    return tmp_path
```

- [ ] **Step 2: Write failing tests for inspect CLI**

```python
# cuda_agent/tests/test_inspect_cli.py
"""Tests for cuda_agent.inspect_cli — local data store CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run_inspect(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "cuda_agent.inspect_cli", *args],
        capture_output=True, text=True,
    )


class TestCompile:
    def test_field_ptx(self, data_store: Path):
        r = _run_inspect("compile", "--data-dir", str(data_store), "--turn", "1", "--field", "ptx")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["all_ok"] is True
        assert ".target sm_90" in data["ptx"]

    def test_field_resource_usage(self, data_store: Path):
        r = _run_inspect("compile", "--data-dir", str(data_store), "--turn", "1", "--field", "resource_usage")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "REG:32" in data["resource_usage"]

    def test_field_all(self, data_store: Path):
        r = _run_inspect("compile", "--data-dir", str(data_store), "--turn", "1", "--field", "all")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "artifacts" in data

    def test_missing_turn(self, data_store: Path):
        r = _run_inspect("compile", "--data-dir", str(data_store), "--turn", "99", "--field", "ptx")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "error" in data


class TestEvaluate:
    def test_all_configs(self, data_store: Path):
        r = _run_inspect("evaluate", "--data-dir", str(data_store), "--turn", "1")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert len(data["configs"]) == 2

    def test_filter_config(self, data_store: Path):
        r = _run_inspect("evaluate", "--data-dir", str(data_store), "--turn", "1",
                         "--config", "vec1d-n65536")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert len(data["configs"]) == 1
        assert "vec1d-n65536" in data["configs"]


class TestProfile:
    def test_all_configs(self, data_store: Path):
        r = _run_inspect("profile", "--data-dir", str(data_store), "--turn", "1")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["configs"]["vec1d-n65536"]["summary"]["ncu_profiled"] is True


class TestRaw:
    def test_response_side(self, data_store: Path):
        r = _run_inspect("raw", "--data-dir", str(data_store), "--turn", "1",
                         "--stage", "compile", "--side", "response")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["all_ok"] is True

    def test_request_side(self, data_store: Path):
        r = _run_inspect("raw", "--data-dir", str(data_store), "--turn", "1",
                         "--stage", "compile", "--side", "request")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "metadata" in data
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/centos/kernel_lab && python -m pytest cuda_agent/tests/test_inspect_cli.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Implement inspect_cli.py**

See implementation in spec. Core logic:
- argparse with subcommands: compile, evaluate, profile, raw
- Each reads `<data-dir>/turn_<T>/<stage>.attempt_<NNN>.response.json`
- Extracts requested fields
- Prints JSON to stdout
- `--attempt` defaults to latest (scans directory for highest attempt number)

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/centos/kernel_lab && python -m pytest cuda_agent/tests/test_inspect_cli.py -v`
Expected: all 9 tests PASS

- [ ] **Step 6: Commit**

```bash
git add cuda_agent/inspect_cli.py cuda_agent/tests/conftest.py cuda_agent/tests/test_inspect_cli.py
git commit -m "feat(agent): add inspect CLI for local data store retrieval"
```

---

### Task 3: System Prompt Builder (`cuda_agent/prompts.py`)

**Files:**
- Rewrite: `cuda_agent/prompts.py`
- Create: `cuda_agent/tests/test_prompts.py`

- [ ] **Step 1: Write failing tests**

```python
# cuda_agent/tests/test_prompts.py
"""Tests for cuda_agent.prompts — system prompt builder."""

from cuda_agent.prompts import build_system_prompt, format_initial_prompt
from cuda_agent.task import OptimizationTask


def _make_task(**overrides) -> OptimizationTask:
    defaults = dict(
        run_tag="test_run",
        version="v1",
        direction_id=7,
        direction_slug="vecadd",
        reference_files={"ref.py": "# ref"},
        initial_generated_files={"gen.cu": "// gen"},
        configs={"cfg1": {"shape": [1024]}},
    )
    defaults.update(overrides)
    return OptimizationTask(**defaults)


class TestBuildSystemPrompt:
    def test_contains_role(self):
        prompt = build_system_prompt(_make_task())
        assert "CUDA kernel optimization" in prompt

    def test_contains_three_phases(self):
        prompt = build_system_prompt(_make_task())
        assert "Phase 1" in prompt
        assert "Phase 2" in prompt
        assert "Phase 3" in prompt
        assert "brainstorming" in prompt

    def test_contains_superpowers_paths(self):
        prompt = build_system_prompt(_make_task())
        assert "SKILL.md" in prompt

    def test_contains_platform_adaptation(self):
        prompt = build_system_prompt(_make_task())
        assert "Platform Adaptation" in prompt

    def test_contains_cuda_exec_skill(self):
        prompt = build_system_prompt(_make_task())
        assert "compile" in prompt.lower()
        assert "evaluate" in prompt.lower()

    def test_contains_inspect_cli_docs(self):
        prompt = build_system_prompt(_make_task())
        assert "inspect_cli" in prompt
        assert "vecadd" in prompt  # data-dir should contain direction_slug

    def test_contains_kb_docs_skill(self):
        prompt = build_system_prompt(_make_task())
        assert "doc_retrieval" in prompt

    def test_data_dir_uses_task_metadata(self):
        prompt = build_system_prompt(_make_task(run_tag="R", version="V",
                                                direction_id=3, direction_slug="S"))
        assert "R/V/3_S" in prompt


class TestFormatInitialPrompt:
    def test_contains_metadata(self):
        prompt = format_initial_prompt(_make_task())
        assert "test_run" in prompt
        assert "vecadd" in prompt

    def test_contains_code(self):
        prompt = format_initial_prompt(_make_task())
        assert "# ref" in prompt
        assert "// gen" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/centos/kernel_lab && python -m pytest cuda_agent/tests/test_prompts.py -v`
Expected: FAIL — `build_system_prompt` not found

- [ ] **Step 3: Rewrite prompts.py**

Replace the static `SYSTEM_PROMPT` constant with `build_system_prompt(task)`.
Keep `format_initial_prompt(task)` unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/centos/kernel_lab && python -m pytest cuda_agent/tests/test_prompts.py -v`
Expected: all 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add cuda_agent/prompts.py cuda_agent/tests/test_prompts.py
git commit -m "feat(agent): build system prompt from skills + superpowers phases"
```

---

### Task 4: Agent Configuration (`cuda_agent/agent.py`)

**Files:**
- Modify: `cuda_agent/agent.py`
- Create: `cuda_agent/tests/test_hooks.py`

- [ ] **Step 1: Write failing tests for edit path restriction hook**

```python
# cuda_agent/tests/test_hooks.py
"""Tests for agent.py hooks — edit path restriction."""

import asyncio
from pathlib import Path

from cuda_agent.agent import _make_restrict_edit_path


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestRestrictEditPath:
    def test_allows_file_inside_allowed_dir(self, tmp_path: Path):
        hook = _make_restrict_edit_path(str(tmp_path))
        result = _run(hook(
            {"tool_input": {"file_path": str(tmp_path / "gen.cu")}},
            None, None,
        ))
        decision = getattr(result, "hookSpecificOutput", None)
        # No deny decision = allowed
        assert decision is None or getattr(decision, "permissionDecision", None) != "deny"

    def test_denies_file_outside_allowed_dir(self, tmp_path: Path):
        hook = _make_restrict_edit_path(str(tmp_path / "allowed"))
        result = _run(hook(
            {"tool_input": {"file_path": "/etc/passwd"}},
            None, None,
        ))
        decision = result.hookSpecificOutput
        assert decision.permissionDecision == "deny"

    def test_denies_parent_traversal(self, tmp_path: Path):
        allowed = tmp_path / "sub"
        allowed.mkdir()
        hook = _make_restrict_edit_path(str(allowed))
        result = _run(hook(
            {"tool_input": {"file_path": str(allowed / ".." / "escape.txt")}},
            None, None,
        ))
        decision = result.hookSpecificOutput
        assert decision.permissionDecision == "deny"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/centos/kernel_lab && python -m pytest cuda_agent/tests/test_hooks.py -v`
Expected: FAIL — `_make_restrict_edit_path` not found

- [ ] **Step 3: Add edit restriction hook and update agent options**

Add `_make_restrict_edit_path()` to agent.py.
Update `run_optimization()` to use `build_system_prompt(task)` and `allowed_tools`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/centos/kernel_lab && python -m pytest cuda_agent/tests/test_hooks.py -v`
Expected: all 3 tests PASS

- [ ] **Step 5: Run all tests**

Run: `cd /home/centos/kernel_lab && python -m pytest cuda_agent/tests/ -v`
Expected: all tests across all 4 test files PASS

- [ ] **Step 6: Commit**

```bash
git add cuda_agent/agent.py cuda_agent/tests/test_hooks.py
git commit -m "feat(agent): add edit path restriction hook, wire up plugin integration"
```
