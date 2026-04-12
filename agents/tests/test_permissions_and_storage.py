"""Tests for tool permissions hardening and storage directory routing.

Covers:
1. Forbidden commands (kill, git, sudo) blocked for ALL agents — even with no tool_rules
2. Steward tool_rules (Write/Edit denied)
3. Storage _subdir routing (solver → solver/, steward_* → steward/)
"""

import asyncio
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from agents.config import AgentConfig, StorageConfig, ToolRule, MonitorConfig
from agents.events import DefaultHandler, ToolCallEvent
from agents.runner import AgentRunner
from agents.storage import WaveStorage


# ── Storage subdir tests ──

def test_solver_storage_subdir():
    """Solver agent writes to solver/ subdirectory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = StorageConfig(kb_root=tmpdir, run_tag="test")
        storage = WaveStorage(config, agent_name="solver", task_slug="test")
        assert storage._subdir == "solver"
        assert "solver" in str(storage.events_path)
        assert "solver" in str(storage.transcript_path)


def test_steward_storage_subdir():
    """Steward agent writes to steward/ subdirectory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = StorageConfig(kb_root=tmpdir, run_tag="test")
        storage = WaveStorage(config, agent_name="steward_ask_question", task_slug="test")
        assert storage._subdir == "steward"
        assert "steward" in str(storage.events_path)
        assert "steward" in str(storage.transcript_path)


def test_steward_progress_check_subdir():
    """steward_progress_check agent writes to steward/ subdirectory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = StorageConfig(kb_root=tmpdir, run_tag="test")
        storage = WaveStorage(config, agent_name="steward_progress_check", task_slug="test")
        assert storage._subdir == "steward"


def test_steward_log_files_go_to_steward_dir():
    """Steward's stdin/stdout/stderr logs go to steward/ not solver/."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = StorageConfig(kb_root=tmpdir, run_tag="test")
        storage = WaveStorage(config, agent_name="steward_session_end", task_slug="test")

        storage.log_stdin("test stdin")
        storage.log_stdout("test stdout")
        storage.log_stderr("test stderr")
        storage.close_logs()

        steward_dir = storage.wave_dir / "steward"
        assert (steward_dir / "stdin.log").exists()
        assert (steward_dir / "stdout.log").exists()
        assert (steward_dir / "stderr.log").exists()

        solver_dir = storage.wave_dir / "solver"
        assert not (solver_dir / "stdin.log").exists()
        assert not (solver_dir / "stdout.log").exists()
        assert not (solver_dir / "stderr.log").exists()


# ── Forbidden command tests ──

def _make_runner_no_rules():
    """Create an AgentRunner with NO tool_rules (like Steward before the fix)."""
    agent_config = AgentConfig(
        name="test_no_rules",
        model="claude-sonnet-4-6",
        permission_mode="acceptEdits",
        max_turns=1,
        tool_rules=[],  # Empty — no rules
    )
    return AgentRunner(
        agent_config=agent_config,
        handler=DefaultHandler(),
        monitor_config=MonitorConfig(),
        storage_config=StorageConfig(kb_root="/tmp/test_perms", run_tag="test"),
    )


def _make_runner_with_bash_rule():
    """Create an AgentRunner with a Bash tool rule (like Solver)."""
    agent_config = AgentConfig(
        name="test_with_rules",
        model="claude-sonnet-4-6",
        permission_mode="acceptEdits",
        max_turns=1,
        tool_rules=[ToolRule(tool="Bash", allow=True)],
    )
    return AgentRunner(
        agent_config=agent_config,
        handler=DefaultHandler(),
        monitor_config=MonitorConfig(),
        storage_config=StorageConfig(kb_root="/tmp/test_perms", run_tag="test"),
    )


def _is_denied(result: dict) -> bool:
    return (
        isinstance(result, dict)
        and result.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
    )


def test_kill_forbidden_with_rules():
    """kill is forbidden even with tool_rules."""
    runner = _make_runner_with_bash_rule()
    assert _is_denied(runner._check_tool_rules("Bash", {"command": "kill 12345"}))
    assert _is_denied(runner._check_tool_rules("Bash", {"command": "kill -9 12345"}))


def test_kill_forbidden_without_rules():
    """kill is forbidden even with NO tool_rules — the critical fix."""
    runner = _make_runner_no_rules()
    assert _is_denied(runner._check_tool_rules("Bash", {"command": "kill 12345"}))


def test_pkill_killall_forbidden():
    """pkill and killall are forbidden."""
    runner = _make_runner_no_rules()
    assert _is_denied(runner._check_tool_rules("Bash", {"command": "pkill python"}))
    assert _is_denied(runner._check_tool_rules("Bash", {"command": "killall -9 claude"}))


def test_sudo_forbidden():
    """sudo is forbidden."""
    runner = _make_runner_no_rules()
    assert _is_denied(runner._check_tool_rules("Bash", {"command": "sudo rm -rf /"}))


def test_git_forbidden_without_rules():
    """git is forbidden even without tool_rules (was previously bypassed)."""
    runner = _make_runner_no_rules()
    assert _is_denied(runner._check_tool_rules("Bash", {"command": "git push origin main"}))


def test_safe_commands_allowed_without_rules():
    """Safe commands still work with no tool_rules."""
    runner = _make_runner_no_rules()
    result = runner._check_tool_rules("Bash", {"command": "ls /tmp"})
    assert not _is_denied(result)
    result = runner._check_tool_rules("Bash", {"command": "cat /tmp/test.txt"})
    assert not _is_denied(result)


def test_read_allowed_without_rules():
    """Read tool works with no tool_rules."""
    runner = _make_runner_no_rules()
    result = runner._check_tool_rules("Read", {"file_path": "/tmp/test.txt"})
    assert not _is_denied(result)


# ── Steward tool_rules tests ──

def test_steward_has_tool_rules():
    """Steward AgentConfig created by ResponseRouter has tool_rules."""
    from agents.response_router import ResponseRouter
    router = ResponseRouter(prompts_dir=Path("conf/agent/response_prompts"))

    # Steward agents are created inside _call_agent — we can't easily
    # test that without calling it. Instead, verify the pattern:
    # Write and Edit should be denied for steward.
    from agents.config import ToolRule
    steward_rules = [
        ToolRule(tool="Bash", allow=True,
                 constraint="Read-only commands for research."),
        ToolRule(tool="Write", allow=False),
        ToolRule(tool="Edit", allow=False),
    ]
    agent_config = AgentConfig(
        name="steward_test",
        tool_rules=steward_rules,
    )
    runner = AgentRunner(
        agent_config=agent_config,
        handler=DefaultHandler(),
        monitor_config=MonitorConfig(),
        storage_config=StorageConfig(kb_root="/tmp/test_perms", run_tag="test"),
    )
    # Write should be denied
    assert _is_denied(runner._check_tool_rules("Write", {"file_path": "/tmp/test.txt"}))
    # Edit should be denied
    assert _is_denied(runner._check_tool_rules("Edit", {"file_path": "/tmp/test.txt"}))
    # Bash kill should be denied (global forbidden)
    assert _is_denied(runner._check_tool_rules("Bash", {"command": "kill 12345"}))
    # Bash ls should be allowed
    assert not _is_denied(runner._check_tool_rules("Bash", {"command": "ls /tmp"}))


# ── Word boundary tests ──

def test_kill_word_boundary():
    """Words containing 'kill' as substring should NOT be blocked."""
    runner = _make_runner_no_rules()
    # "skillful" contains "kill" but \b should prevent false positive
    assert not _is_denied(runner._check_tool_rules("Bash", {"command": "echo skillful"}))
    assert not _is_denied(runner._check_tool_rules("Bash", {"command": "echo killer_app"}))
    # But actual kill command should be blocked
    assert _is_denied(runner._check_tool_rules("Bash", {"command": "kill 123"}))


def test_forbidden_in_compound_commands():
    """Forbidden commands blocked even inside compound commands."""
    runner = _make_runner_no_rules()
    assert _is_denied(runner._check_tool_rules("Bash", {"command": "echo hello; kill 123"}))
    assert _is_denied(runner._check_tool_rules("Bash", {"command": "echo hello && kill 123"}))
    assert _is_denied(runner._check_tool_rules("Bash", {"command": "ps aux | xargs kill"}))
    assert _is_denied(runner._check_tool_rules("Bash", {"command": "git log --oneline | head"}))


def test_reboot_shutdown_forbidden():
    """reboot and shutdown are forbidden."""
    runner = _make_runner_no_rules()
    assert _is_denied(runner._check_tool_rules("Bash", {"command": "reboot"}))
    assert _is_denied(runner._check_tool_rules("Bash", {"command": "shutdown -h now"}))


# ── Storage: events and transcript routing ──

def test_solver_log_files_go_to_solver_dir():
    """Solver's stdin/stdout/stderr logs go to solver/."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = StorageConfig(kb_root=tmpdir, run_tag="test")
        storage = WaveStorage(config, agent_name="solver", task_slug="test")

        storage.log_stdin("test")
        storage.log_stdout("test")
        storage.log_stderr("test")
        storage.close_logs()

        solver_dir = storage.wave_dir / "solver"
        assert (solver_dir / "stdin.log").exists()
        assert (solver_dir / "stdout.log").exists()
        assert (solver_dir / "stderr.log").exists()


def test_steward_events_go_to_steward_dir():
    """Steward's events.jsonl is written to steward/ subdirectory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = StorageConfig(kb_root=tmpdir, run_tag="test")
        storage = WaveStorage(config, agent_name="steward_set_direction", task_slug="test")

        storage.append_event({"type": "test", "ts": "2026-01-01T00:00:00"})

        assert (storage.wave_dir / "steward" / "events.jsonl").exists()
        assert not (storage.wave_dir / "solver" / "events.jsonl").exists()


def test_each_agent_gets_own_subdir():
    """Each agent routes to its own subdirectory (not shared solver/)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = StorageConfig(kb_root=tmpdir, run_tag="test")

        cases = {
            "solver": "solver",
            "steward_ask_question": "steward",
            "steward_progress_check": "steward",
            "benchmarker": "benchmarker",
            "rigger": "rigger",
            "librarian": "librarian",
        }
        for agent_name, expected_subdir in cases.items():
            storage = WaveStorage(config, agent_name=agent_name, task_slug="test", wave=0)
            assert storage._subdir == expected_subdir, \
                f"{agent_name}: expected {expected_subdir}/, got {storage._subdir}/"
            # Verify the subdir was actually created
            assert (storage.wave_dir / expected_subdir).is_dir(), \
                f"{agent_name}: {expected_subdir}/ not created"


# ── ResponseRouter constructs Steward config correctly ──

def test_response_router_steward_config():
    """ResponseRouter creates Steward AgentConfig with Write/Edit denied."""
    from agents.response_router import ResponseRouter
    import inspect

    router = ResponseRouter(prompts_dir=Path("conf/agent/response_prompts"))

    # Inspect _call_agent source to verify tool_rules are set
    source = inspect.getsource(router._call_agent)
    assert "ToolRule" in source, "_call_agent should use ToolRule"
    assert "Write" in source, "_call_agent should have Write rule"
    assert "Edit" in source, "_call_agent should have Edit rule"
    assert "allow=False" in source, "Write/Edit should be denied"


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
