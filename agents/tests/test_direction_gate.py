"""Tests for the direction gate pre-tool hook.

Verifies that Write/Edit to watched dirs are blocked when no direction is set,
and allowed when a direction is active.
"""

import asyncio
from dataclasses import dataclass, field
from unittest.mock import MagicMock


@dataclass
class FakeState:
    current_direction: dict | None = None
    mode: str = "exploring"


class FakeHandler:
    """Minimal handler with direction gate attrs."""
    def __init__(self, gate_dirs: list[str], gate_tools: list[str], direction: dict | None = None):
        self._direction_gate_dirs = gate_dirs
        self._direction_gate_tools = gate_tools
        self._direction_gate_message = "You must set_direction first."
        self.state = FakeState(current_direction=direction)

    async def on_tool_call(self, event):
        pass

    async def on_tool_result(self, event):
        pass


def _simulate_pre_tool(handler, tool_name: str, tool_input: dict, tool_rules_result: dict | None = None):
    """Simulate the direction gate logic from runner.py on_pre_tool_use.

    Args:
        handler: handler with _direction_gate_dirs, _direction_gate_tools, state
        tool_name: e.g. "Write", "Edit", "Bash"
        tool_input: e.g. {"file_path": "/home/zhenc/kernel_lab_kb/..."}
        tool_rules_result: what _check_tool_rules would return (None, {}, or constraint dict)

    Returns:
        result dict (deny or None/constraint)
    """
    result = tool_rules_result

    # Direction gate — exact copy of runner.py logic (after fix)
    tool_denied = (
        isinstance(result, dict)
        and result.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
    )
    if not tool_denied:
        needs_direction = False
        gate_dirs = getattr(handler, '_direction_gate_dirs', None) or []
        gate_tools = getattr(handler, '_direction_gate_tools', [])

        if tool_name in gate_tools:
            if tool_name in ("Write", "Edit"):
                path = tool_input.get("file_path", "")
                if any(path.startswith(d) for d in gate_dirs):
                    needs_direction = True
            elif tool_name == "Bash":
                cmd = tool_input.get("command", "")
                if any(w in cmd for w in [" > ", " >> ", "tee ", "cp ", "mv ", "sed -i"]):
                    if any(d in cmd for d in gate_dirs):
                        needs_direction = True

        if needs_direction and hasattr(handler, 'state') and not getattr(handler.state, 'current_direction', None):
            gate_msg = getattr(handler, '_direction_gate_message', "You must set_direction first.")
            result = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": gate_msg,
                }
            }

    return result


# ── Tests ──

def test_write_blocked_without_direction():
    """Write to watched dir without direction → deny."""
    handler = FakeHandler(
        gate_dirs=["/home/zhenc/kernel_lab", "/home/zhenc/kernel_lab_kb"],
        gate_tools=["Write", "Edit", "Bash"],
        direction=None,
    )
    result = _simulate_pre_tool(
        handler, "Write",
        {"file_path": "/home/zhenc/kernel_lab_kb/runs/test/gen/sm90/matmul/cuda/cuda.cu"},
    )
    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_write_allowed_with_direction():
    """Write to watched dir with active direction → allow."""
    handler = FakeHandler(
        gate_dirs=["/home/zhenc/kernel_lab", "/home/zhenc/kernel_lab_kb"],
        gate_tools=["Write", "Edit", "Bash"],
        direction={"name": "warp-specialization"},
    )
    result = _simulate_pre_tool(
        handler, "Write",
        {"file_path": "/home/zhenc/kernel_lab_kb/runs/test/gen/sm90/matmul/cuda/cuda.cu"},
    )
    # Should NOT deny — direction is set
    if result and isinstance(result, dict):
        assert result.get("hookSpecificOutput", {}).get("permissionDecision") != "deny"


def test_write_outside_watched_dir_allowed():
    """Write to unrelated path → allow regardless of direction."""
    handler = FakeHandler(
        gate_dirs=["/home/zhenc/kernel_lab_kb"],
        gate_tools=["Write", "Edit", "Bash"],
        direction=None,
    )
    result = _simulate_pre_tool(
        handler, "Write",
        {"file_path": "/tmp/test.cu"},
    )
    # /tmp is not in watched dirs — should not deny
    if result and isinstance(result, dict):
        assert result.get("hookSpecificOutput", {}).get("permissionDecision") != "deny"


def test_gate_runs_even_with_constraint_result():
    """Direction gate must run even when tool_rules returned a constraint (non-deny).

    This was the bug: tool_rules returns {"hookSpecificOutput": {"additionalContext": "..."}}
    for Write with a constraint, which is truthy, and the old code skipped the gate.
    """
    handler = FakeHandler(
        gate_dirs=["/home/zhenc/kernel_lab_kb"],
        gate_tools=["Write", "Edit", "Bash"],
        direction=None,
    )
    # Simulate tool_rules returning a constraint (non-deny)
    constraint_result = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": "Constraint: 允许写入 ~/kernel_lab_kb/runs/<run_tag>/",
        }
    }
    result = _simulate_pre_tool(
        handler, "Write",
        {"file_path": "/home/zhenc/kernel_lab_kb/runs/test/gen/sm90/matmul/cuda/cuda.cu"},
        tool_rules_result=constraint_result,
    )
    # Gate should override the constraint result with a deny
    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_gate_skipped_when_tool_rules_denied():
    """If tool_rules already denied, direction gate should not override."""
    handler = FakeHandler(
        gate_dirs=["/home/zhenc/kernel_lab_kb"],
        gate_tools=["Write", "Edit", "Bash"],
        direction=None,
    )
    deny_result = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "Tool 'Write' is not allowed.",
        }
    }
    result = _simulate_pre_tool(
        handler, "Write",
        {"file_path": "/home/zhenc/kernel_lab_kb/runs/test/gen/sm90/matmul/cuda/cuda.cu"},
        tool_rules_result=deny_result,
    )
    # Should keep the original deny, not override
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "not allowed" in result["hookSpecificOutput"]["permissionDecisionReason"]


def test_bash_write_blocked_without_direction():
    """Bash with redirect to watched dir without direction → deny."""
    handler = FakeHandler(
        gate_dirs=["/home/zhenc/kernel_lab_kb"],
        gate_tools=["Write", "Edit", "Bash"],
        direction=None,
    )
    result = _simulate_pre_tool(
        handler, "Bash",
        {"command": "cat file.cu > /home/zhenc/kernel_lab_kb/runs/test/gen/cuda.cu"},
    )
    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_bash_read_allowed_without_direction():
    """Bash read-only command → allow even without direction."""
    handler = FakeHandler(
        gate_dirs=["/home/zhenc/kernel_lab_kb"],
        gate_tools=["Write", "Edit", "Bash"],
        direction=None,
    )
    result = _simulate_pre_tool(
        handler, "Bash",
        {"command": "ls /home/zhenc/kernel_lab_kb/runs/test/"},
    )
    # ls is not a write command — should not deny
    if result and isinstance(result, dict):
        assert result.get("hookSpecificOutput", {}).get("permissionDecision") != "deny"


def test_edit_blocked_without_direction():
    """Edit to watched dir without direction → deny."""
    handler = FakeHandler(
        gate_dirs=["/home/zhenc/kernel_lab_kb"],
        gate_tools=["Write", "Edit", "Bash"],
        direction=None,
    )
    result = _simulate_pre_tool(
        handler, "Edit",
        {"file_path": "/home/zhenc/kernel_lab_kb/runs/test/gen/sm90/matmul/cuda/cuda.cu"},
    )
    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
