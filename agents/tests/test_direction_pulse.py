"""Tests for the direction pulse post-tool hook.

Verifies that Steward alignment reviews fire correctly based on
direction state, tool type, watched dirs, and cooldown.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from agents.config import DirectionConfig, PulseTrigger


@dataclass
class FakeState:
    current_direction: dict | None = None
    mode: str = "building"
    error_count: int = 0
    current_action: str = ""


class FakeToolResultEvent:
    def __init__(self, tool_name: str, is_error: bool = False):
        self.tool_name = tool_name
        self.tool_use_id = "test"
        self.result_summary = ""
        self.is_error = is_error


def _make_direction_config(
    file_write_tools=None,
    command_match_tool="Bash",
    watched_dirs=None,
    triggers=None,
):
    return DirectionConfig(
        gate_tools=["Write", "Edit", "Bash"],
        gate_watched_dirs=watched_dirs or ["~/kernel_lab_kb"],
        pulse_file_write_tools=file_write_tools or ["Write", "Edit"],
        pulse_command_match_tool=command_match_tool,
        pulse_watched_dirs=watched_dirs or ["~/kernel_lab_kb"],
        pulse_triggers=triggers or {
            "compile": PulseTrigger(match="exec.action=compile", cooldown=60),
            "trial": PulseTrigger(match="exec.action=trial", cooldown=60),
        },
    )


def _detect_trigger(config: DirectionConfig, pulse_dirs: list[str],
                     tool_name: str, tool_input: dict) -> str | None:
    """Simulate Workshop._detect_trigger logic."""
    if tool_name in config.pulse_file_write_tools:
        path = tool_input.get("file_path", "")
        if any(path.startswith(d) for d in pulse_dirs):
            return "file_write"
    if tool_name == config.pulse_command_match_tool:
        cmd = tool_input.get("command", "")
        for name, trigger in config.pulse_triggers.items():
            if trigger.match and trigger.match in cmd:
                return name
    return None


def _should_pulse(state, config, pulse_dirs, tool_name, tool_input,
                   last_fired=None) -> tuple[bool, str | None]:
    """Simulate the full post-tool pulse decision from Workshop.on_tool_result.

    Returns (should_fire, trigger_name).
    """
    # No direction → no pulse
    if not state.current_direction:
        return False, None

    trigger = _detect_trigger(config, pulse_dirs, tool_name, tool_input)
    if not trigger:
        return False, None

    # Cooldown
    now = datetime.now()
    trigger_config = config.pulse_triggers.get(trigger)
    cooldown = trigger_config.cooldown if trigger_config else 60
    if last_fired and (now - last_fired).total_seconds() < cooldown:
        return False, trigger

    return True, trigger


# ── Tests ──

def test_no_pulse_without_direction():
    """No direction set → pulse should not fire."""
    state = FakeState(current_direction=None)
    config = _make_direction_config()
    pulse_dirs = config.pulse_dirs_resolved()

    should_fire, trigger = _should_pulse(
        state, config, pulse_dirs,
        "Write", {"file_path": os.path.expanduser("~/kernel_lab_kb/runs/test/gen/cuda.cu")},
    )
    assert not should_fire


def test_pulse_fires_with_direction_and_write():
    """Direction set + Write to watched dir → pulse fires."""
    state = FakeState(current_direction={"name": "warp-spec"})
    config = _make_direction_config()
    pulse_dirs = config.pulse_dirs_resolved()

    should_fire, trigger = _should_pulse(
        state, config, pulse_dirs,
        "Write", {"file_path": os.path.expanduser("~/kernel_lab_kb/runs/test/gen/cuda.cu")},
    )
    assert should_fire
    assert trigger == "file_write"


def test_no_pulse_for_unwatched_tool():
    """Direction set but tool not in file_write_tools → no pulse."""
    state = FakeState(current_direction={"name": "warp-spec"})
    config = _make_direction_config()
    pulse_dirs = config.pulse_dirs_resolved()

    should_fire, trigger = _should_pulse(
        state, config, pulse_dirs,
        "Read", {"file_path": os.path.expanduser("~/kernel_lab_kb/runs/test/gen/cuda.cu")},
    )
    assert not should_fire


def test_no_pulse_for_write_outside_watched_dir():
    """Direction set + Write to unrelated dir → no pulse."""
    state = FakeState(current_direction={"name": "warp-spec"})
    config = _make_direction_config()
    pulse_dirs = config.pulse_dirs_resolved()

    should_fire, trigger = _should_pulse(
        state, config, pulse_dirs,
        "Write", {"file_path": "/tmp/scratch.cu"},
    )
    assert not should_fire


def test_pulse_fires_for_compile_command():
    """Direction set + Bash with compile match → pulse fires."""
    state = FakeState(current_direction={"name": "warp-spec"})
    config = _make_direction_config()
    pulse_dirs = config.pulse_dirs_resolved()

    should_fire, trigger = _should_pulse(
        state, config, pulse_dirs,
        "Bash", {"command": ".venv/bin/python -m cuda_exec.exec_cli exec.action=compile ..."},
    )
    assert should_fire
    assert trigger == "compile"


def test_pulse_fires_for_trial_command():
    """Direction set + Bash with trial match → pulse fires."""
    state = FakeState(current_direction={"name": "warp-spec"})
    config = _make_direction_config()
    pulse_dirs = config.pulse_dirs_resolved()

    should_fire, trigger = _should_pulse(
        state, config, pulse_dirs,
        "Bash", {"command": "exec.action=trial exec.kernel=matmul"},
    )
    assert should_fire
    assert trigger == "trial"


def test_cooldown_blocks_repeat_pulse():
    """Pulse within cooldown window → should not fire."""
    state = FakeState(current_direction={"name": "warp-spec"})
    config = _make_direction_config()
    pulse_dirs = config.pulse_dirs_resolved()

    # Last fired 10 seconds ago, cooldown is 60s
    last_fired = datetime.now() - timedelta(seconds=10)
    should_fire, trigger = _should_pulse(
        state, config, pulse_dirs,
        "Write", {"file_path": os.path.expanduser("~/kernel_lab_kb/runs/test/gen/cuda.cu")},
        last_fired=last_fired,
    )
    assert not should_fire
    assert trigger == "file_write"  # trigger detected but cooldown blocked


def test_cooldown_expired_allows_pulse():
    """Pulse after cooldown expired → should fire."""
    state = FakeState(current_direction={"name": "warp-spec"})
    config = _make_direction_config()
    pulse_dirs = config.pulse_dirs_resolved()

    # Last fired 120 seconds ago, cooldown is 60s
    last_fired = datetime.now() - timedelta(seconds=120)
    should_fire, trigger = _should_pulse(
        state, config, pulse_dirs,
        "Write", {"file_path": os.path.expanduser("~/kernel_lab_kb/runs/test/gen/cuda.cu")},
        last_fired=last_fired,
    )
    assert should_fire


def test_edit_triggers_pulse():
    """Edit to watched dir → pulse fires (Edit is in file_write_tools)."""
    state = FakeState(current_direction={"name": "warp-spec"})
    config = _make_direction_config()
    pulse_dirs = config.pulse_dirs_resolved()

    should_fire, trigger = _should_pulse(
        state, config, pulse_dirs,
        "Edit", {"file_path": os.path.expanduser("~/kernel_lab_kb/runs/test/gen/cuda.cu")},
    )
    assert should_fire
    assert trigger == "file_write"


def test_bash_without_match_no_pulse():
    """Bash command without compile/trial pattern → no pulse."""
    state = FakeState(current_direction={"name": "warp-spec"})
    config = _make_direction_config()
    pulse_dirs = config.pulse_dirs_resolved()

    should_fire, trigger = _should_pulse(
        state, config, pulse_dirs,
        "Bash", {"command": "ls ~/kernel_lab_kb/runs/test/"},
    )
    assert not should_fire
