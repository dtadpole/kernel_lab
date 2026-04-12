"""Tests for the direction system — state machine, direction IO, config parsing."""

import json
import os
import tempfile
from pathlib import Path

from agents.config import SystemConfig, DirectionConfig, PulseTrigger
from agents.direction import write_direction, read_active_direction, next_seq
from agents.workshop import MODE_EXPLORING, MODE_BUILDING


def test_direction_write_read_roundtrip():
    """Write a direction, read it back — data must match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir) / "directions"
        direction = {
            "name": "warp-specialization",
            "description": "Split warps into producer/consumer",
            "opportunity": "tensor core 3% → 60%",
            "evidence": "NCU shows 3% util",
            "ideas": ["2-WG split", "3-stage pipeline"],
        }
        path = write_direction(d, 1, direction)
        assert path.exists()
        assert "001_warp-specialization.json" in path.name

        loaded = read_active_direction(d)
        assert loaded is not None
        assert loaded["name"] == "warp-specialization"
        assert loaded["ideas"] == ["2-WG split", "3-stage pipeline"]


def test_direction_sequence():
    """Sequence numbers increment correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir) / "directions"
        assert next_seq(d) == 1

        write_direction(d, 1, {"name": "a"})
        assert next_seq(d) == 2

        write_direction(d, 2, {"name": "b"})
        assert next_seq(d) == 3


def test_read_active_returns_latest():
    """read_active_direction returns the last file (highest seq)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir) / "directions"
        write_direction(d, 1, {"name": "first"})
        write_direction(d, 2, {"name": "second"})

        active = read_active_direction(d)
        assert active["name"] == "second"


def test_read_active_empty():
    """read_active_direction returns None for empty or missing dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir) / "directions"
        assert read_active_direction(d) is None

        d.mkdir()
        assert read_active_direction(d) is None


def test_config_direction_parsing():
    """Direction config parsed correctly from agents.yaml."""
    config = SystemConfig.from_yaml("conf/agent/agents.yaml")
    dc = config.direction

    # Gate
    assert "Write" in dc.gate_tools
    assert "Edit" in dc.gate_tools
    assert "Bash" in dc.gate_tools
    assert len(dc.gate_watched_dirs) >= 2

    # Pulse
    assert "Write" in dc.pulse_file_write_tools
    assert "Edit" in dc.pulse_file_write_tools
    assert dc.pulse_command_match_tool == "Bash"
    assert "compile" in dc.pulse_triggers
    assert "trial" in dc.pulse_triggers
    assert "profile" in dc.pulse_triggers
    assert "file_write" in dc.pulse_triggers

    # Cooldowns
    for name, trigger in dc.pulse_triggers.items():
        assert trigger.cooldown > 0


def test_config_gate_dirs_resolved():
    """Gate dirs expand ~ correctly."""
    config = SystemConfig.from_yaml("conf/agent/agents.yaml")
    dirs = config.direction.gate_dirs_resolved()
    for d in dirs:
        assert "~" not in d
        assert d.startswith("/")


def test_mode_constants():
    """Mode constants are defined and distinct."""
    assert MODE_EXPLORING == "exploring"
    assert MODE_BUILDING == "building"
    assert MODE_EXPLORING != MODE_BUILDING


def test_solver_has_direction_tools():
    """Solver agent config includes set_direction and start_exploring."""
    config = SystemConfig.from_yaml("conf/agent/agents.yaml")
    solver = config.get_agent("solver")
    assert "set_direction" in solver.custom_tools
    assert "start_exploring" in solver.custom_tools


def test_response_prompts_exist():
    """All expected response prompt files exist."""
    prompts_dir = Path("conf/agent/response_prompts")
    expected = [
        "ask_question.md",
        "permission.md",
        "session_end.md",
        "progress_check.md",
        "set_direction.md",
        "start_exploring.md",
        "direction_pulse.md",
    ]
    for name in expected:
        path = prompts_dir / name
        assert path.exists(), f"Missing: {path}"
        content = path.read_text()
        assert len(content) > 50, f"Too short: {path}"


def test_deleted_prompts_gone():
    """Removed prompts should not exist."""
    prompts_dir = Path("conf/agent/response_prompts")
    deleted = ["stuck.md", "time_limit.md"]
    for name in deleted:
        assert not (prompts_dir / name).exists(), f"Should be deleted: {name}"

    workshop_dir = Path("conf/agent/prompts")
    assert not (workshop_dir / "workshop_continue.md").exists()
    assert not (workshop_dir / "workshop_explore.md").exists()
    assert not (workshop_dir / "workshop_brainstorm.md").exists()


def test_direction_gate_prompt_exists():
    """Direction gate prompt file exists and is readable."""
    path = Path("conf/agent/prompts/direction_gate.md")
    assert path.exists()
    content = path.read_text()
    assert "set_direction" in content.lower() or "direction" in content.lower()


def test_steward_action_levels():
    """Key actions are mapped to correct intervention levels."""
    from agents.steward import _ACTION_LEVELS

    assert _ACTION_LEVELS["APPROVED"] == 1
    assert _ACTION_LEVELS["ON_TRACK"] == 1
    assert _ACTION_LEVELS["CONTINUE"] == 2
    assert _ACTION_LEVELS["REDIRECT"] == 2
    assert _ACTION_LEVELS["EXPLORE"] == 2
    assert _ACTION_LEVELS["ABORT"] == 3

    # Removed actions should not be present
    assert "REVISE" not in _ACTION_LEVELS
    assert "REJECTED" not in _ACTION_LEVELS
    assert "BRAINSTORM" not in _ACTION_LEVELS
    assert "WRAP_UP" not in _ACTION_LEVELS


def test_context_header_has_direction():
    """Context header includes direction_json and direction_path."""
    from agents.response_router import _CONTEXT_HEADER

    assert "{direction_json}" in _CONTEXT_HEADER
    assert "{direction_path}" in _CONTEXT_HEADER
    assert "{mode}" in _CONTEXT_HEADER
    assert "{recent_events}" in _CONTEXT_HEADER
    assert "{transcript_path}" in _CONTEXT_HEADER
    assert "{events_path}" in _CONTEXT_HEADER


def test_context_header_no_stale_fields():
    """Context header should not have removed fields."""
    from agents.response_router import _CONTEXT_HEADER

    assert "{direction_name}" not in _CONTEXT_HEADER
    assert "{direction_description}" not in _CONTEXT_HEADER
    assert "{direction_opportunity}" not in _CONTEXT_HEADER


def test_all_scenarios_registered():
    """All expected scenarios are in SCENARIO_MAX_TURNS."""
    from agents.response_router import SCENARIO_MAX_TURNS

    expected = [
        "ask_question", "permission", "session_end",
        "progress_check", "set_direction", "start_exploring",
        "direction_pulse",
    ]
    for name in expected:
        assert name in SCENARIO_MAX_TURNS, f"Missing scenario: {name}"

    # Removed scenarios
    assert "stuck" not in SCENARIO_MAX_TURNS
    assert "time_limit" not in SCENARIO_MAX_TURNS


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
