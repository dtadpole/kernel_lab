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


def test_all_scenarios_render_with_jinja2():
    """All scenario templates render correctly with wave_context + scenario vars.

    Verifies that Jinja2 templates produce output containing the expected
    wave context values and scenario-specific values.
    """
    from agents.response_router import ResponseRouter

    router = ResponseRouter(prompts_dir=Path("conf/agent/response_prompts"))

    # Simulate wave_context (what Workshop._get_steward_context() returns)
    wave_context = {
        "mode": "building",
        "direction_json": '{"name": "warp-specialization"}',
        "direction_path": "/tmp/directions/001_warp-specialization.json",
        "transcript_path": "/tmp/transcript.md",
        "events_path": "/tmp/events.jsonl",
        "recent_events": "compile succeeded",
    }

    # Scenario-specific variables (matching Jinja2 namespace)
    scenario_vars = {
        "ask_question": {"ask_question": {"question": "How should I proceed?"}},
        "permission": {"permission": {"tool_name": "Write", "tool_input": "{}"}},
        "session_end": {"session_end": {
            "result_text": "done", "stop_reason": "end_turn",
            "elapsed_time": "30m", "total_tool_calls": "50",
            "error_count": "2",
        }},
        "progress_check": {"progress_check": {"elapsed_time": "15m"}},
        "set_direction": {"set_direction": {
            "proposed_direction_json": '{"name": "new-approach"}',
        }},
        "direction_pulse": {"direction_pulse": {"trigger_type": "compile"}},
        "start_exploring": {"start_exploring": {"reason": "direction exhausted"}},
    }

    for scenario in router.scenarios:
        variables = {"wave": wave_context, **scenario_vars.get(scenario, {})}
        rendered = router.render_user_message(scenario, variables)

        assert "building" in rendered, \
            f"{scenario}: wave.mode 'building' not rendered"
        assert "warp-specialization" in rendered, \
            f"{scenario}: wave.direction_json not rendered"
        assert "/tmp/transcript.md" in rendered, \
            f"{scenario}: wave.transcript_path not rendered"


def test_jinja2_missing_variable_raises():
    """Missing required template variable raises UndefinedError."""
    import jinja2
    from agents.response_router import ResponseRouter

    router = ResponseRouter(prompts_dir=Path("conf/agent/response_prompts"))

    # Pass wave but omit scenario-specific vars
    try:
        router.render_user_message("ask_question", {
            "wave": {"mode": "exploring", "direction_json": "None",
                     "direction_path": "", "transcript_path": "",
                     "events_path": "", "recent_events": ""},
            # Missing: ask_question.question
        })
        assert False, "Should have raised UndefinedError"
    except jinja2.UndefinedError:
        pass  # Expected


def test_steward_method_signatures():
    """All Steward methods take wave_context: dict as first param, no **kwargs."""
    import inspect
    from agents.steward import Steward

    expected = {
        "answer_question":        ["wave_context", "question"],
        "check_permission":       ["wave_context", "tool_name", "tool_input"],
        "review_session_end":     ["wave_context", "result_text", "stop_reason",
                                   "elapsed_time", "total_tool_calls", "error_count"],
        "check_progress":         ["wave_context", "elapsed_time"],
        "review_direction":       ["wave_context", "proposed_direction"],
        "direction_pulse":        ["wave_context", "trigger_type"],
        "review_start_exploring": ["wave_context", "reason"],
    }

    for method_name, expected_params in expected.items():
        method = getattr(Steward, method_name)
        sig = inspect.signature(method)
        params = [p for p in sig.parameters if p != "self"]

        # Check params match expected
        assert params == expected_params, \
            f"{method_name}: expected {expected_params}, got {params}"

        # No **kwargs
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        assert not has_var_keyword, f"{method_name} should not have **kwargs"


def test_md_files_have_wave_context_variables():
    """Each scenario MD file contains the standard {{ wave.* }} variables."""
    prompts_dir = Path("conf/agent/response_prompts")
    wave_vars = ["wave.mode", "wave.direction_json", "wave.direction_path",
                 "wave.transcript_path", "wave.events_path", "wave.recent_events"]

    for md_file in sorted(prompts_dir.glob("*.md")):
        content = md_file.read_text()
        for var in wave_vars:
            assert f"{{{{ {var} }}}}" in content, \
                f"{md_file.name}: missing {{{{ {var} }}}}"


def test_md_files_have_scenario_variables():
    """Each scenario MD file contains its scenario-specific variables."""
    prompts_dir = Path("conf/agent/response_prompts")

    expected_vars = {
        "ask_question.md": ["ask_question.question"],
        "permission.md": ["permission.tool_name", "permission.tool_input"],
        "session_end.md": ["session_end.result_text", "session_end.stop_reason",
                          "session_end.elapsed_time", "session_end.total_tool_calls",
                          "session_end.error_count"],
        "progress_check.md": ["progress_check.elapsed_time"],
        "set_direction.md": ["set_direction.proposed_direction_json"],
        "direction_pulse.md": ["direction_pulse.trigger_type"],
        "start_exploring.md": ["start_exploring.reason"],
    }

    for filename, vars_list in expected_vars.items():
        content = (prompts_dir / filename).read_text()
        for var in vars_list:
            assert f"{{{{ {var} }}}}" in content, \
                f"{filename}: missing {{{{ {var} }}}}"


def test_system_prompt_is_steward_base_only():
    """System prompt should be steward.md only — no scenario content appended."""
    from agents.response_router import ResponseRouter

    router = ResponseRouter(prompts_dir=Path("conf/agent/response_prompts"))
    base = Path("conf/agent/prompts/steward.md").read_text().strip()

    for name, scenario in router.scenarios.items():
        assert scenario.system_prompt == base, \
            f"{name}: system_prompt differs from steward.md base"


def test_no_old_context_header_pattern():
    """No MD file should use old {variable} syntax (single braces)."""
    import re
    prompts_dir = Path("conf/agent/response_prompts")

    for md_file in sorted(prompts_dir.glob("*.md")):
        content = md_file.read_text()
        # Find {word} patterns that aren't inside {{ }} (old regex syntax)
        # Exclude markdown code blocks
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            if line.strip().startswith("```") or line.strip().startswith("#"):
                continue
            # Match single-brace {var} but not {{ var }}
            matches = re.findall(r'(?<!\{)\{(\w+)\}(?!\})', line)
            assert not matches, \
                f"{md_file.name}:{i}: old {{var}} syntax found: {matches}"


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
