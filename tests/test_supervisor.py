"""Tests for Phase 2: Supervisor + ResponseRouter."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import anyio
import pytest

from agents.config import MonitorConfig, StorageConfig, SystemConfig
from agents.events import (
    AskEvent,
    MonitorAlert,
    PermissionEvent,
    StopEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from agents.response_router import ResponseRouter, ResponseVerdict
from agents.session_log import SessionLog
from agents.runner import RunResult
from agents.steward import Steward, StewardResponse, _ACTION_LEVELS
from agents.supervisor import Supervisor, SupervisorState, _slugify


# ── ResponseVerdict parsing ──

@pytest.mark.quick
def test_verdict_parse_accept():
    v = ResponseVerdict.parse("SUCCESS\nWork looks good.")
    assert v.action == "SUCCESS"
    assert v.detail == ""
    assert "looks good" in v.reasoning


@pytest.mark.quick
def test_verdict_parse_reject_with_detail():
    v = ResponseVerdict.parse("ABORT:Missing benchmark results\nNeed to run ik:bench")
    assert v.action == "ABORT"
    assert v.detail == "Missing benchmark results"
    assert "ik:bench" in v.reasoning


@pytest.mark.quick
def test_verdict_parse_inject():
    v = ResponseVerdict.parse("INJECT:Try warp specialization instead")
    assert v.action == "INJECT"
    assert "warp specialization" in v.detail


@pytest.mark.quick
def test_verdict_parse_extend():
    v = ResponseVerdict.parse("EXTEND:30\nSolver is making progress on tile sizing")
    assert v.action == "EXTEND"
    assert v.detail == "30"


@pytest.mark.quick
def test_verdict_parse_fallback():
    v = ResponseVerdict.parse("some unexpected format")
    assert v.action == "SOME_UNEXPECTED_FORMAT"


# ── ResponseRouter ──

@pytest.mark.quick
def test_router_loads_prompts():
    router = ResponseRouter(prompts_dir="conf/agent/response_prompts")
    assert router.has_scenario("ask_question")
    assert router.has_scenario("permission")
    assert router.has_scenario("stuck")
    assert router.has_scenario("session_end")
    assert router.has_scenario("time_limit")
    print(f"  Loaded {len(router.scenarios)} scenarios: {list(router.scenarios.keys())}")


@pytest.mark.quick
def test_router_build_context():
    router = ResponseRouter(prompts_dir="conf/agent/response_prompts")
    ctx = router.build_context("ask_question", {
        "transcript_path": "/tmp/test/transcript.md",
        "question": "Should I use WGMMA or HMMA?",
    })
    assert "transcript.md" in ctx
    assert "WGMMA or HMMA" in ctx


@pytest.mark.quick
def test_router_build_context_missing_vars():
    router = ResponseRouter(prompts_dir="conf/agent/response_prompts")
    ctx = router.build_context("session_end", {
        "transcript_path": "/tmp/test/transcript.md",
        # Missing most variables
    })
    assert "transcript.md" in ctx
    assert "(not available)" in ctx  # missing vars get placeholder


@pytest.mark.quick
def test_router_unknown_scenario():
    router = ResponseRouter(prompts_dir="conf/agent/response_prompts")
    with pytest.raises(KeyError, match="nonexistent"):
        router.build_context("nonexistent", {})


# ── Supervisor state tracking ──

@pytest.mark.quick
def test_supervisor_state_on_tool_call():
    config = SystemConfig.from_yaml("conf/agent/agents.yaml")
    sup = Supervisor(config)
    sup.state = SupervisorState(phase="solving", task="test")

    event = ToolCallEvent(tool_name="Edit", tool_input={"file": "a.cu"}, tool_use_id="t1")
    anyio.run(sup.on_tool_call, event)

    assert sup.state.turns_completed == 1
    assert sup.state.current_action == "Edit"


@pytest.mark.quick
def test_supervisor_state_on_error():
    config = SystemConfig.from_yaml("conf/agent/agents.yaml")
    sup = Supervisor(config)
    sup.state = SupervisorState(phase="solving", task="test")

    event = ToolResultEvent(tool_name="Bash", tool_use_id="t1", result_summary="error", is_error=True)
    anyio.run(sup.on_tool_result, event)

    assert sup.state.error_count == 1


@pytest.mark.quick
def test_supervisor_monitor_hard_limit():
    """on_monitor_alert with hard_limit should always return interrupt."""
    config = SystemConfig.from_yaml("conf/agent/agents.yaml")
    sup = Supervisor(config)
    sup.state = SupervisorState(phase="solving", task="test")

    alert = MonitorAlert(alert_type="hard_limit", details="6 hours exceeded")

    # hard_limit not in router scenarios → falls through to default
    action = anyio.run(sup.on_monitor_alert, alert)
    # hard_limit is not "total_timeout", so it goes to "stuck" scenario
    # but even if stuck scenario says CONTINUE, hard_limit should interrupt
    # Let's verify the current behavior
    assert action in ("continue", "interrupt")


# ── Steward typed methods + intervention levels ──

@pytest.mark.quick
def test_steward_response_levels():
    for action, expected_level in _ACTION_LEVELS.items():
        sr = StewardResponse(action=action, detail="", reasoning="", intervention_level=expected_level)
        if expected_level >= 2:
            assert sr.needs_solver_interrupt
        if expected_level >= 3:
            assert sr.needs_solver_restart
        if expected_level >= 4:
            assert sr.needs_solver_kill


@pytest.mark.quick
def test_steward_response_inline():
    sr = StewardResponse(action="SUCCESS", detail="", reasoning="ok", intervention_level=1)
    assert not sr.needs_solver_interrupt
    assert not sr.needs_solver_restart
    assert not sr.needs_solver_kill


@pytest.mark.quick
def test_steward_response_inject():
    sr = StewardResponse(action="INJECT", detail="try warp spec", reasoning="", intervention_level=2)
    assert sr.needs_solver_interrupt
    assert not sr.needs_solver_restart


@pytest.mark.quick
def test_steward_response_restart():
    sr = StewardResponse(action="ABORT", detail="missing bench", reasoning="", intervention_level=3)
    assert sr.needs_solver_interrupt
    assert sr.needs_solver_restart
    assert not sr.needs_solver_kill


@pytest.mark.quick
def test_steward_response_kill():
    sr = StewardResponse(action="KILL", detail="", reasoning="", intervention_level=4)
    assert sr.needs_solver_kill


# ── Benchmarker parse logic ──

@pytest.mark.quick
def test_parse_bench_improved_true():
    config = SystemConfig.from_yaml("conf/agent/agents.yaml")
    sup = Supervisor(config)
    result = RunResult(result_text="improved: true, new gem created v002_20260405")
    assert sup._parse_bench_improved(result) is True


@pytest.mark.quick
def test_parse_bench_improved_false():
    config = SystemConfig.from_yaml("conf/agent/agents.yaml")
    sup = Supervisor(config)
    result = RunResult(result_text="No configs beat the previous best. improved: false")
    assert sup._parse_bench_improved(result) is False


@pytest.mark.quick
def test_parse_bench_improved_gem_keyword():
    config = SystemConfig.from_yaml("conf/agent/agents.yaml")
    sup = Supervisor(config)
    result = RunResult(result_text="A new gem was created at v003_20260405_120000")
    assert sup._parse_bench_improved(result) is True


# ── Supervisor consecutive stuck tracking ──

@pytest.mark.quick
def test_supervisor_consecutive_stuck():
    config = SystemConfig.from_yaml("conf/agent/agents.yaml")
    sup = Supervisor(config)
    sup.state = SupervisorState(phase="solving", task="test", consecutive_stuck=2)

    # 3rd consecutive stuck should trigger forced guidance
    alert = MonitorAlert(alert_type="idle_timeout", details="15 min idle")
    action = anyio.run(sup.on_monitor_alert, alert)
    # After 3 stuck, Steward is consulted — any action is valid
    assert action in ("continue", "interrupt") or action.startswith("inject:")


@pytest.mark.quick
def test_supervisor_hard_limit_no_steward():
    config = SystemConfig.from_yaml("conf/agent/agents.yaml")
    sup = Supervisor(config)
    sup.state = SupervisorState(phase="solving", task="test")

    alert = MonitorAlert(alert_type="hard_limit", details="6h exceeded")
    action = anyio.run(sup.on_monitor_alert, alert)
    assert action == "interrupt"


# ── Steward loads all 6 scenarios ──

@pytest.mark.quick
def test_steward_has_all_scenarios():
    steward = Steward(prompts_dir="conf/agent/response_prompts")
    for scenario in ["ask_question", "permission", "stuck", "session_end", "time_limit", "progress_check"]:
        assert steward.router.has_scenario(scenario), f"Missing scenario: {scenario}"


# ── Per-agent MonitorConfig presets ──

@pytest.mark.quick
def test_monitor_config_for_solver():
    mc = MonitorConfig.for_solver()
    assert mc.idle_timeout == 1800      # 30 min
    assert mc.total_timeout == 7200     # 2 hours
    assert mc.hard_limit == 43200       # 12 hours


@pytest.mark.quick
def test_monitor_config_for_benchmarker():
    mc = MonitorConfig.for_benchmarker()
    assert mc.total_timeout == 300      # 5 min
    assert mc.hard_limit == 600         # 10 min


@pytest.mark.quick
def test_monitor_config_for_steward():
    mc = MonitorConfig.for_steward()
    assert mc.total_timeout == 600      # 10 min
    assert mc.hard_limit == 900         # 15 min


# ── Steward journal creation ──

@pytest.mark.quick
def test_steward_router_uses_storage_config():
    """ResponseRouter should accept and pass storage_config."""
    from agents.response_router import ResponseRouter
    sc = StorageConfig(kb_root="/tmp/test_kb", run_tag="test_run")
    router = ResponseRouter(
        prompts_dir="conf/agent/response_prompts",
        storage_config=sc,
    )
    assert router.storage_config.kb_root == "/tmp/test_kb"


@pytest.mark.quick
def test_slugify():
    assert _slugify("Optimize matmul kernel for SM90") == "optimize_matmul_kern"
    assert _slugify("Hello, World!  Test") == "hello_world_test"


@pytest.mark.quick
def test_supervisor_get_status():
    config = SystemConfig.from_yaml("conf/agent/agents.yaml")
    sup = Supervisor(config)
    sup.state = SupervisorState(
        phase="solving",
        task="Optimize matmul",
        iteration=1,
        turns_completed=5,
        error_count=1,
        started_at=datetime.now(),
    )
    status = sup.get_status()
    assert status["phase"] == "solving"
    assert status["iteration"] == 1
    assert status["turns"] == 5


# ── Integration tests ──

@pytest.mark.integration
@pytest.mark.timeout(120)
def test_response_router_ask_question():
    """End-to-end: ResponseRouter calls Agent SDK for ask_question."""
    router = ResponseRouter(prompts_dir="conf/agent/response_prompts")

    async def _test():
        verdict_text = await router.respond_raw("ask_question", {
            "task_description": "Optimize matmul kernel for SM90",
            "session_summary": "Solver has read the kernel source and identified memory layout issues.",
            "question": "Should I use WGMMA or HMMA for this SM90 target?",
            "solver_context": "The kernel currently uses HMMA instructions.",
        })
        assert verdict_text  # got a response
        print(f"  Response: {verdict_text[:200]}")
        return verdict_text

    result = anyio.run(_test)
    assert len(result) > 10


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_response_router_session_end():
    """End-to-end: ResponseRouter calls Agent SDK for session_end verdict."""
    router = ResponseRouter(prompts_dir="conf/agent/response_prompts")

    async def _test():
        verdict = await router.respond("session_end", {
            "task_description": "Read the file agents/__init__.py and report its contents",
            "result_text": "The file agents/__init__.py is empty.",
            "stop_reason": "end_turn",
            "elapsed_time": "0:00:15",
            "total_tool_calls": "2",
            "error_count": "0",
            "session_summary": "Tool: Read agents/__init__.py → empty file. Agent reported contents.",
        })
        assert verdict.action in ("SUCCESS", "CONTINUE", "ABORT")
        print(f"  Verdict: {verdict.action}:{verdict.detail}")
        print(f"  Reasoning: {verdict.reasoning[:200]}")
        return verdict

    result = anyio.run(_test)
    assert result.action  # got a structured verdict


@pytest.mark.integration
@pytest.mark.timeout(180)
def test_supervisor_simple_task():
    """End-to-end: Supervisor runs a simple task through the full loop."""
    config = SystemConfig.from_yaml("conf/agent/agents.yaml")

    # Override for test: use tmpdir for journal, short timeouts
    with tempfile.TemporaryDirectory() as tmpdir:
        config.storage = StorageConfig(kb_root=tmpdir, run_tag="test_run")
        config.monitor = MonitorConfig(
            idle_timeout=60, total_timeout=120, hard_limit=180,
            check_interval=10, loop_threshold=5,
        )
        # Limit solver turns for test
        solver = config.get_agent("solver")
        solver.max_turns = 5

        sup = Supervisor(config, max_iterations=1)

        result = anyio.run(sup.run_task, "Read the file agents/__init__.py and tell me if it is empty.")

        print(f"  Success: {result.success}")
        print(f"  Result: {result.result_text[:200]}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Verdicts: {result.verdict_history}")
        print(f"  Elapsed: {result.elapsed_seconds:.1f}s")

        assert result.result_text  # got some output
        assert result.iterations >= 1
        assert result.verdict_history  # at least one verdict was made
