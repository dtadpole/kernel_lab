"""Tests for Phase 1A: Agent Runner and supporting modules."""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import anyio
import pytest

from agents.config import AgentConfig, MonitorConfig, StorageConfig, SystemConfig, ToolRule
from agents.events import (
    DefaultHandler,
    MonitorAlert,
    PermissionEvent,
    StopEvent,
    ToolCallEvent,
    ToolResultEvent,
    TextOutputEvent,
)
from agents.monitor import AgentMonitor
from agents.session_log import SessionLog
from agents.storage import SessionStorage


# ── Config tests ──

@pytest.mark.quick
def test_config_from_yaml():
    config = SystemConfig.from_yaml("conf/agent/agents.yaml")
    assert "solver" in config.agents
    assert "benchmarker" in config.agents
    assert "rigger" in config.agents
    solver = config.get_agent("solver")
    assert "Read" in solver.builtin_tools
    assert "ask_supervisor" in solver.custom_tools
    assert solver.permission_mode == "acceptEdits"
    assert solver.system_prompt  # loaded from file
    print(f"  Loaded {len(config.agents)} agents: {list(config.agents.keys())}")
    print(f"  Solver prompt: {solver.system_prompt[:60]}...")


@pytest.mark.quick
def test_config_defaults_merge():
    config = SystemConfig.from_yaml("conf/agent/agents.yaml")
    # rigger doesn't set model explicitly, should get default
    rigger = config.get_agent("rigger")
    assert rigger.model == "claude-sonnet-4-6"


@pytest.mark.quick
def test_config_tool_rules():
    config = SystemConfig.from_yaml("conf/agent/agents.yaml")
    benchmarker = config.get_agent("benchmarker")
    edit_rule = next((r for r in benchmarker.tool_rules if r.tool == "Edit"), None)
    assert edit_rule is not None
    assert edit_rule.allow is False


@pytest.mark.quick
def test_config_missing_agent():
    config = SystemConfig.from_yaml("conf/agent/agents.yaml")
    with pytest.raises(KeyError, match="nonexistent"):
        config.get_agent("nonexistent")


# ── Event tests ──

@pytest.mark.quick
def test_event_to_dict():
    event = ToolCallEvent(tool_name="Bash", tool_input={"command": "ls"}, tool_use_id="t1")
    d = event.to_dict()
    assert d["type"] == "ToolCallEvent"
    assert d["tool_name"] == "Bash"
    assert "ts" in d
    assert "id" in d


# ── Storage tests ──

@pytest.mark.quick
def test_storage_creates_session_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        sc = StorageConfig(kb_root=tmpdir, run_tag="test_run")
        storage = SessionStorage(sc, "solver", "test_task")
        assert storage.session_dir.exists()
        assert storage.session_dir.parent.name == "test_task"
        print(f"  Session dir: {storage.session_dir}")


@pytest.mark.quick
def test_storage_append_event():
    with tempfile.TemporaryDirectory() as tmpdir:
        sc = StorageConfig(kb_root=tmpdir, run_tag="test_run")
        storage = SessionStorage(sc, "solver", "test_task")
        storage.append_event({"type": "test", "ts": "now"})
        storage.append_event({"type": "test2", "ts": "now2"})

        lines = storage.events_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["type"] == "test"


@pytest.mark.quick
def test_storage_transcript():
    with tempfile.TemporaryDirectory() as tmpdir:
        sc = StorageConfig(kb_root=tmpdir, run_tag="test_run")
        storage = SessionStorage(sc, "solver", "matmul_opt")
        storage.init_transcript("solver", "Optimize matmul kernel")

        content = storage.transcript_path.read_text()
        assert "slug:" in content
        assert "type: journal" in content
        assert "Optimize matmul kernel" in content


# ── SessionLog tests ──

@pytest.mark.quick
def test_session_log_append_and_query():
    log = SessionLog()
    log.append(ToolCallEvent(tool_name="Read", tool_input={"file": "a.cu"}, tool_use_id="t1"))
    log.append(ToolResultEvent(tool_name="Read", tool_use_id="t1", result_summary="ok"))
    log.append(ToolCallEvent(tool_name="Edit", tool_input={"file": "a.cu"}, tool_use_id="t2"))

    assert len(log.events) == 3
    assert log.tool_call_counts() == {"Read": 1, "Edit": 1}
    assert log.recent_tool_sequence() == ["Read", "Edit"]


@pytest.mark.quick
def test_session_log_summary():
    log = SessionLog()
    log.append(ToolCallEvent(tool_name="Read", tool_input={"file": "a.cu"}, tool_use_id="t1"))
    log.append(TextOutputEvent(text="Analyzing the kernel..."))
    log.append(ToolCallEvent(tool_name="Edit", tool_input={"file": "a.cu"}, tool_use_id="t2"))

    summary = log.to_summary()
    assert "Read" in summary
    assert "Edit" in summary
    assert "Total events: 3" in summary


@pytest.mark.quick
def test_session_log_with_storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        sc = StorageConfig(kb_root=tmpdir, run_tag="test_run")
        storage = SessionStorage(sc, "solver", "test")
        storage.init_transcript("solver", "test task")

        log = SessionLog(storage)
        log.append(ToolCallEvent(tool_name="Bash", tool_input={"cmd": "nvcc"}, tool_use_id="t1"))
        log.append(StopEvent(reason="end_turn", result_text="done"))

        # Check events.jsonl
        lines = storage.events_path.read_text().strip().split("\n")
        assert len(lines) == 2

        # Check transcript
        transcript = storage.transcript_path.read_text()
        assert "Tool: Bash" in transcript
        assert "Session ended" in transcript


# ── Monitor tests ──

@pytest.mark.quick
def test_monitor_idle_detection():
    log = SessionLog()
    # Simulate an old event
    old_event = ToolCallEvent(tool_name="Read", tool_input={}, tool_use_id="t1")
    old_event.timestamp = datetime.now() - timedelta(seconds=400)
    log.events.append(old_event)
    log._start_time = old_event.timestamp

    handler = DefaultHandler()
    monitor = AgentMonitor(log, handler, config=MonitorConfig(idle_timeout=300))
    alert = monitor._check_health()
    assert alert is not None
    assert alert.alert_type == "idle_timeout"


@pytest.mark.quick
def test_monitor_loop_detection():
    log = SessionLog()
    log._start_time = datetime.now()
    for i in range(6):
        log.events.append(ToolCallEvent(tool_name="Bash", tool_input={}, tool_use_id=f"t{i}"))

    handler = DefaultHandler()
    monitor = AgentMonitor(log, handler, config=MonitorConfig(loop_threshold=5))
    alert = monitor._check_health()
    assert alert is not None
    assert alert.alert_type == "loop_detected"


@pytest.mark.quick
def test_monitor_no_alert_when_healthy():
    log = SessionLog()
    log.append(ToolCallEvent(tool_name="Read", tool_input={}, tool_use_id="t1"))
    log.append(ToolCallEvent(tool_name="Edit", tool_input={}, tool_use_id="t2"))

    handler = DefaultHandler()
    monitor = AgentMonitor(log, handler, config=MonitorConfig())
    alert = monitor._check_health()
    assert alert is None


# ── Integration test: full runner ──

@pytest.mark.integration
@pytest.mark.timeout(120)
def test_runner_simple():
    """End-to-end: run a simple prompt through AgentRunner."""
    from agents.runner import AgentRunner, RunResult

    agent_config = AgentConfig(
        name="test_solver",
        builtin_tools=["Read"],
        custom_tools=[],
        permission_mode="acceptEdits",
        max_turns=3,
        model="claude-sonnet-4-6",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        storage_config = StorageConfig(kb_root=tmpdir, run_tag="test_run")

        runner = AgentRunner(
            agent_config=agent_config,
            storage_config=storage_config,
            monitor_config=MonitorConfig(check_interval=5),
        )

        result = anyio.run(runner.run, "Respond with exactly: RUNNER_OK", "test_run")

        assert isinstance(result, RunResult)
        assert result.result_text  # got some response
        assert result.session_id  # got a session id
        assert result.log.events  # events were recorded

        # Check journal files exist
        journal_dir = Path(tmpdir) / "runs" / "test_run" / "journal" / "test_solver" / "test_run"
        sessions = list(journal_dir.iterdir())
        assert len(sessions) == 1
        session_dir = sessions[0]
        assert (session_dir / "meta.json").exists()
        assert (session_dir / "events.jsonl").exists()
        assert (session_dir / "transcript.md").exists()

        print(f"  Result: {result.result_text[:80]}")
        print(f"  Session: {result.session_id}")
        print(f"  Events: {len(result.log.events)}")
        print(f"  Journal: {session_dir}")


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_runner_with_hooks():
    """Verify hooks fire and events are captured."""
    from agents.runner import AgentRunner, RunResult

    captured_events = []

    class CapturingHandler(DefaultHandler):
        async def on_tool_call(self, event):
            captured_events.append(("tool_call", event.tool_name))

        async def on_tool_result(self, event):
            captured_events.append(("tool_result", event.tool_name))

        async def on_text(self, event):
            captured_events.append(("text", event.text[:50]))

        async def on_stop(self, event):
            captured_events.append(("stop", event.reason))

    agent_config = AgentConfig(
        name="test_hooks",
        builtin_tools=["Read", "Glob"],
        custom_tools=[],
        permission_mode="acceptEdits",
        max_turns=5,
        model="claude-sonnet-4-6",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        storage_config = StorageConfig(kb_root=tmpdir, run_tag="test_run")

        runner = AgentRunner(
            agent_config=agent_config,
            storage_config=storage_config,
            handler=CapturingHandler(),
            monitor_config=MonitorConfig(check_interval=5),
        )

        result = anyio.run(
            runner.run,
            "List the Python files in the agents/ directory using Glob. Then respond with HOOKS_OK.",
            "test_hooks",
        )

        assert result.result_text
        # Should have captured at least a tool call and a stop
        event_types = [e[0] for e in captured_events]
        assert "stop" in event_types
        print(f"  Captured events: {captured_events}")
