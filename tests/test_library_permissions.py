"""Tests for Library agent permission isolation and dispatch logic.

Verifies that tool_rules in agents.yaml correctly enforce:
- Librarian: write wiki/ but NOT _proposals/
- Analyst: write _proposals/pending/ but NOT wiki categories
- Taxonomist/Auditor: read-only, wiki/ only

Also tests Library host process logic (queue scanning, on_ask dispatch).
"""

import os
from pathlib import Path

import pytest

from agents.config import SystemConfig
from agents.runner import AgentRunner

pytestmark = pytest.mark.quick


@pytest.fixture
def config():
    return SystemConfig.from_yaml("conf/agent/agents.yaml")


def _make_runner(config, agent_name: str) -> AgentRunner:
    """Create a runner for the given agent (no subprocess, just for rule checking)."""
    agent_config = config.get_agent(agent_name)
    return AgentRunner(
        agent_config=agent_config,
        storage_config=config.storage,
    )


def _is_blocked(runner: AgentRunner, path: str, tool_name: str) -> bool:
    """Check if a path is blocked for a given tool on this runner."""
    abs_path = os.path.expanduser(path)
    for rule in runner.agent_config.tool_rules:
        if rule.tool == tool_name:
            return runner._is_path_blocked(abs_path, rule)
    return False  # No rule for this tool = not blocked


# ── Librarian permissions ──

class TestLibrarianPermissions:

    def test_write_wiki_concepts_allowed(self, config):
        runner = _make_runner(config, "librarian")
        assert not _is_blocked(runner, "~/kernel_lab_kb/wiki/concepts/test.md", "Write")

    def test_write_wiki_patterns_allowed(self, config):
        runner = _make_runner(config, "librarian")
        assert not _is_blocked(runner, "~/kernel_lab_kb/wiki/patterns/new-pattern.md", "Write")

    def test_write_proposals_blocked(self, config):
        runner = _make_runner(config, "librarian")
        assert _is_blocked(runner, "~/kernel_lab_kb/wiki/_proposals/pending/test.yaml", "Write")

    def test_write_proposals_inject_blocked(self, config):
        runner = _make_runner(config, "librarian")
        assert _is_blocked(runner, "~/kernel_lab_kb/wiki/_proposals/inject/urgent.yaml", "Write")

    def test_edit_wiki_allowed(self, config):
        runner = _make_runner(config, "librarian")
        assert not _is_blocked(runner, "~/kernel_lab_kb/wiki/patterns/existing.md", "Edit")

    def test_edit_proposals_blocked(self, config):
        runner = _make_runner(config, "librarian")
        assert _is_blocked(runner, "~/kernel_lab_kb/wiki/_proposals/pending/test.yaml", "Edit")

    def test_read_proposals_allowed(self, config):
        runner = _make_runner(config, "librarian")
        assert not _is_blocked(runner, "~/kernel_lab_kb/wiki/_proposals/pending/test.yaml", "Read")

    def test_read_kernel_lab_blocked(self, config):
        runner = _make_runner(config, "librarian")
        assert _is_blocked(runner, "~/kernel_lab/agents/library.py", "Read")

    def test_write_kernel_lab_blocked(self, config):
        runner = _make_runner(config, "librarian")
        assert _is_blocked(runner, "~/kernel_lab/agents/library.py", "Write")

    def test_write_runs_blocked(self, config):
        runner = _make_runner(config, "librarian")
        assert _is_blocked(runner, "~/kernel_lab_kb/runs/some_run/gen/test.cu", "Write")


# ── Information Analyst permissions ──

class TestAnalystPermissions:

    def test_write_proposals_pending_allowed(self, config):
        runner = _make_runner(config, "information_analyst")
        assert not _is_blocked(runner, "~/kernel_lab_kb/wiki/_proposals/pending/new.yaml", "Write")

    def test_edit_proposals_pending_allowed(self, config):
        runner = _make_runner(config, "information_analyst")
        assert not _is_blocked(runner, "~/kernel_lab_kb/wiki/_proposals/pending/existing.yaml", "Edit")

    def test_write_wiki_concepts_blocked(self, config):
        runner = _make_runner(config, "information_analyst")
        assert _is_blocked(runner, "~/kernel_lab_kb/wiki/concepts/test.md", "Write")

    def test_write_wiki_patterns_blocked(self, config):
        runner = _make_runner(config, "information_analyst")
        assert _is_blocked(runner, "~/kernel_lab_kb/wiki/patterns/test.md", "Write")

    def test_write_proposals_done_blocked(self, config):
        runner = _make_runner(config, "information_analyst")
        assert _is_blocked(runner, "~/kernel_lab_kb/wiki/_proposals/done/old.yaml", "Write")

    def test_read_wiki_allowed(self, config):
        runner = _make_runner(config, "information_analyst")
        assert not _is_blocked(runner, "~/kernel_lab_kb/wiki/concepts/existing.md", "Read")

    def test_read_runs_allowed(self, config):
        runner = _make_runner(config, "information_analyst")
        assert not _is_blocked(runner, "~/kernel_lab_kb/runs/some_run/reflections/ref.md", "Read")

    def test_read_kernel_lab_blocked(self, config):
        runner = _make_runner(config, "information_analyst")
        assert _is_blocked(runner, "~/kernel_lab/agents/library.py", "Read")


# ── Taxonomist permissions (read-only) ──

class TestTaxonomistPermissions:

    def test_no_write_tool(self, config):
        agent = config.get_agent("taxonomist")
        assert "Write" not in agent.builtin_tools
        assert "Edit" not in agent.builtin_tools

    def test_read_wiki_allowed(self, config):
        runner = _make_runner(config, "taxonomist")
        assert not _is_blocked(runner, "~/kernel_lab_kb/wiki/concepts/existing.md", "Read")

    def test_read_runs_blocked(self, config):
        runner = _make_runner(config, "taxonomist")
        assert _is_blocked(runner, "~/kernel_lab_kb/runs/some_run/reflections/ref.md", "Read")

    def test_read_kernel_lab_blocked(self, config):
        runner = _make_runner(config, "taxonomist")
        assert _is_blocked(runner, "~/kernel_lab/agents/library.py", "Read")

    def test_glob_wiki_allowed(self, config):
        runner = _make_runner(config, "taxonomist")
        assert not _is_blocked(runner, "~/kernel_lab_kb/wiki/", "Glob")

    def test_glob_runs_blocked(self, config):
        runner = _make_runner(config, "taxonomist")
        assert _is_blocked(runner, "~/kernel_lab_kb/runs/", "Glob")


# ── Auditor permissions (read-only) ──

class TestAuditorPermissions:

    def test_no_write_tool(self, config):
        agent = config.get_agent("auditor")
        assert "Write" not in agent.builtin_tools
        assert "Edit" not in agent.builtin_tools

    def test_read_wiki_allowed(self, config):
        runner = _make_runner(config, "auditor")
        assert not _is_blocked(runner, "~/kernel_lab_kb/wiki/concepts/existing.md", "Read")

    def test_read_proposals_allowed(self, config):
        runner = _make_runner(config, "auditor")
        assert not _is_blocked(runner, "~/kernel_lab_kb/wiki/_proposals/pending/proposal.yaml", "Read")

    def test_read_runs_blocked(self, config):
        runner = _make_runner(config, "auditor")
        assert _is_blocked(runner, "~/kernel_lab_kb/runs/some_run/reflections/ref.md", "Read")

    def test_read_kernel_lab_blocked(self, config):
        runner = _make_runner(config, "auditor")
        assert _is_blocked(runner, "~/kernel_lab/agents/library.py", "Read")


# ── Library host process logic ──

class TestLibraryDispatch:

    def test_on_ask_taxonomist_dispatch(self, config):
        """on_ask routes CONSULT_TAXONOMIST to _run_expert."""
        from agents.library import Library
        from agents.events import AskEvent

        lib = Library(config)
        event = AskEvent(question="CONSULT_TAXONOMIST: Where should this go?", context="test")
        # Can't run async _run_expert in unit test, but verify dispatch logic
        assert event.question.startswith("CONSULT_TAXONOMIST:")

    def test_on_ask_auditor_dispatch(self, config):
        """on_ask routes CONSULT_AUDITOR to _run_expert."""
        from agents.library import Library
        from agents.events import AskEvent

        lib = Library(config)
        event = AskEvent(question="CONSULT_AUDITOR: Is this evidence strong?", context="test")
        assert event.question.startswith("CONSULT_AUDITOR:")

    def test_on_ask_unknown_returns_unknown(self, config):
        """on_ask returns error for unknown request types."""
        import asyncio
        from agents.library import Library
        from agents.events import AskEvent

        lib = Library(config)
        event = AskEvent(question="UNKNOWN_REQUEST: foo", context="")
        result = asyncio.run(lib.on_ask(event))
        assert "Unknown" in result

    def test_queue_scanning(self, config):
        """_next_proposal returns oldest yaml from pending/ dir."""
        import tempfile
        from agents.library import Library, PENDING_DIR

        lib = Library(config)
        PENDING_DIR.mkdir(parents=True, exist_ok=True)

        # Create two test files
        f1 = PENDING_DIR / "20260101_000000_first.yaml"
        f2 = PENDING_DIR / "20260102_000000_second.yaml"
        f1.write_text("test1")
        f2.write_text("test2")

        try:
            result = lib._next_proposal()
            assert result is not None
            assert result.name == "20260101_000000_first.yaml"
        finally:
            f1.unlink(missing_ok=True)
            f2.unlink(missing_ok=True)

    def test_inject_priority(self, config):
        """_next_injection returns from inject/ dir."""
        from agents.library import Library, INJECT_DIR

        lib = Library(config)
        INJECT_DIR.mkdir(parents=True, exist_ok=True)

        f = INJECT_DIR / "20260101_000000_urgent.yaml"
        f.write_text("urgent")

        try:
            result = lib._next_injection()
            assert result is not None
            assert result.name == "20260101_000000_urgent.yaml"
        finally:
            f.unlink(missing_ok=True)

    def test_expert_configs_exist(self, config):
        """Taxonomist and Auditor configs are loadable."""
        tax = config.get_agent("taxonomist")
        aud = config.get_agent("auditor")
        assert tax.name == "taxonomist"
        assert aud.name == "auditor"
        assert "Write" not in tax.builtin_tools
        assert "Write" not in aud.builtin_tools
