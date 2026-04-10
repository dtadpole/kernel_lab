"""Tests for Library agent permission isolation.

Verifies that tool_rules in agents.yaml correctly enforce:
- Librarian: write wiki/ but NOT _proposals/
- Analyst: write _proposals/pending/ but NOT wiki categories
- Taxonomist/Auditor: read-only, wiki/ only
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
