"""Tests for CLI subcommand structure and renamed find command.

Validates that the CLI refactor (search→find rename, MCP removal) is correct.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "kb"
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# CLI structure tests
# ---------------------------------------------------------------------------

class TestCLIStructure:
    """Verify CLI subcommands are correctly registered."""

    def test_help_lists_find_subcommand(self):
        result = subprocess.run(
            [PYTHON, "-m", "doc_retrieval", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "find" in result.stdout
        assert "search" not in result.stdout.split("{")[1].split("}")[0], \
            "Old 'search' subcommand should not appear in subcommand list"

    def test_find_help(self):
        result = subprocess.run(
            [PYTHON, "-m", "doc_retrieval", "find", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "--mode" in result.stdout
        assert "--top-k" in result.stdout

    def test_browse_help(self):
        result = subprocess.run(
            [PYTHON, "-m", "doc_retrieval", "browse", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "doc_id" in result.stdout

    def test_read_help(self):
        result = subprocess.run(
            [PYTHON, "-m", "doc_retrieval", "read", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "section_id" in result.stdout

    def test_download_help(self):
        result = subprocess.run(
            [PYTHON, "-m", "doc_retrieval", "download", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        # PDF flags should be gone
        assert "--tier" not in result.stdout
        assert "--pdf-only" not in result.stdout

    def test_index_help(self):
        result = subprocess.run(
            [PYTHON, "-m", "doc_retrieval", "index", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "--only" in result.stdout

    def test_old_search_subcommand_rejected(self):
        result = subprocess.run(
            [PYTHON, "-m", "doc_retrieval", "search", "test query"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0, \
            "Old 'search' subcommand should be rejected"


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

class TestImports:
    """Verify renamed functions are importable."""

    def test_cli_find_importable(self):
        from doc_retrieval.searcher import cli_find
        assert callable(cli_find)

    def test_cli_search_removed(self):
        from doc_retrieval import searcher
        assert not hasattr(searcher, "cli_search"), \
            "cli_search should have been renamed to cli_find"


# ---------------------------------------------------------------------------
# Plugin structure tests
# ---------------------------------------------------------------------------

class TestPluginStructure:
    """Verify KB plugin files after MCP removal."""

    def test_mcp_server_deleted(self):
        assert not (PLUGIN_DIR / "mcp_server.py").exists(), \
            "mcp_server.py should have been deleted"

    def test_mcp_json_deleted(self):
        assert not (PLUGIN_DIR / ".mcp.json").exists(), \
            ".mcp.json should have been deleted"

    def test_old_skills_deleted(self):
        for name in ("search", "download", "rebuild", "ingest"):
            assert not (PLUGIN_DIR / "skills" / name).exists(), \
                f"Old skill '{name}' directory should have been deleted"

    def test_docs_skill_exists(self):
        skill = PLUGIN_DIR / "skills" / "docs" / "SKILL.md"
        assert skill.exists()
        content = skill.read_text()
        assert "find" in content
        assert "read" in content
        assert "browse" in content

    def test_index_skill_exists(self):
        skill = PLUGIN_DIR / "skills" / "index" / "SKILL.md"
        assert skill.exists()
        content = skill.read_text()
        assert "download" in content
        assert "parse" in content
        assert "index" in content

    def test_plugin_json_no_mcp(self):
        import json
        pj = json.loads((PLUGIN_DIR / ".claude-plugin" / "plugin.json").read_text())
        desc = pj.get("description", "").lower()
        assert "mcp" not in desc, \
            "plugin.json description should not reference MCP"
