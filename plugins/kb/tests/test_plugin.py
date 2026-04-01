"""Plugin-level tests: loading, skill integrity, MCP removal verification."""

import json
from pathlib import Path

import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.quick
def test_kb_plugin_manifest():
    """plugin.json exists and has required fields."""
    manifest = _PLUGIN_DIR / ".claude-plugin" / "plugin.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text())
    assert data["name"] == "kb"
    assert "description" in data


@pytest.mark.quick
def test_plugin_json_no_mcp():
    """plugin.json should not reference MCP (CLI-only plugin)."""
    pj = json.loads((_PLUGIN_DIR / ".claude-plugin" / "plugin.json").read_text())
    desc = pj.get("description", "").lower()
    assert "mcp" not in desc


@pytest.mark.quick
def test_mcp_removed():
    """MCP infrastructure should not exist (CLI-only plugin)."""
    assert not (_PLUGIN_DIR / ".mcp.json").exists(), \
        ".mcp.json should not exist"
    assert not (_PLUGIN_DIR / "mcp_server.py").exists(), \
        "mcp_server.py should not exist"


@pytest.mark.quick
def test_old_skills_deleted():
    """Old skill directories from pre-refactor should not exist."""
    for name in ("search", "download", "rebuild", "ingest"):
        assert not (_PLUGIN_DIR / "skills" / name).exists(), \
            f"Old skill '{name}' directory should have been deleted"


@pytest.mark.quick
def test_current_skills_exist():
    """All current skill directories have SKILL.md."""
    for skill_name in ("docs", "index", "service"):
        skill_file = _PLUGIN_DIR / "skills" / skill_name / "SKILL.md"
        assert skill_file.exists(), f"Missing {skill_file}"


@pytest.mark.quick
def test_docs_skill_references_cli():
    """docs skill invokes doc_retrieval CLI commands."""
    content = (_PLUGIN_DIR / "skills" / "docs" / "SKILL.md").read_text()
    assert "find" in content
    assert "read" in content
    assert "browse" in content


@pytest.mark.quick
def test_index_skill_references_cli():
    """index skill invokes doc_retrieval CLI commands."""
    content = (_PLUGIN_DIR / "skills" / "index" / "SKILL.md").read_text()
    assert "download" in content
    assert "parse" in content
    assert "index" in content


@pytest.mark.quick
def test_service_skill_references_deploy_cli():
    """service skill references the deploy CLI."""
    content = (_PLUGIN_DIR / "skills" / "service" / "SKILL.md").read_text()
    assert "deploy" in content
    assert "start" in content
    assert "health" in content


@pytest.mark.quick
def test_commands_match_skills():
    """Each user-invocable skill has a matching command file."""
    commands_dir = _PLUGIN_DIR / "commands"
    for cmd_name in ("docs", "index", "service"):
        cmd_file = commands_dir / f"{cmd_name}.md"
        assert cmd_file.exists(), f"Missing command file {cmd_file}"
