"""Plugin-level tests: loading, tool registration, integrity."""

import importlib.util
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
def test_kb_mcp_config():
    """.mcp.json exists and references the MCP server."""
    mcp_json = _PLUGIN_DIR / ".mcp.json"
    assert mcp_json.exists()
    data = json.loads(mcp_json.read_text())
    assert "kb" in data["mcpServers"]


@pytest.mark.quick
def test_kb_mcp_server_imports():
    """MCP server module can be imported without errors."""
    spec = importlib.util.spec_from_file_location(
        "kb_mcp", _PLUGIN_DIR / "mcp_server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "mcp")


@pytest.mark.quick
def test_kb_mcp_tools_registered():
    """All 4 expected tools are registered."""
    spec = importlib.util.spec_from_file_location(
        "kb_mcp", _PLUGIN_DIR / "mcp_server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tools = set(mod.mcp._tool_manager._tools.keys())
    expected = {"search_docs", "lookup_doc_section", "browse_toc", "read_section"}
    assert expected == tools, f"Missing: {expected - tools}, Extra: {tools - expected}"


@pytest.mark.quick
def test_kb_skills_exist():
    """All skill directories have SKILL.md."""
    skills_dir = _PLUGIN_DIR / "skills"
    expected_skills = {"search", "ingest", "rebuild", "download"}
    for skill_name in expected_skills:
        skill_file = skills_dir / skill_name / "SKILL.md"
        assert skill_file.exists(), f"Missing {skill_file}"
