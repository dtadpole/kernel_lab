"""Plugin-level tests: loading, tool registration, integrity."""

import importlib.util
from pathlib import Path

import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Quick tests — no remote access
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_cuda_plugin_manifest():
    """plugin.json exists and has required fields."""
    import json
    manifest = _PLUGIN_DIR / ".claude-plugin" / "plugin.json"
    assert manifest.exists(), f"Missing {manifest}"
    data = json.loads(manifest.read_text())
    assert data["name"] == "cuda"
    assert "description" in data
    assert "version" in data


@pytest.mark.quick
def test_cuda_mcp_config():
    """.mcp.json exists and references the MCP server."""
    import json
    mcp_json = _PLUGIN_DIR / ".mcp.json"
    assert mcp_json.exists(), f"Missing {mcp_json}"
    data = json.loads(mcp_json.read_text())
    assert "cuda" in data["mcpServers"]


@pytest.mark.quick
def test_cuda_mcp_server_imports():
    """MCP server module can be imported without errors."""
    spec = importlib.util.spec_from_file_location(
        "cuda_mcp", _PLUGIN_DIR / "mcp_server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "mcp")


@pytest.mark.quick
def test_cuda_mcp_tools_registered():
    """All 10 expected tools are registered."""
    spec = importlib.util.spec_from_file_location(
        "cuda_mcp", _PLUGIN_DIR / "mcp_server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tools = set(mod.mcp._tool_manager._tools.keys())
    expected = {
        "compile", "evaluate", "profile", "execute", "read_file",
        "get_data_point", "get_compile_data", "get_evaluate_data",
        "get_profile_data",
    }
    assert expected == tools, f"Missing: {expected - tools}, Extra: {tools - expected}"


@pytest.mark.quick
def test_cuda_skills_exist():
    """All skill directories have SKILL.md."""
    skills_dir = _PLUGIN_DIR / "skills"
    expected_skills = {"exec", "inspect", "service"}
    for skill_name in expected_skills:
        skill_file = skills_dir / skill_name / "SKILL.md"
        assert skill_file.exists(), f"Missing {skill_file}"


@pytest.mark.quick
def test_cuda_deploy_cli_syntax():
    """deploy/cli.py has valid Python syntax."""
    import ast
    cli_path = _PLUGIN_DIR / "deploy" / "cli.py"
    ast.parse(cli_path.read_text())


@pytest.mark.quick
def test_cuda_systemd_unit_exists():
    """Hardened systemd unit template exists."""
    unit = _PLUGIN_DIR / "deploy" / "cuda-exec.service"
    assert unit.exists()
    content = unit.read_text()
    assert "ExecStart=" in content
    assert "NoNewPrivileges=true" in content
