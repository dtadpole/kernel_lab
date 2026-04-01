"""Plugin-level tests: manifest, skills, structure integrity."""

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
def test_kb_no_mcp():
    """KB plugin is skill-only — no MCP server or .mcp.json."""
    assert not (_PLUGIN_DIR / ".mcp.json").exists()
    assert not (_PLUGIN_DIR / "mcp_server.py").exists()


@pytest.mark.quick
def test_kb_skills_exist():
    """All skill directories have SKILL.md."""
    skills_dir = _PLUGIN_DIR / "skills"
    expected_skills = {"docs", "index"}
    for skill_name in expected_skills:
        skill_file = skills_dir / skill_name / "SKILL.md"
        assert skill_file.exists(), f"Missing {skill_file}"


@pytest.mark.quick
def test_kb_skills_have_frontmatter():
    """Each SKILL.md has required frontmatter fields."""
    skills_dir = _PLUGIN_DIR / "skills"
    for skill_dir in skills_dir.iterdir():
        if not skill_dir.is_dir():
            continue
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        content = skill_file.read_text()
        assert content.startswith("---"), f"{skill_file} missing frontmatter"
        assert "name:" in content, f"{skill_file} missing name field"
        assert "description:" in content, f"{skill_file} missing description field"
