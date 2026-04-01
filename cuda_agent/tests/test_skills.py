"""Tests for cuda_agent.skills — skill file loader."""

from pathlib import Path

import pytest

from cuda_agent.skills import load_plugin_skill, superpowers_base_dir, superpowers_skill_path

pytestmark = pytest.mark.quick


class TestLoadPluginSkill:
    def test_loads_cuda_exec_skill(self):
        content = load_plugin_skill("cuda", "exec")
        assert "name: exec" in content
        assert "compile" in content.lower()

    def test_loads_cuda_inspect_skill(self):
        content = load_plugin_skill("cuda", "inspect")
        assert "name: inspect" in content

    def test_loads_kb_docs_skill(self):
        content = load_plugin_skill("kb", "docs")
        assert "name: docs" in content
        assert "doc_retrieval" in content

    def test_nonexistent_skill_raises(self):
        with pytest.raises(FileNotFoundError):
            load_plugin_skill("cuda", "nonexistent")


class TestSuperpowersPath:
    def test_base_dir_exists(self):
        base = superpowers_base_dir()
        assert base.is_dir()

    def test_skill_path_returns_existing_file(self):
        path = superpowers_skill_path("brainstorming")
        assert Path(path).is_file()

    def test_skill_path_has_skill_md_suffix(self):
        path = superpowers_skill_path("writing-plans")
        assert path.endswith("SKILL.md")
