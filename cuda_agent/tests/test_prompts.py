"""Tests for cuda_agent.prompts — system prompt builder."""

import pytest

from cuda_agent.prompts import build_system_prompt, format_initial_prompt
from cuda_agent.task import OptimizationTask

pytestmark = pytest.mark.quick


def _make_task(**overrides) -> OptimizationTask:
    defaults = dict(
        run_tag="test_run",
        version="v1",
        direction_id=7,
        direction_slug="vecadd",
        reference_files={"ref.py": "# ref"},
        initial_generated_files={"gen.cu": "// gen"},
        configs={"cfg1": {"shape": [1024]}},
    )
    defaults.update(overrides)
    return OptimizationTask(**defaults)


class TestBuildSystemPrompt:
    def test_contains_role(self):
        prompt = build_system_prompt(_make_task())
        assert "CUDA kernel optimization" in prompt

    def test_contains_three_phases(self):
        prompt = build_system_prompt(_make_task())
        assert "Phase 1" in prompt
        assert "Phase 2" in prompt
        assert "Phase 3" in prompt
        assert "brainstorming" in prompt

    def test_contains_superpowers_paths(self):
        prompt = build_system_prompt(_make_task())
        assert "SKILL.md" in prompt

    def test_contains_platform_adaptation(self):
        prompt = build_system_prompt(_make_task())
        assert "Platform Adaptation" in prompt

    def test_contains_cuda_exec_skill(self):
        prompt = build_system_prompt(_make_task())
        assert "compile" in prompt.lower()
        assert "evaluate" in prompt.lower()

    def test_contains_inspect_cli_docs(self):
        prompt = build_system_prompt(_make_task())
        assert "inspect_cli" in prompt
        assert "vecadd" in prompt

    def test_contains_kb_docs_skill(self):
        prompt = build_system_prompt(_make_task())
        assert "doc_retrieval" in prompt

    def test_data_dir_uses_task_metadata(self):
        prompt = build_system_prompt(_make_task(
            run_tag="R", version="V", direction_id=3, direction_slug="S",
        ))
        assert "R/V/3_S" in prompt


class TestFormatInitialPrompt:
    def test_contains_metadata(self):
        prompt = format_initial_prompt(_make_task())
        assert "test_run" in prompt
        assert "vecadd" in prompt

    def test_contains_code(self):
        prompt = format_initial_prompt(_make_task())
        assert "# ref" in prompt
        assert "// gen" in prompt
