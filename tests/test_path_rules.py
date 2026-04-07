"""Tests for ToolRule blocked_paths / allowed_paths enforcement."""

import os
import pytest

from agents.config import AgentConfig, StorageConfig, ToolRule
from agents.runner import AgentRunner


def _make_runner(blocked_paths, allowed_paths, run_tag="test_run_123"):
    """Create a minimal AgentRunner with Read+Bash tool rules for testing."""
    rule_kwargs = dict(allow=True, blocked_paths=blocked_paths, allowed_paths=allowed_paths)
    config = AgentConfig(
        name="test_solver",
        tool_rules=[
            ToolRule(tool="Read", constraint="test", **rule_kwargs),
            ToolRule(tool="Bash", **rule_kwargs),
        ],
    )
    storage = StorageConfig(run_tag=run_tag)
    return AgentRunner(
        agent_config=config,
        storage_config=storage,
        cwd="/home/zhenc/kernel_lab",
    )


# ── Basic blocking ──

@pytest.mark.quick
def test_blocked_path_absolute():
    runner = _make_runner(
        blocked_paths=["/home/zhenc/kernel_lab/"],
        allowed_paths=[],
    )
    result = runner._check_tool_rules("Read", {"file_path": "/home/zhenc/kernel_lab/data/peak/cuda.cu"})
    assert result.get("decision") == "block"


@pytest.mark.quick
def test_blocked_path_with_tilde():
    runner = _make_runner(
        blocked_paths=["~/kernel_lab/"],
        allowed_paths=[],
    )
    result = runner._check_tool_rules("Read", {"file_path": "/home/zhenc/kernel_lab/data/gen/cuda.cu"})
    assert result.get("decision") == "block"


@pytest.mark.quick
def test_unblocked_path_passes():
    runner = _make_runner(
        blocked_paths=["~/kernel_lab/"],
        allowed_paths=[],
    )
    result = runner._check_tool_rules("Read", {"file_path": "/tmp/something.txt"})
    assert result.get("decision") != "block"


# ── Allowed overrides blocked ──

@pytest.mark.quick
def test_allowed_overrides_blocked_absolute():
    runner = _make_runner(
        blocked_paths=["~/kernel_lab/"],
        allowed_paths=["~/kernel_lab/cuda_exec/"],
    )
    result = runner._check_tool_rules("Read", {"file_path": "/home/zhenc/kernel_lab/cuda_exec/impls.py"})
    assert result.get("decision") != "block"


@pytest.mark.quick
def test_allowed_overrides_blocked_relative():
    runner = _make_runner(
        blocked_paths=["~/kernel_lab/"],
        allowed_paths=["cuda_exec/"],
    )
    result = runner._check_tool_rules("Read", {"file_path": "cuda_exec/impls.py"})
    assert result.get("decision") != "block"


@pytest.mark.quick
def test_blocked_not_in_allowed():
    runner = _make_runner(
        blocked_paths=["~/kernel_lab/"],
        allowed_paths=["~/kernel_lab/cuda_exec/"],
    )
    result = runner._check_tool_rules("Read", {"file_path": "/home/zhenc/kernel_lab/data/peak/cuda.cu"})
    assert result.get("decision") == "block"


# ── Dynamic <run_tag> substitution ──

@pytest.mark.quick
def test_run_tag_substitution_allows_current_run():
    runner = _make_runner(
        blocked_paths=["~/kernel_lab_kb/"],
        allowed_paths=["~/kernel_lab_kb/runs/<run_tag>/"],
        run_tag="supervisor_run_20260406_215504",
    )
    result = runner._check_tool_rules("Read", {
        "file_path": "/home/zhenc/kernel_lab_kb/runs/supervisor_run_20260406_215504/gen/cuda.cu"
    })
    assert result.get("decision") != "block"


@pytest.mark.quick
def test_run_tag_substitution_blocks_other_run():
    runner = _make_runner(
        blocked_paths=["~/kernel_lab_kb/"],
        allowed_paths=["~/kernel_lab_kb/runs/<run_tag>/"],
        run_tag="supervisor_run_20260406_215504",
    )
    result = runner._check_tool_rules("Read", {
        "file_path": "/home/zhenc/kernel_lab_kb/runs/run_h8_4/gems/v001/cuda.cu"
    })
    assert result.get("decision") == "block"


# ── Full solver config scenario ──

@pytest.mark.quick
def test_solver_read_rules_full():
    """Test the actual Solver read rules from agents.yaml."""
    run_tag = "supervisor_run_20260406_215504"
    config = AgentConfig(
        name="solver",
        tool_rules=[
            ToolRule(
                tool="Read",
                allow=True,
                blocked_paths=[
                    "~/kernel_lab/",
                    "~/kernel_lab_kb/",
                ],
                allowed_paths=[
                    "cuda_exec/",
                    "plugins/",
                    "conf/",
                    "DESIGN.md",
                    "AGENTS.md",
                    "~/kernel_lab/cuda_exec/",
                    "~/kernel_lab/plugins/",
                    "~/kernel_lab/conf/",
                    "~/kernel_lab/DESIGN.md",
                    "~/kernel_lab/AGENTS.md",
                    f"~/kernel_lab_kb/runs/<run_tag>/",
                    f"~/.cuda_exec/<run_tag>/",
                ],
            ),
        ],
    )
    storage = StorageConfig(run_tag=run_tag)
    runner = AgentRunner(
        agent_config=config, storage_config=storage,
        cwd="/home/zhenc/kernel_lab",
    )

    # Allowed
    assert runner._check_tool_rules("Read", {"file_path": "/home/zhenc/kernel_lab/cuda_exec/impls.py"}).get("decision") != "block"
    assert runner._check_tool_rules("Read", {"file_path": "/home/zhenc/kernel_lab/plugins/ik/SKILL.md"}).get("decision") != "block"
    assert runner._check_tool_rules("Read", {"file_path": "/home/zhenc/kernel_lab/conf/agent/agents.yaml"}).get("decision") != "block"
    assert runner._check_tool_rules("Read", {"file_path": "/home/zhenc/kernel_lab/DESIGN.md"}).get("decision") != "block"
    assert runner._check_tool_rules("Read", {"file_path": f"/home/zhenc/kernel_lab_kb/runs/{run_tag}/gen/cuda.cu"}).get("decision") != "block"

    # Blocked
    assert runner._check_tool_rules("Read", {"file_path": "/home/zhenc/kernel_lab/data/peak/sm90/cuda.cu"}).get("decision") == "block"
    assert runner._check_tool_rules("Read", {"file_path": "/home/zhenc/kernel_lab/data/gen/sm90/cuda.cu"}).get("decision") == "block"
    assert runner._check_tool_rules("Read", {"file_path": "/home/zhenc/kernel_lab/docs/design/supervisor.md"}).get("decision") == "block"
    assert runner._check_tool_rules("Read", {"file_path": "/home/zhenc/kernel_lab_kb/runs/run_h8_4/gems/v001/cuda.cu"}).get("decision") == "block"
    assert runner._check_tool_rules("Read", {"file_path": "/home/zhenc/kernel_lab/agents/supervisor.py"}).get("decision") == "block"


# ── Glob and Grep use path param ──

@pytest.mark.quick
def test_glob_blocked():
    config = AgentConfig(
        name="solver",
        tool_rules=[
            ToolRule(tool="Glob", allow=True,
                     blocked_paths=["~/kernel_lab/"], allowed_paths=["~/kernel_lab/cuda_exec/"]),
        ],
    )
    runner = AgentRunner(
        agent_config=config, storage_config=StorageConfig(run_tag="test"),
        cwd="/home/zhenc/kernel_lab",
    )
    result = runner._check_tool_rules("Glob", {"path": "/home/zhenc/kernel_lab/data/peak"})
    assert result.get("decision") == "block"

    result = runner._check_tool_rules("Glob", {"path": "/home/zhenc/kernel_lab/cuda_exec"})
    assert result.get("decision") != "block"


# ── Bash command scanning ──

@pytest.mark.quick
def test_bash_cat_blocked_path():
    runner = _make_runner(
        blocked_paths=["~/kernel_lab_kb/"],
        allowed_paths=["~/kernel_lab_kb/runs/<run_tag>/"],
        run_tag="my_run",
    )
    # cat from another run — should be blocked
    result = runner._check_tool_rules("Bash", {
        "command": "cat ~/kernel_lab_kb/runs/old_run/gems/v001/cuda.cu"
    })
    assert result.get("decision") == "block"


@pytest.mark.quick
def test_bash_cat_allowed_path():
    runner = _make_runner(
        blocked_paths=["~/kernel_lab_kb/"],
        allowed_paths=["~/kernel_lab_kb/runs/<run_tag>/"],
        run_tag="my_run",
    )
    # cat from current run — should be allowed
    result = runner._check_tool_rules("Bash", {
        "command": "cat ~/kernel_lab_kb/runs/my_run/gen/cuda.cu"
    })
    assert result.get("decision") != "block"


@pytest.mark.quick
def test_bash_cat_blocked_data():
    """cat from blocked data/ dir — should block."""
    runner = _make_runner(
        blocked_paths=["~/kernel_lab/"],
        allowed_paths=["~/kernel_lab/cuda_exec/"],
    )
    result = runner._check_tool_rules("Bash", {
        "command": "cat ~/kernel_lab/data/peak/sm90/matmul/cuda.cu"
    })
    assert result.get("decision") == "block"


@pytest.mark.quick
def test_bash_cat_allowed_path():
    """cat from allowed cuda_exec/ — should pass."""
    runner = _make_runner(
        blocked_paths=["~/kernel_lab/"],
        allowed_paths=["~/kernel_lab/cuda_exec/"],
    )
    result = runner._check_tool_rules("Bash", {
        "command": "cat ~/kernel_lab/cuda_exec/impls.py"
    })
    assert result.get("decision") != "block"


@pytest.mark.quick
def test_bash_execution_venv_allowed():
    """Running python from .venv/ — should pass (.venv/ whitelisted)."""
    runner = _make_runner(
        blocked_paths=["~/kernel_lab/"],
        allowed_paths=["~/kernel_lab/.venv/"],
    )
    result = runner._check_tool_rules("Bash", {
        "command": "/home/zhenc/kernel_lab/.venv/bin/python -m cuda_exec.exec_cli exec.action=compile"
    })
    assert result.get("decision") != "block"


@pytest.mark.quick
def test_bash_execution_data_gen_blocked():
    """Running nvcc on data/gen/ — should block."""
    runner = _make_runner(
        blocked_paths=["~/kernel_lab/"],
        allowed_paths=["~/kernel_lab/.venv/"],
    )
    result = runner._check_tool_rules("Bash", {
        "command": "nvcc /home/zhenc/kernel_lab/data/gen/sm90/matmul/cuda.cu -o output"
    })
    assert result.get("decision") == "block"


@pytest.mark.quick
def test_bash_execution_data_peak_blocked():
    """Running nvcc on data/peak/ — should block."""
    runner = _make_runner(
        blocked_paths=["~/kernel_lab/"],
        allowed_paths=["~/kernel_lab/.venv/"],
    )
    result = runner._check_tool_rules("Bash", {
        "command": "nvcc /home/zhenc/kernel_lab/data/peak/sm90/matmul/cuda.cu -o output"
    })
    assert result.get("decision") == "block"


@pytest.mark.quick
def test_bash_no_path_passes():
    runner = _make_runner(
        blocked_paths=["~/kernel_lab/"],
        allowed_paths=[],
    )
    result = runner._check_tool_rules("Bash", {
        "command": "nvidia-smi"
    })
    assert result.get("decision") != "block"


@pytest.mark.quick
def test_bash_worktrees_blocked():
    config = AgentConfig(
        name="solver",
        tool_rules=[
            ToolRule(tool="Bash", allow=True,
                     blocked_paths=["~/.claude/worktrees/"],
                     allowed_paths=[]),
        ],
    )
    runner = AgentRunner(
        agent_config=config, storage_config=StorageConfig(run_tag="test"),
        cwd="/home/zhenc/kernel_lab",
    )
    result = runner._check_tool_rules("Bash", {
        "command": "cat ~/.claude/worktrees/fa4-4wg/data/gen/sm90/matmul/cuda/cuda.cu"
    })
    assert result.get("decision") == "block"


@pytest.mark.quick
def test_bash_git_forbidden():
    """git commands are always forbidden."""
    runner = _make_runner(blocked_paths=[], allowed_paths=[])
    assert runner._check_tool_rules("Bash", {"command": "git log --oneline"}).get("decision") == "block"
    assert runner._check_tool_rules("Bash", {"command": "git show abc123"}).get("decision") == "block"
    assert runner._check_tool_rules("Bash", {"command": "git log -p --all"}).get("decision") == "block"
    assert runner._check_tool_rules("Bash", {"command": "git diff HEAD~5"}).get("decision") == "block"
    assert runner._check_tool_rules("Bash", {"command": 'ssh localhost "git log"'}).get("decision") == "block"


@pytest.mark.quick
def test_bash_head_blocked():
    runner = _make_runner(
        blocked_paths=["~/kernel_lab_kb/"],
        allowed_paths=["~/kernel_lab_kb/runs/<run_tag>/"],
        run_tag="my_run",
    )
    result = runner._check_tool_rules("Bash", {
        "command": "head -50 /home/zhenc/kernel_lab_kb/runs/other_run/gems/cuda.cu"
    })
    assert result.get("decision") == "block"


@pytest.mark.quick
def test_bash_ls_navigation_allowed():
    """ls on blocked path — allowed (navigation command)."""
    runner = _make_runner(
        blocked_paths=["~/kernel_lab/"],
        allowed_paths=[],
    )
    result = runner._check_tool_rules("Bash", {
        "command": "ls ~/kernel_lab/data/"
    })
    assert result.get("decision") != "block"


@pytest.mark.quick
def test_bash_find_blocked():
    """find on blocked path — blocked (not a navigation command)."""
    runner = _make_runner(
        blocked_paths=["~/kernel_lab/"],
        allowed_paths=[],
    )
    result = runner._check_tool_rules("Bash", {
        "command": "find ~/kernel_lab/data/peak -name '*.cu'"
    })
    assert result.get("decision") == "block"


@pytest.mark.quick
def test_bash_ls_kernel_lab_kb_allowed():
    """ls on kernel_lab_kb/runs/ — allowed (navigation)."""
    runner = _make_runner(
        blocked_paths=["~/kernel_lab_kb/"],
        allowed_paths=["~/kernel_lab_kb/runs/<run_tag>/"],
        run_tag="my_run",
    )
    result = runner._check_tool_rules("Bash", {
        "command": "ls ~/kernel_lab_kb/runs/"
    })
    assert result.get("decision") != "block"


@pytest.mark.quick
def test_bash_ssh_cat_blocked():
    """ssh localhost cat from blocked path — should block."""
    runner = _make_runner(
        blocked_paths=["~/kernel_lab/"],
        allowed_paths=["~/kernel_lab/data/ref/"],
    )
    result = runner._check_tool_rules("Bash", {
        "command": 'ssh localhost "cat ~/kernel_lab/data/peak/sm90/cuda.cu"'
    })
    assert result.get("decision") == "block"


@pytest.mark.quick
def test_bash_ssh_cat_allowed():
    """ssh localhost cat from allowed path — should pass."""
    runner = _make_runner(
        blocked_paths=["~/kernel_lab/"],
        allowed_paths=["~/kernel_lab/data/ref/"],
    )
    result = runner._check_tool_rules("Bash", {
        "command": 'ssh localhost "cat ~/kernel_lab/data/ref/matmul/cublas/cublas.cu"'
    })
    assert result.get("decision") != "block"


@pytest.mark.quick
def test_read_data_ref_allowed():
    """Read from data/ref/ — should pass (whitelisted)."""
    runner = _make_runner(
        blocked_paths=["~/kernel_lab/"],
        allowed_paths=["~/kernel_lab/data/ref/", "~/kernel_lab/data/configs/"],
    )
    result = runner._check_tool_rules("Read", {
        "file_path": "/home/zhenc/kernel_lab/data/ref/matmul/cublas/cublas.cu"
    })
    assert result.get("decision") != "block"


@pytest.mark.quick
def test_read_data_gen_blocked():
    """Read from data/gen/ — should block (not whitelisted)."""
    runner = _make_runner(
        blocked_paths=["~/kernel_lab/"],
        allowed_paths=["~/kernel_lab/data/ref/", "~/kernel_lab/data/configs/"],
    )
    result = runner._check_tool_rules("Read", {
        "file_path": "/home/zhenc/kernel_lab/data/gen/sm90/matmul/cuda.cu"
    })
    assert result.get("decision") == "block"


@pytest.mark.quick
def test_read_data_peak_blocked():
    """Read from .peak/ — should block."""
    runner = _make_runner(
        blocked_paths=["~/kernel_lab/"],
        allowed_paths=["~/kernel_lab/data/ref/", "~/kernel_lab/data/configs/"],
    )
    result = runner._check_tool_rules("Read", {
        "file_path": "/home/zhenc/kernel_lab/.peak/sm90/matmul/cuda.cu"
    })
    assert result.get("decision") == "block"
