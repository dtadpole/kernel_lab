"""CLI-level tests: verify doc_retrieval subcommands work end-to-end."""

import json
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
# venv may be in the main repo root, not a worktree copy
_VENV = _REPO_ROOT / "doc_retrieval" / ".venv" / "bin" / "python"
if not _VENV.exists():
    # fall back to the main repo
    _VENV = Path("/home/centos/kernel_lab/doc_retrieval/.venv/bin/python")
_PYTHON = str(_VENV)


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_PYTHON, "-m", "doc_retrieval", *args],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.quick
def test_find_returns_results():
    """find subcommand returns results with expected fields."""
    result = _run_cli("find", "shared memory bank conflicts", "--mode", "bm25")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    output = result.stdout
    assert "Result 1" in output
    assert "score:" in output.lower() or "score" in output
    assert "Title:" in output
    assert "URL:" in output


@pytest.mark.quick
def test_browse_returns_toc():
    """browse subcommand returns TOC structure."""
    result = _run_cli("browse", "cuda-c-programming-guide", "--depth", "1")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    data = json.loads(result.stdout)
    assert isinstance(data, list)
    assert len(data) > 0
    assert "section_id" in data[0]
    assert "title" in data[0]


@pytest.mark.quick
def test_read_returns_section():
    """read subcommand returns a section with content and nav."""
    result = _run_cli("read", "cuda-c-programming-guide", "programming-model")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    data = json.loads(result.stdout)
    assert "title" in data
    assert "content" in data
    assert "nav" in data
