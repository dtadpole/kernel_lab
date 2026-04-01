"""CLI tool-level tests: verify doc_retrieval CLI commands work end-to-end.

These tests require the search index to be built (run kb:index rebuild first).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON = str(REPO_ROOT / "doc_retrieval" / ".venv" / "bin" / "python")


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, "-m", "doc_retrieval", *args],
        capture_output=True, text=True, timeout=30,
        cwd=str(REPO_ROOT),
    )


@pytest.mark.integration
def test_find_returns_results():
    """find returns search results with expected fields."""
    r = _run_cli("find", "shared memory bank conflicts", "--mode", "bm25", "--top-k", "3")
    assert r.returncode == 0
    # Output contains "Result 1" when results are found
    assert "Result 1" in r.stdout


@pytest.mark.integration
def test_browse_returns_toc():
    """browse returns JSON TOC structure for a known doc."""
    r = _run_cli("browse", "cuda-c-programming-guide", "--depth", "1")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert isinstance(data, list)
    assert len(data) > 0
    assert "section_id" in data[0]


@pytest.mark.integration
def test_read_returns_section():
    """read returns section content with navigation."""
    r = _run_cli("read", "cuda-c-programming-guide", "introduction")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert "content" in data
    assert "nav" in data
