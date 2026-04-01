"""Tests for agent.py hooks — edit path restriction logic.

Tests the pure path-checking function is_path_allowed() which does not
depend on claude_agent_sdk types, so it can run in the project venv.
"""

from pathlib import Path

import pytest

from cuda_agent.path_check import is_path_allowed

pytestmark = pytest.mark.quick


class TestIsPathAllowed:
    def test_file_inside_allowed_dir(self, tmp_path: Path):
        assert is_path_allowed(str(tmp_path / "gen.cu"), str(tmp_path)) is True

    def test_file_in_subdirectory(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        assert is_path_allowed(str(sub / "file.cu"), str(tmp_path)) is True

    def test_file_outside_allowed_dir(self, tmp_path: Path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        assert is_path_allowed("/etc/passwd", str(allowed)) is False

    def test_parent_traversal_denied(self, tmp_path: Path):
        allowed = tmp_path / "sub"
        allowed.mkdir()
        # ../escape.txt resolves to tmp_path/escape.txt — outside allowed
        assert is_path_allowed(str(allowed / ".." / "escape.txt"), str(allowed)) is False

    def test_sibling_directory_denied(self, tmp_path: Path):
        a = tmp_path / "a"
        a.mkdir()
        b = tmp_path / "b"
        b.mkdir()
        assert is_path_allowed(str(b / "file.txt"), str(a)) is False

    def test_allowed_dir_itself(self, tmp_path: Path):
        # Edge case: the allowed dir path itself should be allowed
        assert is_path_allowed(str(tmp_path), str(tmp_path)) is True

    def test_prefix_collision(self, tmp_path: Path):
        # /tmp/foo should NOT allow /tmp/foobar/file
        foo = tmp_path / "foo"
        foo.mkdir()
        foobar = tmp_path / "foobar"
        foobar.mkdir()
        assert is_path_allowed(str(foobar / "file.txt"), str(foo)) is False
