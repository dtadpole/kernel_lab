"""Tests for cuda_agent.inspect_cli -- local data store CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.quick


def _run_inspect(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "cuda_agent.inspect_cli", *args],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )


class TestCompile:
    def test_field_ptx(self, data_store: Path):
        r = _run_inspect("compile", "--data-dir", str(data_store), "--turn", "1", "--field", "ptx")
        assert r.returncode == 0, r.stderr
        data = json.loads(r.stdout)
        assert data["all_ok"] is True
        assert ".target sm_90" in data["ptx"]

    def test_field_resource_usage(self, data_store: Path):
        r = _run_inspect("compile", "--data-dir", str(data_store), "--turn", "1", "--field", "resource_usage")
        assert r.returncode == 0, r.stderr
        data = json.loads(r.stdout)
        assert "REG:32" in data["resource_usage"]

    def test_field_all(self, data_store: Path):
        r = _run_inspect("compile", "--data-dir", str(data_store), "--turn", "1", "--field", "all")
        assert r.returncode == 0, r.stderr
        data = json.loads(r.stdout)
        assert "artifacts" in data

    def test_missing_turn(self, data_store: Path):
        r = _run_inspect("compile", "--data-dir", str(data_store), "--turn", "99", "--field", "ptx")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "error" in data


class TestEvaluate:
    def test_all_configs(self, data_store: Path):
        r = _run_inspect("evaluate", "--data-dir", str(data_store), "--turn", "1")
        assert r.returncode == 0, r.stderr
        data = json.loads(r.stdout)
        assert len(data["configs"]) == 2

    def test_filter_config(self, data_store: Path):
        r = _run_inspect("evaluate", "--data-dir", str(data_store), "--turn", "1",
                         "--config", "vec1d-n65536")
        assert r.returncode == 0, r.stderr
        data = json.loads(r.stdout)
        assert len(data["configs"]) == 1
        assert "vec1d-n65536" in data["configs"]


class TestProfile:
    def test_all_configs(self, data_store: Path):
        r = _run_inspect("profile", "--data-dir", str(data_store), "--turn", "1")
        assert r.returncode == 0, r.stderr
        data = json.loads(r.stdout)
        assert data["configs"]["vec1d-n65536"]["summary"]["ncu_profiled"] is True


class TestRaw:
    def test_response_side(self, data_store: Path):
        r = _run_inspect("raw", "--data-dir", str(data_store), "--turn", "1",
                         "--stage", "compile", "--side", "response")
        assert r.returncode == 0, r.stderr
        data = json.loads(r.stdout)
        assert data["all_ok"] is True

    def test_request_side(self, data_store: Path):
        r = _run_inspect("raw", "--data-dir", str(data_store), "--turn", "1",
                         "--stage", "compile", "--side", "request")
        assert r.returncode == 0, r.stderr
        data = json.loads(r.stdout)
        assert "metadata" in data
