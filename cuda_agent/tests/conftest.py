"""Shared test fixtures for cuda_agent tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def data_store(tmp_path: Path) -> Path:
    """Create a sample data store with compile/evaluate/profile responses."""
    turn_dir = tmp_path / "turn_1"
    turn_dir.mkdir()

    compile_resp = {
        "all_ok": True,
        "attempt": 1,
        "artifacts": {
            "ptx": {"content": ".version 8.0\n.target sm_90\n.entry kernel() {}"},
            "sass": {"content": "MOV R0, R1;\nIADD R2, R0, R1;"},
            "resource_usage": {"content": "REG:32 SMEM:1024 STACK:0"},
            "binary": {"encoding": "base64", "content": "AQID..."},
        },
        "tool_outputs": {
            "nvcc": {"content": "nvcc: success"},
            "ptxas": {"content": "ptxas info: 32 registers"},
        },
    }
    (turn_dir / "compile.attempt_001.response.json").write_text(
        json.dumps(compile_resp, indent=2)
    )
    (turn_dir / "compile.attempt_001.request.json").write_text(
        json.dumps({"metadata": {"turn": 1}}, indent=2)
    )

    eval_resp = {
        "all_ok": True,
        "configs": {
            "vec1d-n65536": {
                "status": "ok",
                "correctness": {"passed": True, "max_abs_error": 0.0},
                "performance": {"latency_ms": {"median": 0.05}},
            },
            "tensor2d-1024x1024": {
                "status": "ok",
                "correctness": {"passed": True, "max_abs_error": 1e-6},
                "performance": {"latency_ms": {"median": 0.12}},
            },
        },
    }
    (turn_dir / "evaluate.attempt_001.response.json").write_text(
        json.dumps(eval_resp, indent=2)
    )

    profile_resp = {
        "all_ok": True,
        "configs": {
            "vec1d-n65536": {
                "status": "ok",
                "summary": {"side": "generated", "ncu_profiled": True},
            },
        },
    }
    (turn_dir / "profile.attempt_001.response.json").write_text(
        json.dumps(profile_resp, indent=2)
    )

    return tmp_path
