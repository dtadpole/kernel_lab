"""Exec tool-level tests: individual MCP tool calls."""

import importlib.util
import json
from pathlib import Path

import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cuda_mcp():
    """Import the cuda MCP server module."""
    spec = importlib.util.spec_from_file_location(
        "cuda_mcp", _PLUGIN_DIR / "mcp_server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Integration tests — require remote cuda_exec service
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_compile_vecadd(cuda_mcp, sample_metadata, vecadd_fixtures):
    """compile tool returns structured response."""
    result = await cuda_mcp.compile(
        metadata=sample_metadata,
        reference_files={"cutedsl.py": vecadd_fixtures["reference"]},
        generated_files={"generated.cu": vecadd_fixtures["generated"]},
    )
    data = json.loads(result)
    assert "all_ok" in data
    assert "artifacts" in data
    assert "tool_outputs" in data


@pytest.mark.integration
@pytest.mark.asyncio
async def test_compile_has_ptx(cuda_mcp, sample_metadata, vecadd_fixtures):
    """compile produces PTX output."""
    result = await cuda_mcp.compile(
        metadata=sample_metadata,
        reference_files={"cutedsl.py": vecadd_fixtures["reference"]},
        generated_files={"generated.cu": vecadd_fixtures["generated"]},
    )
    data = json.loads(result)
    ptx = data.get("artifacts", {}).get("ptx", {})
    assert ptx.get("path") is not None, "PTX artifact should have a path"
