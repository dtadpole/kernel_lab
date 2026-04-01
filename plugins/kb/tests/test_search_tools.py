"""Search tool-level tests: individual MCP tool calls."""

import importlib.util
import json
from pathlib import Path

import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def kb_mcp():
    """Import the kb MCP server module."""
    spec = importlib.util.spec_from_file_location(
        "kb_mcp", _PLUGIN_DIR / "mcp_server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_docs_returns_results(kb_mcp):
    """search_docs returns a list of results with expected fields."""
    result = await kb_mcp.search_docs(query="shared memory bank conflicts")
    data = json.loads(result)
    assert isinstance(data, list)
    if data:  # index may not be built
        assert "title" in data[0]
        assert "score" in data[0]
        assert "url" in data[0]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_browse_toc_returns_structure(kb_mcp):
    """browse_toc returns TOC structure for a known doc."""
    result = await kb_mcp.browse_toc(doc_id="cuda-c-programming-guide")
    data = json.loads(result)
    assert isinstance(data, (dict, list))
