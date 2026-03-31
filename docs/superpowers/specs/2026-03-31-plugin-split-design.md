# Plugin Split Design: Knowledge Search + CUDA Toolkit Execution Service

**Date:** 2026-03-31
**Status:** Approved

## Summary

Split the monolithic `cuda_agent/mcp_server.py` (37KB, 13 tools) into two independent Claude Code plugins, each with its own MCP server and Skill. Both plugins work in Claude Code CLI (dev) and Agent SDK (runtime).

## Directory Structure

```
plugins/
├── knowledge-search/
│   ├── .claude-plugin/plugin.json
│   ├── skills/search/SKILL.md
│   ├── .mcp.json
│   ├── mcp_server.py
│   └── requirements.txt
│
└── cuda-toolkit-exec/
    ├── .claude-plugin/plugin.json
    ├── skills/exec/SKILL.md
    ├── .mcp.json
    ├── mcp_server.py
    └── requirements.txt
```

## Tool Distribution

### Knowledge Search MCP Server (4 tools)

| Tool | Source | Description |
|------|--------|-------------|
| `search_docs` | `cuda_search_docs` | BM25/dense/hybrid search |
| `lookup_doc_section` | `cuda_lookup_doc_section` | Full section by chunk_id |
| `browse_toc` | `cuda_browse_toc` | Browse doc table of contents |
| `read_section` | `cuda_read_section` | Read section with nav context |

Depends on: `doc_retrieval` package (in-process import).

### CUDA Toolkit Execution Service MCP Server (9 tools)

| Tool | Source | Description |
|------|--------|-------------|
| `compile` | `cuda_compile` | Compile CUDA source |
| `evaluate` | `cuda_evaluate` | Correctness + performance |
| `profile` | `cuda_profile` | NCU profiling |
| `execute` | `cuda_execute` | Ad-hoc command execution |
| `read_file` | `cuda_read_file` | On-demand file reading |
| `get_compile_data` | `cuda_get_compile_data` | Structured compile results |
| `get_evaluate_data` | `cuda_get_evaluate_data` | Structured evaluate results |
| `get_profile_data` | `cuda_get_profile_data` | Structured profile results |
| `get_data_point` | `cuda_get_data_point` | Raw request/response fallback |

Depends on: `cuda_exec` HTTP API (remote calls via httpx).

## Architecture

- Each plugin is a standalone stdio MCP server using FastMCP
- Claude Code loads plugins via `.mcp.json` configuration
- Agent SDK loads via `mcp_servers={}` in `ClaudeAgentOptions`
- `cuda_` prefix dropped from tool names — plugin namespace provides disambiguation

## cuda_agent Integration

`cuda_agent/agent.py` loads two MCP servers instead of one:

```python
mcp_servers={
    "knowledge-search": {
        "command": "python",
        "args": [str(repo_root / "plugins/knowledge-search/mcp_server.py")],
        "env": {/* doc_retrieval config */}
    },
    "cuda-toolkit-exec": {
        "command": "python",
        "args": [str(repo_root / "plugins/cuda-toolkit-exec/mcp_server.py")],
        "env": {/* cuda_exec URL, key, data dir */}
    }
}
```

## Deletions

- `cuda_agent/mcp_server.py` — replaced by two plugin MCP servers

## Future Work

- MCP interface refactoring (tool signatures, response format)
- Plugin versioning and distribution
