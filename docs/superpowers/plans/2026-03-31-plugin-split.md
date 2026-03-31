# Plugin Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `cuda_agent/mcp_server.py` into two independent Claude Code plugins (knowledge-search and cuda-toolkit-exec), each with its own MCP server and Skill.

**Architecture:** Each plugin is a self-contained directory under `plugins/` with a `.claude-plugin/plugin.json` manifest, `.mcp.json` for MCP server config, a FastMCP stdio `mcp_server.py`, and a `skills/` directory with workflow guidance. `cuda_agent/agent.py` loads both plugins as separate MCP servers.

**Tech Stack:** Python 3.12, FastMCP, httpx, doc_retrieval (in-process), cuda_exec (HTTP API)

---

### Task 1: Create knowledge-search plugin scaffold

**Files:**
- Create: `plugins/knowledge-search/.claude-plugin/plugin.json`
- Create: `plugins/knowledge-search/.mcp.json`
- Create: `plugins/knowledge-search/requirements.txt`

- [ ] **Step 1: Create plugin.json manifest**

```json
{
  "name": "knowledge-search",
  "description": "CUDA documentation search and retrieval via BM25/dense/hybrid search over indexed NVIDIA docs",
  "version": "0.1.0",
  "author": {
    "name": "D.Tadpole"
  }
}
```

Write to `plugins/knowledge-search/.claude-plugin/plugin.json`.

- [ ] **Step 2: Create .mcp.json**

```json
{
  "mcpServers": {
    "knowledge-search": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "${pluginDir}"
    }
  }
}
```

Write to `plugins/knowledge-search/.mcp.json`.

- [ ] **Step 3: Create requirements.txt**

```
mcp>=1.20,<2.0
```

Write to `plugins/knowledge-search/requirements.txt`.

- [ ] **Step 4: Commit**

```bash
git add plugins/knowledge-search/.claude-plugin/plugin.json plugins/knowledge-search/.mcp.json plugins/knowledge-search/requirements.txt
git commit -m "feat: scaffold knowledge-search plugin"
```

---

### Task 2: Create knowledge-search MCP server

**Files:**
- Create: `plugins/knowledge-search/mcp_server.py`

- [ ] **Step 1: Write mcp_server.py**

Extract the 4 doc retrieval tools from `cuda_agent/mcp_server.py` (lines 720-947) into a standalone FastMCP server. Key changes:
- Drop `cuda_` prefix from tool names → `search_docs`, `lookup_doc_section`, `browse_toc`, `read_section`
- Keep the lazy-loaded `_doc_searcher` singleton pattern
- Keep the `doc_retrieval.searcher.DocSearcher` import
- Server name: `"knowledge-search"`
- Env vars: reads `DOC_RETRIEVAL_ROOT` and `OPENAI_API_KEY` from environment
- Needs repo root on `PYTHONPATH` to import `doc_retrieval`

Full content of `plugins/knowledge-search/mcp_server.py`:

```python
"""FastMCP stdio server for CUDA documentation search and retrieval.

Provides 4 tools for searching and browsing indexed NVIDIA CUDA Toolkit
documentation using BM25, dense (embedding), or hybrid retrieval.

Configuration (environment variables):
    DOC_RETRIEVAL_ROOT    Storage root for indices/chunks.
                          Default: ~/.doc_retrieval
    OPENAI_API_KEY        API key for embedding model (dense/hybrid search).
"""

from __future__ import annotations

import json
import sys
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

# ---------------------------------------------------------------------------
# Lazy-loaded searcher
# ---------------------------------------------------------------------------

_doc_searcher = None


def _get_doc_searcher():
    """Lazy-load the document searcher singleton."""
    global _doc_searcher
    if _doc_searcher is None:
        try:
            from doc_retrieval.searcher import DocSearcher

            _doc_searcher = DocSearcher()
        except Exception as exc:
            return None, str(exc)
    return _doc_searcher, None


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("knowledge-search")


@mcp.tool()
async def search_docs(
    query: Annotated[
        str,
        Field(description="Natural language search query about CUDA programming"),
    ],
    mode: Annotated[
        Literal["bm25", "dense", "hybrid"],
        Field(description="Search mode: bm25 (keyword), dense (semantic), hybrid (combined)"),
    ] = "hybrid",
    top_k: Annotated[
        int,
        Field(description="Number of results to return (1-20)", ge=1, le=20),
    ] = 5,
) -> str:
    """Search NVIDIA CUDA Toolkit documentation.

    Searches the indexed CUDA documentation corpus using the specified
    retrieval mode.  Returns matching chunks with section metadata.

    Example queries:
        - "shared memory bank conflicts"
        - "warp divergence impact on performance"
        - "atomicCAS signature and semantics"
        - "cudaMemcpyAsync stream synchronization"
        - "PTX ld.global instruction"

    Each result includes a ``url`` field with an anchor (e.g.
    ``...index.html#thread-hierarchy``).  Extract the doc slug and
    anchor to follow up:

        result = search_docs("shared memory bank conflicts")
        # Found section_path "Performance Guidelines > Device Memory Accesses"

        # Read the full section for more context:
        read_section("cuda-c-programming-guide", "device-memory-accesses")

        # Or browse the parent chapter's TOC:
        browse_toc("cuda-c-programming-guide", "performance-guidelines")
    """
    searcher, err = _get_doc_searcher()
    if searcher is None:
        return json.dumps({"error": f"Doc searcher unavailable: {err}"})

    if mode == "bm25":
        results = searcher.search_bm25(query, top_k)
    elif mode == "dense":
        results = searcher.search_dense(query, top_k)
    else:
        results = searcher.search_hybrid(query, top_k)

    return json.dumps(
        [
            {
                "title": r.title,
                "section_path": r.section_path,
                "url": r.source_url,
                "text": r.text[:2000],
                "score": round(r.score, 4),
            }
            for r in results
        ],
        indent=2,
    )


@mcp.tool()
async def lookup_doc_section(
    url: Annotated[
        str,
        Field(description="URL of the CUDA documentation page (from a prior search result)"),
    ],
    section: Annotated[
        str | None,
        Field(description="Section heading to filter to (optional)"),
    ] = None,
) -> str:
    """Retrieve full text from a specific CUDA documentation page or section.

    Given a URL from a prior search_docs result, retrieves all
    chunks from that page.  Optionally filter to a specific section.

    Prefer ``read_section`` for HTML docs — it returns structured
    content with navigation context (parent, siblings).  Use this tool
    when you have a URL but not a doc_id/section_id, or for PDF docs
    which don't have anchor-based navigation.
    """
    searcher, err = _get_doc_searcher()
    if searcher is None:
        return json.dumps({"error": f"Doc searcher unavailable: {err}"})

    chunks = searcher._load_chunks()
    matching = [c for c in chunks if c["source_url"] == url]

    if section:
        section_lower = section.lower()
        matching = [
            c for c in matching
            if section_lower in c["section_path"].lower()
        ]

    if not matching:
        return json.dumps({"error": f"No chunks found for url={url}, section={section}"})

    matching.sort(key=lambda c: c["chunk_index"])
    text = "\n\n".join(c["text"] for c in matching)

    if len(text) > 8000:
        text = text[:8000] + "\n\n[... truncated ...]"

    return json.dumps(
        {
            "url": url,
            "section": section,
            "num_chunks": len(matching),
            "text": text,
        },
        indent=2,
    )


@mcp.tool()
async def browse_toc(
    doc_id: Annotated[str, Field(description=(
        "Document slug, e.g. 'cuda-c-programming-guide', 'parallel-thread-execution'"
    ))],
    section_id: Annotated[str | None, Field(description=(
        "Section anchor ID to expand. Omit for top-level chapters."
    ))] = None,
    depth: Annotated[int, Field(description="Expansion depth", ge=1, le=5)] = 2,
) -> str:
    """Browse the table of contents of a CUDA documentation page.

    Available doc_ids:
        cuda-c-programming-guide, parallel-thread-execution,
        cuda-c-best-practices-guide, inline-ptx-assembly,
        blackwell-tuning-guide, blackwell-compatibility-guide

    Usage patterns:

        # List top-level chapters:
        browse_toc("cuda-c-programming-guide")

        # Expand a chapter to see its sub-sections:
        browse_toc("cuda-c-programming-guide", "performance-guidelines")

        # Deep expand (3 levels):
        browse_toc("cuda-c-programming-guide", "programming-model", depth=3)

        # Then read a specific section:
        read_section("cuda-c-programming-guide", "device-memory-accesses")
    """
    searcher, err = _get_doc_searcher()
    if err:
        return err
    result = searcher.browse_toc(doc_id=doc_id, section_id=section_id, depth=depth)
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
async def read_section(
    doc_id: Annotated[str, Field(description=(
        "Document slug, e.g. 'cuda-c-programming-guide', 'parallel-thread-execution'"
    ))],
    section_id: Annotated[str, Field(description=(
        "Section anchor ID from TOC or search result, e.g. 'thread-hierarchy'"
    ))],
) -> str:
    """Read the full content of a specific documentation section.

    Returns the section content as lightweight HTML with navigation
    context.  The response includes a ``nav`` object with
    ``parent``, ``prev_sibling``, and ``next_sibling`` section IDs
    for continued browsing.

    Usage patterns:

        # After searching — expand a result to full section:
        results = search_docs("shared memory bank conflicts")
        # url = ".../cuda-c-programming-guide/index.html#shared-memory"
        section = read_section("cuda-c-programming-guide", "shared-memory")

        # Read the next section using nav context:
        read_section("cuda-c-programming-guide", section.nav.next_sibling)

        # Go up to the parent chapter:
        browse_toc("cuda-c-programming-guide", section.nav.parent)

        # After browsing TOC — read a section you picked:
        toc = browse_toc("parallel-thread-execution", "instruction-set")
        read_section("parallel-thread-execution", "data-movement-and-conversion-instructions-ld")
    """
    searcher, err = _get_doc_searcher()
    if err:
        return err
    result = searcher.read_section(doc_id=doc_id, section_id=section_id)
    if result is None:
        return json.dumps({"error": f"Section '{section_id}' not found in '{doc_id}'"})
    if len(result.get("content", "")) > 8000:
        result["content"] = result["content"][:8000] + "\n... [truncated]"
    return json.dumps(result, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

- [ ] **Step 2: Commit**

```bash
git add plugins/knowledge-search/mcp_server.py
git commit -m "feat: knowledge-search MCP server with 4 doc retrieval tools"
```

---

### Task 3: Create knowledge-search Skill

**Files:**
- Create: `plugins/knowledge-search/skills/search/SKILL.md`

- [ ] **Step 1: Write SKILL.md**

```markdown
---
name: search
description: Search NVIDIA CUDA Toolkit documentation for programming concepts, API references, and optimization techniques
user-invocable: true
argument-hint: <query>
---

# CUDA Documentation Search

Search indexed NVIDIA CUDA Toolkit documentation using the knowledge-search MCP tools.

## Available Tools

- **search_docs** — BM25/dense/hybrid search over all indexed docs
- **lookup_doc_section** — Retrieve full text from a URL (from search results)
- **browse_toc** — Browse document table of contents by doc_id
- **read_section** — Read a specific section with navigation context

## Workflow

1. Start with `search_docs` using a natural language query: `$ARGUMENTS`
2. Review the results — each has a `url` and `section_path`
3. To read more context, use `read_section` with the doc_id and section anchor
4. To explore nearby sections, use `browse_toc` to see the TOC structure
5. Navigate using `nav.parent`, `nav.prev_sibling`, `nav.next_sibling` from read_section results

## Available Documents

- `cuda-c-programming-guide` — CUDA C++ Programming Guide
- `parallel-thread-execution` — PTX ISA Reference
- `cuda-c-best-practices-guide` — CUDA C++ Best Practices Guide
- `inline-ptx-assembly` — Inline PTX Assembly in CUDA
- `blackwell-tuning-guide` — Blackwell Tuning Guide
- `blackwell-compatibility-guide` — Blackwell Compatibility Guide
```

- [ ] **Step 2: Commit**

```bash
git add plugins/knowledge-search/skills/search/SKILL.md
git commit -m "feat: knowledge-search skill with doc search workflow"
```

---

### Task 4: Create cuda-toolkit-exec plugin scaffold

**Files:**
- Create: `plugins/cuda-toolkit-exec/.claude-plugin/plugin.json`
- Create: `plugins/cuda-toolkit-exec/.mcp.json`
- Create: `plugins/cuda-toolkit-exec/requirements.txt`

- [ ] **Step 1: Create plugin.json manifest**

```json
{
  "name": "cuda-toolkit-exec",
  "description": "Remote CUDA kernel compilation, evaluation, profiling, and execution via the cuda_exec service",
  "version": "0.1.0",
  "author": {
    "name": "D.Tadpole"
  }
}
```

Write to `plugins/cuda-toolkit-exec/.claude-plugin/plugin.json`.

- [ ] **Step 2: Create .mcp.json**

```json
{
  "mcpServers": {
    "cuda-toolkit-exec": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "${pluginDir}"
    }
  }
}
```

Write to `plugins/cuda-toolkit-exec/.mcp.json`.

- [ ] **Step 3: Create requirements.txt**

```
mcp>=1.20,<2.0
httpx>=0.27,<1.0
```

Write to `plugins/cuda-toolkit-exec/requirements.txt`.

- [ ] **Step 4: Commit**

```bash
git add plugins/cuda-toolkit-exec/.claude-plugin/plugin.json plugins/cuda-toolkit-exec/.mcp.json plugins/cuda-toolkit-exec/requirements.txt
git commit -m "feat: scaffold cuda-toolkit-exec plugin"
```

---

### Task 5: Create cuda-toolkit-exec MCP server

**Files:**
- Create: `plugins/cuda-toolkit-exec/mcp_server.py`

- [ ] **Step 1: Write mcp_server.py**

Extract the 9 execution tools from `cuda_agent/mcp_server.py` (lines 1-718) into a standalone FastMCP server. Key changes:
- Drop `cuda_` prefix from tool names → `compile`, `evaluate`, `profile`, `execute`, `read_file`, `get_compile_data`, `get_evaluate_data`, `get_profile_data`, `get_data_point`
- Keep all helpers: `_load_bearer_token`, `_save_data_point`, `_list_available`, `_compact_response`, `_post`, `_read_stored_response`, `_not_found_error`, `_data_store_not_configured`
- Keep all config env vars: `CUDA_EXEC_URL`, `CUDA_EXEC_KEY_PATH`, `CUDA_AGENT_MCP_*`, `CUDA_AGENT_DATA_DIR`
- Server name: `"cuda-toolkit-exec"`

Full content: take lines 1-718 from `cuda_agent/mcp_server.py`, rename all `cuda_compile` → `compile`, `cuda_evaluate` → `evaluate`, etc., update internal docstring references, change `FastMCP("cuda_toolkit")` → `FastMCP("cuda-toolkit-exec")`.

- [ ] **Step 2: Commit**

```bash
git add plugins/cuda-toolkit-exec/mcp_server.py
git commit -m "feat: cuda-toolkit-exec MCP server with 9 execution tools"
```

---

### Task 6: Create cuda-toolkit-exec Skill

**Files:**
- Create: `plugins/cuda-toolkit-exec/skills/exec/SKILL.md`

- [ ] **Step 1: Write SKILL.md**

```markdown
---
name: exec
description: Compile, evaluate, and profile CUDA kernels using the remote cuda_exec service
user-invocable: true
argument-hint: <action> [options]
---

# CUDA Toolkit Execution Service

Compile, evaluate, and profile CUDA kernels using the cuda-toolkit-exec MCP tools.

## Available Tools

### Action Tools (proxy to cuda_exec HTTP API)
- **compile** — Compile CUDA source to binary/PTX/SASS
- **evaluate** — Correctness + performance testing against configs
- **profile** — NCU profiling (generated or reference side)
- **execute** — Ad-hoc command execution
- **read_file** — On-demand file reading from turn directories

### Data Retrieval Tools (read from local data store)
- **get_compile_data** — Structured compile results (ptx, sass, resource_usage, tool_outputs)
- **get_evaluate_data** — Structured correctness/performance with config filtering
- **get_profile_data** — NCU summary with config filtering
- **get_data_point** — Raw uncompacted request/response fallback

## Workflow

1. **Compile first**: Call `compile` with metadata, reference_files, and generated_files
2. **Evaluate**: Call `evaluate` with the same metadata and configs to test correctness + performance
3. **Profile** (optional): Call `profile` to get NCU hardware metrics
4. **Iterate**: Modify source code → increment `metadata.turn` → compile again

## Workflow Rules

- Compile exactly once per turn before evaluate or profile
- New source code requires a new turn (increment metadata.turn)
- Old turns are immutable — never recompile on a previous turn number
- One compile fans out to many evaluate/profile calls with different configs

## Metadata Format

Every tool requires a `metadata` dict:
```json
{
  "run_tag": "optim_001",
  "version": "v1",
  "direction_id": 7,
  "direction_slug": "vector-add",
  "turn": 1
}
```
```

- [ ] **Step 2: Commit**

```bash
git add plugins/cuda-toolkit-exec/skills/exec/SKILL.md
git commit -m "feat: cuda-toolkit-exec skill with workflow guidance"
```

---

### Task 7: Update cuda_agent/agent.py to load two plugins

**Files:**
- Modify: `cuda_agent/agent.py:34` (remove `_MCP_SERVER_SCRIPT`)
- Modify: `cuda_agent/agent.py:159-206` (replace single MCP server with two)

- [ ] **Step 1: Update agent.py**

Replace the single MCP server configuration with two separate ones. Key changes:

1. Remove line 34: `_MCP_SERVER_SCRIPT = str(Path(__file__).resolve().parent / "mcp_server.py")`
2. Add plugin paths:
```python
_REPO_ROOT = Path(__file__).resolve().parents[1]
_KNOWLEDGE_SEARCH_SERVER = str(_REPO_ROOT / "plugins" / "knowledge-search" / "mcp_server.py")
_CUDA_TOOLKIT_EXEC_SERVER = str(_REPO_ROOT / "plugins" / "cuda-toolkit-exec" / "mcp_server.py")
```

3. In `run_optimization()`, split `mcp_env` into two env dicts and create two MCP server entries:

```python
# Shared env
base_env = {
    "HOME": os.environ.get("HOME", str(Path.home())),
}

# Knowledge search env
ks_env = {
    **base_env,
    "PYTHONPATH": f"{repo_root}:{existing_path}" if existing_path else repo_root,
}
if doc_root:
    ks_env["DOC_RETRIEVAL_ROOT"] = doc_root
if openai_key:
    ks_env["OPENAI_API_KEY"] = openai_key

# CUDA toolkit exec env
exec_env = {
    **base_env,
    "CUDA_EXEC_URL": task.cuda_exec_url or cfg.service.cuda_exec_url,
    "CUDA_AGENT_MCP_REQUEST_TIMEOUT": str(cfg.mcp.request_timeout),
    "CUDA_AGENT_MCP_CONNECT_TIMEOUT": str(cfg.mcp.connect_timeout),
    "CUDA_AGENT_MCP_MAX_CONTENT_CHARS": str(cfg.mcp.max_content_chars),
    "CUDA_AGENT_MCP_TOOL_TIMEOUT": str(cfg.mcp.tool_timeout_seconds),
    "CUDA_AGENT_DATA_DIR": str(log_dir),
}
if key_path:
    exec_env["CUDA_EXEC_KEY_PATH"] = str(Path(key_path).expanduser())

options = ClaudeAgentOptions(
    ...
    mcp_servers={
        "knowledge-search": {
            "command": sys.executable,
            "args": [_KNOWLEDGE_SEARCH_SERVER],
            "env": ks_env,
        },
        "cuda-toolkit-exec": {
            "command": sys.executable,
            "args": [_CUDA_TOOLKIT_EXEC_SERVER],
            "env": exec_env,
        },
    },
)
```

- [ ] **Step 2: Update hook deny message to reference new tool names**

In `_deny_direct_cuda_toolkit`, update the message from "cuda_compile, cuda_evaluate..." to "compile, evaluate, profile, execute (via cuda-toolkit-exec MCP)".

- [ ] **Step 3: Commit**

```bash
git add cuda_agent/agent.py
git commit -m "refactor: agent loads two plugin MCP servers instead of monolith"
```

---

### Task 8: Delete old mcp_server.py

**Files:**
- Delete: `cuda_agent/mcp_server.py`

- [ ] **Step 1: Delete the file**

```bash
git rm cuda_agent/mcp_server.py
```

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: remove monolithic mcp_server.py, replaced by plugins"
```

---

### Task 9: Smoke test — knowledge-search MCP server

- [ ] **Step 1: Test that the server starts and lists tools**

```bash
cd /home/centos/kernel_lab
PYTHONPATH="$PWD" python -c "
import subprocess, json
proc = subprocess.Popen(
    ['python', 'plugins/knowledge-search/mcp_server.py'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    env={**__import__('os').environ, 'PYTHONPATH': '$PWD'}
)
# Send MCP initialize + tools/list
proc.stdin.write(b'...')  # MCP protocol init
proc.stdin.flush()
"
```

Alternatively, run the server directly to check for import errors:

```bash
cd /home/centos/kernel_lab
PYTHONPATH="$PWD" timeout 3 python plugins/knowledge-search/mcp_server.py 2>&1 || true
```

Expected: no import errors, server starts and waits for stdio input.

- [ ] **Step 2: Test search_docs tool via MCP client**

Use the `mcp` CLI or a Python test script to call `search_docs` with a sample query and verify results come back.

---

### Task 10: Smoke test — cuda-toolkit-exec MCP server

- [ ] **Step 1: Test that the server starts**

```bash
cd /home/centos/kernel_lab
timeout 3 python plugins/cuda-toolkit-exec/mcp_server.py 2>&1 || true
```

Expected: server starts (may warn about missing key file if `~/.keys/cuda_exec.key` doesn't exist, but should not crash on import).

- [ ] **Step 2: Test tool listing**

Verify all 9 tools are registered by importing the module and checking `mcp.list_tools()`.
