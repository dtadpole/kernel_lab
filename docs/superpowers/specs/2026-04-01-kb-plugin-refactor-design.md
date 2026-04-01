# KB Plugin Refactor: Drop MCP, CLI-Only

## Goal

Simplify the KB plugin from 4 skills + 1 MCP server to 2 skills + 0 MCP. Agent uses `python -m doc_retrieval` CLI subcommands via Bash.

## Skills

### `/kb:docs` — find and read documentation

Agent runs CLI subcommands:

| Subcommand | Usage | Purpose |
|------------|-------|---------|
| `find` | `python -m doc_retrieval find "query" [--mode hybrid] [--top-k 5]` | Search chunks (BM25/dense/hybrid) |
| `read` | `python -m doc_retrieval read <doc_id> <section_id>` | Read full section with nav context |
| `browse` | `python -m doc_retrieval browse <doc_id> [--section-id ID] [--depth N]` | Browse TOC tree |

Flow: `find` → `read` → `browse` (search → expand → explore nearby).

### `/kb:index` — manage the search index

Agent runs CLI subcommands:

| Command | Purpose |
|---------|---------|
| `python -m doc_retrieval download [--tier 1\|2\|3\|all]` | Fetch raw docs from NVIDIA |
| `python -m doc_retrieval parse` | Parse into chunks/TOC/sections |
| `python -m doc_retrieval index [--only bm25\|dense]` | Build BM25 + FAISS indices |
| `rm -rf ~/.doc_retrieval` | Nuke all derived artifacts |

Full rebuild: `download && parse && index`.

## Changes

### Delete
- `plugins/kb/mcp_server.py`
- `plugins/kb/.mcp.json`
- `plugins/kb/skills/search/SKILL.md`
- `plugins/kb/skills/download/SKILL.md`
- `plugins/kb/skills/rebuild/SKILL.md`
- `plugins/kb/skills/ingest/SKILL.md`

### Create
- `plugins/kb/skills/docs/SKILL.md`
- `plugins/kb/skills/index/SKILL.md`

### Modify
- `plugins/kb/.claude-plugin/plugin.json` — remove MCP description
- `doc_retrieval/cli.py` — rename `search` subcommand to `find`
- `doc_retrieval/searcher.py` — rename `cli_search` to `cli_find`
- `doc_retrieval/AGENTS.md` — update commands

### Dropped
- `lookup_doc_section` — URL-based retrieval removed. `find` results give doc_id + section info; `read` with doc_id+section_id covers the use case.

## Net result
- Before: 4 skills, 4 MCP tools, 1 MCP server, 7 files in plugins/kb/
- After: 2 skills, 0 MCP tools, 0 servers, 4 files in plugins/kb/
