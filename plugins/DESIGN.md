# Plugin Design: KB & CUDA

Two Claude Code plugins for the kernel_lab project.

---

## KB Plugin (`knowledge-search`)

MCP Server: 1 (BM25/dense/hybrid search over indexed NVIDIA docs)

| Skill | User Intent | MCP Tools Used |
|-------|-------------|----------------|
| `search` | Query documentation | search_docs, lookup_doc_section, browse_toc, read_section |
| `ingest` | Add new documents to the index | parse + index pipeline |
| `rebuild` | Rebuild the entire search index | full reindex pipeline |
| `download` | Fetch latest raw docs from NVIDIA | downloader pipeline |

Invocation: `/kb:search`, `/kb:ingest`, `/kb:rebuild`, `/kb:download`

---

## CUDA Plugin (`cuda-toolkit-exec`)

MCP Server: 1 (proxies cuda_exec HTTP API + local data store)

| Skill | User Intent | MCP Tools Used |
|-------|-------------|----------------|
| `exec` | Compile, evaluate, profile a kernel | compile, evaluate, profile, execute, read_file |
| `inspect` | Review results from past runs | get_compile_data, get_evaluate_data, get_profile_data, get_data_point |
| `deploy` | Start, stop, health-check the cuda_exec service | shell commands (uvicorn lifecycle, /healthz) |

Invocation: `/cuda:exec`, `/cuda:inspect`, `/cuda:deploy`

---

## Design Principles

- **One skill = one distinct user intent.** Steps that always chain together (compile → evaluate) belong in the same skill. Actions a user initiates independently (run vs. inspect vs. deploy) are separate skills.
- **Skills orchestrate tools, not mirror them.** A skill may use many MCP tools; an MCP tool may be used by multiple skills. The mapping is not 1:1.
- **MCP tools are not user-facing.** Users invoke skills (`/plugin:skill`); Claude decides which MCP tools to call during execution.
