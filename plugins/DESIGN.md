# Plugin Design: `ik`

Single unified Claude Code plugin for kernel_lab. CLI-only — no MCP server.

---

## Skills

| Skill | User Intent | Implementation |
|-------|-------------|----------------|
| `exec` | Compile, evaluate, profile a kernel locally | Python CLI: calls `cuda_exec` handlers directly |
| `inspect` | Review results from past runs | File reads from `~/.cuda_exec/` data store |
| `docs` | Search NVIDIA documentation | Python CLI: `doc_retrieval` module |
| `index` | Manage documentation search index | Python CLI: `doc_retrieval` download/parse/index |
| `optimize` | Autonomous kernel optimization loop | Orchestration: calls exec, inspect, docs |

Invocation: `/ik:exec`, `/ik:inspect`, `/ik:docs`, `/ik:index`, `/ik:optimize`

---

## Design Principles

- **CLI-only.** Skills describe bash/Python commands for Claude to run. No MCP server overhead.
- **One skill = one distinct user intent.** Steps that always chain together (compile → evaluate) belong in the same skill. Actions a user initiates independently (run vs. inspect) are separate skills.
- **Local execution.** All CUDA work runs on local GPUs via `CUDA_VISIBLE_DEVICES`. No remote dispatch.
- **Skills orchestrate, not mirror.** `optimize` calls `exec`, `inspect`, and `docs` — skills compose.

---

## Deprecated

Old plugins (`plugins/deprecated/cuda/`, `plugins/deprecated/kb/`) had MCP servers and remote dispatch. Replaced by `ik` for simplicity.
