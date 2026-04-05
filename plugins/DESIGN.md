# Plugin Design: `ik`

Single unified Claude Code plugin for kernel_lab. CLI-only — no MCP server.

---

## Skills

| Skill | User Intent | Used By | Implementation |
|-------|-------------|---------|----------------|
| `exec` | Compile, trial, profile a kernel (iterative dev) | Solver | Python CLI: calls `cuda_exec` handlers directly |
| `bench` | Formal benchmark (compile + trial ALL configs) | Judge | Python CLI: `cuda_exec.formal.formal_benchmark()` |
| `docs` | Search NVIDIA documentation | Any | Python CLI: `doc_retrieval` module (BM25) |
| `index` | Manage documentation search index | Rigger | Python CLI: `doc_retrieval` download/parse/index |
| `optimize` | Autonomous kernel optimization loop | Solver | Orchestration: calls exec, bench, docs |

Invocation: `/ik:exec`, `/ik:bench`, `/ik:docs`, `/ik:index`, `/ik:optimize`

---

## Naming

| Term | Meaning | Context |
|------|---------|---------|
| **compile** | Build CUDA binary from source | Both exec and bench |
| **trial** | Quick correctness + latency check (subset of configs OK) | exec (dev iteration) |
| **profile** | NCU hardware counter deep dive | exec only |
| **bench** | Formal benchmark: compile + trial ALL configs, comprehensive | Judge agent |

The Solver does many **trials**; the Judge runs one **bench**.

---

## Agent Roles

```
Solver  ──(ik:exec)──────→  compile, trial, profile (iterative dev)
        ──(ik:optimize)──→  autonomous optimization loop
        ──(ik:docs)──────→  search NVIDIA docs
           │
           │  "ready for bench"
           ▼
Supervisor ──→ Judge ──(ik:bench)──→ formal benchmark → verdict

Rigger  ──(ik:index)─────→  download, parse, build search index
```

---

## Design Principles

- **CLI-only.** Skills describe bash/Python commands for Claude to run. No MCP server overhead.
- **One skill = one distinct user intent.** exec (dev tools) vs bench (formal assessment) serve different agents with different trust models.
- **Local execution.** All CUDA work runs on local GPUs via `CUDA_VISIBLE_DEVICES`. No remote dispatch.
- **Skills compose.** `optimize` calls `exec` and `bench`. `bench` chains compile + trial atomically.

---

## Deprecated

Old plugins (`plugins/deprecated/cuda/`, `plugins/deprecated/kb/`) had MCP servers, remote dispatch, and dense embedding search. Replaced by `ik` for simplicity.
