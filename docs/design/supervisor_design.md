# Workshop — Kernel Optimization System Design

> Note: The Python process managing this system is `agents/workshop.py`
> (renamed from `supervisor.py` to avoid collision with Claude Agent SDK's
> "supervisor" concept).

## Architecture Overview

```
Supervisor (read-only, online orchestration)
│
├── Solver              Generate/modify kernel code (raw CUDA/PTX)
│   ├── Plan → skeleton → implement step by step
│   ├── request_formal_bench → Supervisor → Benchmarker
│   ├── ask_supervisor → Supervisor → Steward
│   └── Security: path blacklist/whitelist, no git, no CUTLASS
│
├── Benchmarker         Independent, objective evaluation
│   ├── Compilation check
│   ├── Correctness verification (✓/✗ vs golden reference)
│   └── Full benchmark — results go directly to Supervisor
│
├── Steward             Guidance agent (called at key decision points)
│   ├── session_end — review Solver work (SUCCESS/CONTINUE/ABORT)
│   ├── stuck — analyze why Solver is idle (30 min timeout)
│   └── answer_question — respond to Solver's questions
│
├── Liveness + Strategy  Unified in Runner main loop (no separate task)
│   ├── Liveness (every 10s): heartbeat_timeout 5min (LLM), tool_timeout 20min (tool/rate_limit)
│   ├── Strategy (every 60s): hard_limit, idle_timeout, loop_detection, progress_check
│   └── All subprocess creation via asyncio (zero threads)
│
└── Harness             Executable Skills
    ├── ik:exec — compile, trial, profile
    ├── ik:docs — CUDA documentation search
    └── ik:bench — formal benchmark (Benchmarker only)
```

## Key Decisions

| Decision | Conclusion | Rationale |
|----------|-----------|-----------|
| Benchmarker independent or Solver tool | Must be independent | Solver cannot be both player and referee |
| Validator | Removed | Benchmarker handles this — no need for separate component |
| Supervisor role | Read-only | Does not write code or modify files — only orchestrates |
| Steward usage | Key decisions only | progress_check and time_limit handled in code to save cost |
| Correctness vs performance | Correctness first | ✗ in bench = STOP optimizing, fix correctness |
| Code constraints | Raw CUDA/PTX only | CUTLASS, cuDNN, cuBLAS forbidden — write from scratch |
| Historical code access | Blocked | No access to other runs, gems, peak, or git history |
| KB format | Markdown Wiki | Natively readable/writable by LLMs |
| KB search | BM25 | Sufficient for now — add dense retrieval when semantic gap appears |

## Agent Configuration

| Agent | Model | Budget | Max Turns | Purpose |
|-------|-------|--------|-----------|---------|
| Solver | claude-sonnet-4-6 | $50 | 2000 | Write and optimize kernel code |
| Benchmarker | claude-sonnet-4-6 | $5 | 20 | Run formal benchmark |
| Steward | claude-sonnet-4-6 | $5 | 5-20 | Review sessions, answer questions |
| Rigger | claude-sonnet-4-6 | $10 | 500 | Modify harness infrastructure |

## Wave / Session Lifecycle

**Three layers:**

| Layer | Definition | Lifetime |
|-------|-----------|----------|
| **Task** | One optimization job | Forever (until external kill) |
| **Wave** | One subprocess = one PID | start() → stop() |
| **Session** | One dialogue round | client.query() → ResultMessage |

```
Task (run_continuous — runs forever)
│
├── Wave 0 (PID 111) — fresh AgentRunner
│   ├── Session 0: initial prompt → end_turn → Steward: CONTINUE
│   ├── Session 1: continue prompt → end_turn → Steward: SUCCESS → kill
│   └── Wave ends
│
├── Wave 1 (PID 222) — fresh AgentRunner
│   ├── Session 0: initial prompt → end_turn → Steward: CONTINUE
│   └── Session 1: continue prompt → exception → kill
│   └── Wave ends
│
└── ...forever
```

**Rules:**
- Each Wave = new AgentRunner, new subprocess, complete system + user prompt
- Within a Wave: `client.query(continue_prompt)` to same subprocess (new Session)
- Wave ends on: SUCCESS, ABORT, exception, or liveness timeout
- After Wave ends → always start next Wave
- Gem produced + reflection received → SUCCESS → sleep 10s → kill → next Wave

## Security — Path Blacklist/Whitelist

### Blacklist (all blocked, per tool: Read, Bash, Glob, Grep)

| Path | Purpose |
|------|---------|
| `~/kernel_lab/` | Entire repo |
| `~/kernel_lab_kb/` | Entire KB |
| `~/.cuda_exec/` | All scratch directories |
| `~/.cuda_exec_bench/` | Benchmarker staging |
| `~/.claude/worktrees/` | Old worktree copies |
| `~/.claude/projects/` | Claude memory files |

### Whitelist (exceptions to blacklist)

| Path | Purpose |
|------|---------|
| `~/kernel_lab/cuda_exec/` | Harness code |
| `~/kernel_lab/plugins/` | Skill definitions |
| `~/kernel_lab/conf/` | Configuration |
| `~/kernel_lab/data/ref/` | Reference implementations (read-only) |
| `~/kernel_lab/data/configs/` | Benchmark configs (read-only) |
| `~/kernel_lab/.venv/` | Python interpreter (Bash only) |
| `~/kernel_lab/DESIGN.md` | Design document |
| `~/kernel_lab/AGENTS.md` | Agent specification |
| `~/kernel_lab_kb/runs/<run_tag>/` | Current run only (dynamic) |
| `~/.cuda_exec/<run_tag>/` | Current run scratch (dynamic) |

### Enforcement

- **Read/Glob/Grep**: `file_path`/`path` checked against blocked/allowed
- **Bash**: all commands scanned for blocked paths; navigation (`ls`) allowed
  but not recursive (`ls -R`); `find` blocked on protected paths
- **Forbidden commands**: `git`, `gh` — always blocked regardless of path
- **Hook format**: `permissionDecision: "deny"` in PreToolUse hook output
- **Compile isolation**: `exec_cli.py` only stages `ref-*` impls, never `peak-*`
- **Run isolation**: `CUDA_EXEC_RUN_TAG` env var ensures `impls.py` resolves
  gen-cuda from current run, not fallback to other runs

### Correctness Enforcement

1. **formal.py**: defaults `correct: False` (not null) — no check = fail
2. **formal.py**: outputs correctness summary after bench table
   (`⚠️ gen-cuda: CORRECTNESS FAILED on N/M configs`)
3. **Supervisor**: detects ✗ in bench result, prepends warning before table
4. **Solver prompt**: "Correctness First — ABSOLUTE RULE"
5. **Steward session_end**: checks for ✗, CONTINUE with "fix correctness"

## Process Model

```
Python process (Supervisor)        asyncio event loop
│
├── ClaudeSDKClient
│   ├── client.query(prompt)       Send instructions
│   ├── client.receive_response()  Receive message stream
│   ├── client.interrupt()         Graceful interrupt
│   └── client.disconnect()        Hard terminate
│        │
│        │ stdio (JSON stream)     IPC channel
│        ▼
│   ┌──────────────────────┐
│   │ claude CLI subprocess │       SDK auto-spawn
│   │ (Solver agent)       │
│   │  ├── Claude API call │
│   │  ├── Execute tools   │       Read/Edit/Bash etc.
│   │  └── MCP tools       │       ask_supervisor, request_formal_bench
│   └──────────────────────┘
│
├── Liveness + Strategy            Inline in main poll loop
│   ├── Every 10s: heartbeat age check (LLM 5min / tool 20min)
│   └── Every 60s: hard_limit, idle, loop, progress (create_task, non-blocking)
│
└── Process group (os.setpgrp)     Kill supervisor → kill all children
```

### Key SDK Options

- `--strict-mcp-config`: only load our MCP servers, skip default plugins
  (prevents stream closed crashes from data/datamate/meta plugins)
- `include_partial_messages=True`: streaming heartbeat during thinking
  (prevents event loop blocking, enables Monitor to run)

## Directory Layout

### Code Repo (`kernel_lab/`)

```
kernel_lab/
├── agents/              # Supervisor, Runner, Monitor, Steward, Events
├── cuda_exec/           # Compile/trial/profile engine
├── plugins/ik/          # Skills (exec, bench, optimize, docs)
├── conf/
│   └── agent/
│       ├── agents.yaml          # Agent configs, monitor timings, budgets
│       ├── prompts/             # System prompts (solver.md, steward.md, etc.)
│       ├── response_prompts/    # Steward scenario prompts (session_end.md, etc.)
│       └── tasks/               # Task descriptions (matmul.md, fa4.md)
├── data/
│   ├── ref/             # Reference implementations (cublas, pytorch)
│   ├── configs/         # Benchmark configs (matmul.json, etc.)
│   └── nvidia-docs/     # CUDA documentation
├── docs/                # Design documents
├── results/             # Benchmark results
└── tests/               # Tests (test_path_rules.py, test_supervisor.py)
```

### KB Repo (`kernel_lab_kb/`)

```
kernel_lab_kb/
├── runs/<run_tag>/              # Per-run isolation
│   ├── gen/{arch}/{kernel}/     # Solver's kernel code
│   ├── gems/{kernel}/{impl}/   # Best results (per-run)
│   ├── impls/<bench_ts>/       # Formal bench snapshots
│   └── journal/                # Agent trajectory
│       ├── solver/<task_slug>/<session_id>/
│       │   ├── events.jsonl    # Structured event stream
│       │   ├── transcript.md   # Human-readable transcript
│       │   ├── meta.json       # Session metadata
│       │   └── heartbeat       # Last activity timestamp
│       ├── benchmarker/...
│       └── steward_*/...
│
│  ── Knowledge (future) ──
├── declarative/         # Concepts + relationships (wikilinks)
├── procedural/          # Principles + boundary conditions
├── episodic/            # Session summaries (Reflector output)
└── index/               # BM25 search index
```

### Scratch Directory (`~/.cuda_exec/<run_tag>/`)

Created by Supervisor at startup. Contains compile artifacts, trial logs,
and workspace files. Isolated per run_tag.

### Bench Directory (`~/.cuda_exec_bench/`)

Used by Benchmarker only. Solver has no access.

## Solver Workflow

1. **Explore** — read ref code, configs, eval harness
2. **Seed** — check current run's gems for starting point
3. **Plan** — output design plan as text (mandatory before coding)
4. **Skeleton** — write kernel structure with TODO placeholders, compile
5. **Implement** — fill in one TODO at a time, compile after each
6. **Trial** — verify correctness (max_abs_error vs cuBLAS)
7. **Bench** — call request_formal_bench
8. **Analyze** — if ✗: fix correctness; if no improvement: profile both
   gen-cuda and ref-cublas, compare metrics, target the gap
9. **Iterate** — brainstorm (data-backed, specific, measurable, feasible),
   implement one change, repeat from step 6

## Future — Reflector + Knowledge Base

```
Reflector (offline + on-demand)
└── Read Episodic → Analyze → Update KB

Triggers:
├── Timed: after each session ends
└── Supervisor-initiated:
    ├── Solver repeatedly makes the same mistake
    ├── Unexpectedly good result appears
    ├── Exploration direction clearly off-track
    └── Encounters new situation not in KB
```

### Knowledge Graph Structure

```
declarative/sm90-wgmma.md
  ├── [[wgmma-interleaved-layout]]     → declarative/
  ├── [[tma-descriptors]]              → declarative/
  └── [[wgmma-optimization-strategy]]  → procedural/

episodic/2026-04-05-wgmma-session.md
  ├── sources: runs/.../journal/...    → execution records (backlink)
  ├── [[sm90-wgmma]]                   → declarative/ (forward link)
  └── [[wgmma-optimization-strategy]]  → procedural/ (forward link)
```

Forward links via `[[wikilinks]]`, backlinks via `sources:` frontmatter.
Together they form a traversable knowledge graph.

### Backlink Rules

All higher-level knowledge must be traceable to original evidence.

| Knowledge Layer | Must backlink to | Example |
|----------------|-----------------|---------|
| episodic/ summary | runs/.../journal/ raw trajectory | `sources: [runs/.../transcript.md]` |
| declarative/ concept | Documentation or discovery session | `sources: [data/nvidia-docs/ptx-isa.html#wgmma]` |
| procedural/ strategy | Benchmark or session that validated it | `sources: [gems/.../report.md]` |

## Detailed Design Documents

- **Supervisor** — [docs/design/supervisor.md](design/supervisor.md)
- **Steward** — [docs/design/steward.md](design/steward.md)
