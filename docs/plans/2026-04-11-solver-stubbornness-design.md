# Solver Stubbornness Enhancement

**Date:** 2026-04-11
**Status:** Design validated
**Goal:** Make Solver commit to optimization directions and persist through difficulty.

## Problem

Three failure modes:

1. **Stuck loop** — Solver loops on the same fix instead of decomposing. Stalls.
2. **Premature abandonment** — Big architecture changes need 5-10 iterations. Solver gives up after 2.
3. **Local tunnel vision** — Solver gets stuck on a minor sub-problem, loses global perspective.

## Three-Layer Framework

| Layer | Component | Controls |
|-------|-----------|----------|
| **L1** | Solver prompt | What Solver can do by itself |
| **L2** | Steward prompts | What requires global perspective |
| **L3** | Workshop + MCP tools | What must be enforced by code |

## Change 1: Breakdown Methodology (L1)

**Addresses:** FM1

### Key Principles addition (`solver.md`)

```
- **Stay the course** — compilation errors mean bugs, not wrong direction. Fix the bugs.
- **Big architecture changes take time** — 5-10 iterations to get right.
  Do NOT judge them by the first 2 attempts. Decompose and persist.
- If stuck 15+ minutes on a sub-problem, call ask_supervisor for global perspective.
```

### Replace "After 4 consecutive failures" in Phase 7

```
### When stuck for more than 2-3 attempts — DECOMPOSE

STOP trying to fix the full kernel. Break the problem down:

1. Extract the problematic section into a minimal standalone test
2. Verify each building block independently
3. Once the minimal test works, integrate back ONE piece at a time
4. Profile the failure — even incorrect kernels have NCU data

You may ONLY switch direction after exhausting ALL FOUR steps
AND calling set_direction (which triggers Steward review).
```

## Change 2: Direction System (L1 + L2 + L3)

**Addresses:** FM2

### Concept

A "Direction" is the optimization strategy Solver commits to after brainstorming. One active direction at a time. No sub-directions. **Every `set_direction` goes through Steward review** — no auto-approve.

### Data Structure (5 fields)

```json
{
  "name": "warp-specialization",
  "description": "Split warps into TMA producer and WGMMA consumer groups",
  "opportunity": "tensor core 3% → 60%+, expect ~10x TFLOPS gain",
  "evidence": "NCU: ref cuBLAS uses 12 warps = 3 WGs, 85% tensor core util; our kernel 1 WG, 3% util",
  "ideas": [
    "2-WG producer/consumer split",
    "3-stage TMA pipeline with mbarrier",
    "(alt) async WGMMA overlap with epilogue"
  ]
}
```

Seq, wave, timestamp encoded in file path. Only APPROVED directions written to disk.

### Storage — per-wave sequence

```
w000/directions/
  001_tma-pipeline.json
w001/directions/
  001_tma-pipeline.json          # inherited
w002/directions/
  001_tma-pipeline.json          # inherited
  002_warp-specialization.json   # direction change, approved
```

### MCP Tool: `set_direction(direction_json)`

Workshop sends every call to Steward. Steward responds APPROVED / REVISE / REJECTED. Only APPROVED persists to disk and activates.

### Steward Role

```
Core Role: Methodologist with Global Perspective

You see the FULL trajectory. Solver sees only its current problem.
You do NOT write code. You tell Solver:
- What to focus on
- When to decompose
- When to persist
- When to step back
```

### Steward Direction Review

Evaluates: specificity, evidence quality, opportunity realism, idea concreteness. If direction change: whether Solver exhausted the current direction.

### Solver Prompt Addition

```
After brainstorming (Phase 3), register your direction via set_direction.
Steward reviews every direction. Must include evidence and expected gain.
If Steward asks for revision, revise and re-submit.
You cannot write kernel code until a direction is approved.
```

## Change 3: Direction Monitoring (L2 + L3)

**Addresses:** FM2, FM3

### Pre-tool gate — no direction → no writes

Block Write, Edit, and Bash write operations on kernel files if no active direction. Covers all write paths including bash redirects.

### Periodic progress check

Activate the existing `progress_check` scenario (currently auto-continued). Every 10-15 min, Steward reviews trajectory against the registered direction. Checks for: direction drift, stuck loop, local tunnel vision, healthy progress.

### Event-triggered "Diffusion" reviews

At key operation points, Steward generates a **fresh, contextual** reinforcement of the direction from a new angle. NOT a static reminder — each review connects the current activity to the direction's 初心 from a different perspective.

**Triggers** (detected from stream events):

| Trigger | Steward connects to direction from angle of... |
|---------|------------------------------------------------|
| Write/Edit `*.cu` | architectural alignment |
| compile | expected outcome vs actual result |
| trial | correctness priorities |
| profile | NCU metrics vs direction targets |
| `request_formal_bench` | pre-bench alignment |
| `submit_bench_reflection` | reflection vs 初心 |

**Implementation:** async, non-blocking. Workshop detects trigger in stream → fires Steward review → injects guidance into Solver via `client.query()`. Rate limit: at most one review per 2 minutes.

**Steward prompt principle:** Each review must offer a NEW ANGLE. Connect the specific result to the direction. If drift, explain what drifted and why 初心 still holds. If progress, acknowledge and point to next milestone. 2-3 sentences.

## Change 4: Journal Directory Restructure (L3)

### New structure — wave-first

```
journal/
  w000_<ts>/
    solver/                      # Solver output
      transcript.md
      events.jsonl
      stdout.log
      stdin.log
      stderr.log
    steward/                     # Steward calls during this wave
      direction_review_<ts>/
      progress_check_<ts>/
      stuck_<ts>/
      session_end_<ts>/
    directions/                  # Per-wave sequence
      001_tma-pipeline.json
    process_start.json
    process_end.json
    heartbeat.json
```

- Wave = atomic unit
- No task_slug layer
- No per-agent top-level
- Steward under wave

## Summary

| File | Change | FM |
|------|--------|-----|
| `solver.md` | Persistence + decomposition + direction rules | 1, 2, 3 |
| `steward.md` | Global Perspective Methodologist role | 2, 3 |
| `response_prompts/progress_check.md` | Direction-aware monitoring | 3 |
| `response_prompts/direction_review.md` | New — APPROVED/REVISE/REJECTED | 2 |
| `runner.py` | `set_direction` MCP tool | 2 |
| `workshop.py` | Direction tracking, pre-tool gate, diffusion reviews | 2, 3 |
| `steward.py` | `review_direction`, `review_direction_alignment` | 2 |
| `storage.py` | Wave-first journal restructure | — |

## What Does NOT Change

- Steward remains fire-and-forget
- Wave/Session lifecycle unchanged
- Existing scenarios unchanged
- One active direction at a time
