# Librarian System — Knowledge Base Design

## 1. Purpose

Turn raw reflections from kernel optimization work into organized, canonical
knowledge. The system acts as a library: raw materials come in, get analyzed
and classified, then enter the permanent collection only after review.

## 2. System Overview

```
                    ┌─────────────────┐
                    │    Librarian    │  Final arbiter
                    │  (orchestrate,  │  Only role with
                    │   integrate,    │  canonical write
                    │   decide)       │  authority
                    └──┬─────┬───┬───┘
                       │     │   │
            ┌──────────┘     │   └──────────┐
            ▼                ▼              ▼
   ┌─────────────┐  ┌──────────────┐  ┌──────────┐
   │ Information  │  │  Taxonomist  │  │  Auditor  │
   │  Analyst     │  │  (structure, │  │  (verify, │
   │  (extract,   │  │   classify,  │  │   check,  │
   │   propose)   │  │   place)     │  │   challenge)
   └──────┬───────┘  └──────────────┘  └──────────┘
          │
          ▼
   ┌──────────────┐
   │ Raw Sources   │
   │ (reflections, │
   │  trajectories,│
   │  raw files)   │
   └──────────────┘
```

## 3. Knowledge Layers

Three layers separate raw evidence from canonical knowledge.

| Layer | Content | Location | Mutability |
|-------|---------|----------|------------|
| **Raw Source** | Solver reflections, referenced files, trajectories | `kernel_lab_kb/runs/<run_tag>/reflections/` | Immutable — never modified |
| **Proposed** | Information Analyst output (proposals) | `kernel_lab_kb/wiki/_proposals/` | Temporary — consumed by Librarian |
| **Canon** | Librarian-approved wiki pages | `kernel_lab_kb/wiki/` | Controlled — only Librarian writes |

**Rule**: Raw sources never enter canon directly. They must pass through
Information Analyst extraction and Librarian review first.

## 4. Role Definitions

### 4.1 Librarian

The Librarian is a **router, integrator, and final arbiter** — not the most
knowledgeable person in the room, but the most decisive.

**Responsibilities:**
- Receive proposals from Information Analyst
- Decide whether to consult Taxonomist, Auditor, or both
- Integrate expert opinions into a final decision
- Execute the canonical write (create, update, merge, split, link, defer)
- Maintain quality gate — only well-evidenced, reusable knowledge enters canon

**Does NOT**: Deep-dive into source materials, design taxonomy from scratch,
or validate evidence line-by-line.

### 4.2 Information Analyst

Reads raw reflections and source files, extracts structured knowledge, and
produces proposals for Librarian review.

**First implementation**: Reflection Analysis (other analysis types later).

**Two-phase process:**

**Phase 1 — Source-grounded extraction** (no wiki access):
- Read reflection text and referenced original files
- Extract knowledge units independently
- Bind each claim to its evidence source
- Assess confidence and limitations
- *Purpose*: Avoid anchoring bias from existing wiki structure

**Phase 2 — Wiki-aware reconciliation** (limited wiki access):
- Search existing wiki for related pages (top-K semantic match)
- Determine: is this a create, update, or link?
- Identify touch set (pages that would be affected)
- Produce final proposal with target page suggestions
- *Purpose*: Align proposal with existing knowledge without being captured by it

**Output**: Proposed Knowledge Change Packet (see §6).

**Does NOT**: Write to canonical wiki, decide final taxonomy placement, or
resolve structural conflicts.

### 4.3 Taxonomist

Decides **where knowledge belongs** in the canonical wiki structure.

**Responsibilities:**
- Page boundaries — should this be one page or three?
- Canonical ownership — which page "owns" this piece of knowledge?
- Hierarchy decisions — category, topic, sub-topic placement
- Merge/split/move — when pages should combine or separate
- Hub/index page design — creating navigational structure
- Naming conventions — consistent page titles and slugs

**Key distinction from Information Analyst:**
- Analyst: "I found these candidate knowledge units" (what was discovered)
- Taxonomist: "Here's where they belong in the structure" (who owns what)

**Does NOT**: Read source materials directly, extract knowledge from
reflections, or validate evidence.

### 4.4 Auditor

Finds **problems, conflicts, and weak spots** in proposals.

**Responsibilities:**
- Evidence validation — is the claim actually supported by the source?
- Conflict detection — does it contradict existing wiki content?
- Over-generalization check — is local experience being treated as universal?
- Missing context — are limiting conditions being omitted?
- Hidden assumptions — what's being taken for granted?
- Future risk — will this create technical debt in the wiki?

**Thinking style**: Evidence-oriented, boundary-sensitive, critical. Good at
seeing "what's wrong" that others miss.

**Does NOT**: Build structure (that's Taxonomist), extract knowledge from
sources (that's Analyst), or make final decisions (that's Librarian).

## 5. Wiki Structure

### 5.1 Physical Layout

```
kernel_lab_kb/wiki/
├── _proposals/           # Staging area (Analyst output, consumed by Librarian)
├── _index.md             # Master index / table of contents
├── concepts/             # Technical concepts, GPU architecture, algorithms
│   ├── _index.md         # Hub page for concepts
│   ├── wgmma.md
│   ├── tma-descriptors.md
│   └── smem-swizzle.md
├── patterns/             # Optimization patterns, coding patterns
│   ├── _index.md
│   ├── warp-specialization.md
│   └── smem-epilogue.md
├── problems/             # Known issues, failure modes, gotchas
│   ├── _index.md
│   ├── register-spilling.md
│   └── l2-cache-thrashing.md
├── decisions/            # Design decisions with rationale
│   └── _index.md
└── references/           # External resources, papers, docs
    └── _index.md
```

**Rules:**
- Max 2-3 levels of directory nesting
- `_index.md` in each directory serves as hub/navigation page
- `_proposals/` is temporary staging, not part of canon
- Hub pages are first-class citizens, not afterthoughts

### 5.2 Page Model

Every wiki page has YAML frontmatter:

```yaml
---
id: "wgmma-descriptor-layout"        # Stable internal ID (survives renames)
title: "WGMMA Descriptor Layout"     # Human-readable title
category: "concepts"                  # Top-level category
tags: ["sm90", "wgmma", "tensor-core", "hopper"]
aliases: ["wgmma-desc", "wgmma-layout"]
status: active                        # active | historical | deprecated | under_review
created: 2026-04-10
updated: 2026-04-10
sources:                              # Backlinks to evidence
  - "runs/supervisor_run_20260409/reflections/20260409_103057.md"
---

# WGMMA Descriptor Layout

Content here...

## Related

- [[tma-descriptors]]
- [[smem-swizzle]]
```

**Wikilinks**: Use `[[page-slug]]` syntax for cross-references.
Page slug = the `id` field, which is stable across renames.

### 5.3 Categories

Initial categories (can evolve):

| Category | Purpose | Examples |
|----------|---------|---------|
| `concepts/` | Technical concepts and architecture | WGMMA, TMA, barriers, swizzle |
| `patterns/` | Reusable optimization approaches | Warp specialization, SMEM epilogue, pipeline depth |
| `problems/` | Known issues and failure modes | Register spilling, bank conflicts, clock throttling |
| `decisions/` | Design decisions with rationale | Why 128×256 tile, why 3-WG design |
| `references/` | External resources | Papers, NVIDIA docs, GTC talks |

## 6. Proposal Format

Information Analyst output — the Proposed Knowledge Change Packet:

```yaml
# Proposed Knowledge Change Packet
essence: "SMEM-buffered epilogue eliminates 50% global store sector waste"

knowledge_units:
  - type: lesson
    claim: "Per-thread scalar BF16 stores waste 50% of L1 sectors"
    evidence:
      - "NCU profile: 16/32 bytes per sector on global stores"
      - "runs/.../reflections/20260409_103057.md"
    confidence: high
    limitations: "Specific to m64n64k16 WGMMA output layout"

  - type: pattern
    claim: "SMEM staging with padded stride (72 = 64+8) eliminates bank conflicts"
    evidence:
      - "Bank analysis: bank(L) = (row*36+L)%32, all unique for L∈[0,31]"
    confidence: high
    limitations: "Padding ratio depends on tile width"

  - type: opportunity
    claim: "TMA store epilogue could further reduce epilogue cycles"
    evidence:
      - "cuBLAS uses TMA store (nsys trace confirms nvjet kernel)"
    confidence: medium
    limitations: "Requires significant architectural change"

novelty: new                    # new | supplement | conflict | duplicate

target_pages:
  - page: "smem-epilogue"
    action: create
  - page: "wgmma-descriptor-layout"
    action: link

recommended_action: create

proposed_draft: |
  ---
  id: smem-epilogue
  title: "SMEM-Buffered Coalesced Epilogue"
  category: patterns
  tags: [epilogue, smem, coalescing, store-optimization]
  status: active
  ---
  # SMEM-Buffered Coalesced Epilogue
  ...

open_issues:
  - "Does the 8-element padding generalize to other tile widths?"
  - "How much does the epilogue barrier cost relative to compute?"

source_refs:
  - reflection: "runs/.../reflections/20260409_103057.md"
  - raw_file: "results/sm90/h100_sxm/matmul/20260407_0430_v25.md"
```

## 7. Knowledge Unit Types

| Type | Description | Example |
|------|-------------|---------|
| **Problem** | An issue encountered | "cuBLAS autotune timeout in formal bench" |
| **Opportunity** | A potential improvement | "TMA store could replace scalar epilogue" |
| **Hypothesis** | An untested theory | "Cluster 2×1 multicast would halve A traffic" |
| **Lesson** | A validated learning | "50% sector waste from uncoalesced BF16 stores" |
| **Open Question** | Needs further investigation | "Does padding ratio scale with tile width?" |
| **Evidence** | Data supporting a claim | "NCU: 16/32 bytes per sector" |
| **Constraint** | A limitation or boundary | "H100 R&R SKU peaks at 800 TFLOPS (650W TDP)" |
| **Pattern** | A reusable approach | "Dual-tile kernel: NQ=2 for small, NQ=4 for large" |

## 8. Workflow

### 8.1 Three Paths

**Light path** — low risk, clear content, single page:
```
Reflection → Information Analyst → Librarian → Write to canon
```

**Medium path** — structure unclear or multiple pages:
```
Reflection → Information Analyst → Librarian → Taxonomist → Librarian → Write
```

**Heavy path** — high value, potential conflicts, restructuring:
```
Reflection → Information Analyst → Librarian → Taxonomist + Auditor → Librarian → Write
```

### 8.2 Escalation Rules

| Condition | Who to involve |
|-----------|---------------|
| Clear content, single page update, no conflicts | Librarian alone |
| Placement unclear, multiple pages affected | + Taxonomist |
| New topic area, merge/split needed | + Taxonomist |
| Subjective reflection, weak evidence | + Auditor |
| Contradicts existing wiki, high-impact claim | + Auditor |
| New top-level category, major restructuring | + Taxonomist + Auditor |
| Important canonical page, high-value knowledge | + Taxonomist + Auditor |
| Changes to category tree, conflicting experts | Escalate to human |

## 9. Canon Entry Criteria

### What qualifies:
- **Reusable** — applies beyond a single incident
- **Evidenced** — has supporting data or source reference
- **Generalizable** — or clearly scoped to specific conditions
- **Useful** — helps future work (optimization, debugging, design)
- **Non-trivial** — not just restating obvious facts

### What stays out:
- Emotional reactions without technical content
- Weak-evidence speculation (→ stays as "hypothesis" with low confidence)
- Context-free experience fragments
- Extremely situation-specific conclusions (→ stays in raw reflection)

## 10. Conflict Handling

When a new proposal conflicts with existing wiki:

| Situation | Action |
|-----------|--------|
| Old content is wrong | Analyst flags → Auditor confirms → Librarian updates |
| Both valid, different conditions | Create nuanced page with scoping/conditions |
| New evidence insufficient | Mark as "under_review", don't overwrite |

**Principle**: Never silently overwrite. Always preserve provenance of both
old and new knowledge.

## 11. Temporal Awareness

Wiki pages track time dimension via frontmatter:

| Field | Purpose |
|-------|---------|
| `created` | When page was first created |
| `updated` | When page was last modified |
| `status` | `active` / `historical` / `deprecated` / `under_review` |

**Rule**: Knowledge can become stale. Periodic review (future extension) should
check `updated` dates and flag pages that haven't been confirmed in N months.

## 12. Process Architecture

Two independent Python processes, sharing `kernel_lab_kb/` via filesystem:

```
Workshop (agents/workshop.py)           Library (agents/library.py)
python -m agents.workshop               python -m agents.library
│                                        │
├── Solver (LLM agent)                  ├── Librarian (LLM agent)
│   ├── Read/Write/Bash                 │   ├── Read/Write wiki
│   ├── ik:exec, ik:docs               │   └── MCP tools:
│   └── MCP tools:                      │       ├── consult_taxonomist
│       ├── request_bench               │       ├── consult_auditor
│       ├── ask_supervisor              │       └── clarify_analyst
│       └── submit_reflection ──┐       │
│                               │       ├── Information Analyst (LLM) ×N
├── Steward (LLM agent)        │       │   └── reads reflections, writes proposals
│   └── reviews Solver          │       │
│                               │       ├── Taxonomist (LLM, read-only)
└── Benchmarker (subprocess)    │       │   └── advises on structure
                                │       │
                                │       └── Auditor (LLM, read-only)
                                │           └── validates evidence
                                │
                   ┌────────────┘
                   ▼
kernel_lab_kb/
├── runs/<run_tag>/reflections/    ← Workshop writes, Library reads
├── wiki/_proposals/               ← Analyst writes, Librarian reads
└── wiki/{concepts,patterns,...}/  ← Librarian writes (canonical)
```

**Workshop** (`agents/workshop.py`): Manages the Solver optimization loop.
Renamed from `supervisor.py` to avoid collision with Claude Agent SDK's
"supervisor" concept (which refers to the Python client process itself).

**Library** (`agents/library.py`): Manages the knowledge base pipeline.
Librarian is a long-running LLM agent that consumes proposals from a queue.
Expert agents (Taxonomist, Auditor) are spawned on-demand via MCP tool
dispatch, same pattern as Workshop spawning Benchmarker.

**Permission isolation** (via `agents.yaml` tool_rules):

| Agent | Can write | Read-only |
|-------|-----------|-----------|
| Solver | `runs/<run_tag>/gen/`, `reflections/` | `data/ref/`, `data/configs/` |
| Steward | — | transcript |
| Librarian | `wiki/` (canonical) | `wiki/_proposals/`, `wiki/**` |
| Information Analyst | `wiki/_proposals/` | `runs/*/reflections/`, `wiki/**` |
| Taxonomist | — | `wiki/**` |
| Auditor | — | `wiki/**` |

**Triggers for Information Analyst:**
- After each Wave completes (automatic batch)
- On demand by Librarian
- Batch processing of accumulated reflections

**Wiki as Solver input** (future): Before starting a new optimization task,
Solver queries the wiki for relevant patterns, known problems, and past
lessons for the target kernel.

## 13. Failure Modes

| Failure | Cause | Mitigation |
|---------|-------|------------|
| Over-generalization | Local experience → universal rule | Auditor checks evidence breadth |
| Anchoring bias | Analyst shaped by existing wiki | Two-phase extraction (source-first) |
| Page fragmentation | Too many small pages | Taxonomist enforces minimum page substance |
| Over-conservative | Auditor blocks everything | Librarian can override with justification |
| Taxonomy bloat | Too many categories | Limit depth to 2-3 levels |
| Stale knowledge | No expiry mechanism | Temporal awareness + periodic review |
| Duplicate pages | Analyst misses existing content | Phase 2 wiki search catches most cases |
| Loss of nuance | Complex insight simplified too much | Auditor flags missing context |

## 14. Future Extensions

- **Additional analysis types**: Source Analysis, Conflict Analysis, Coverage Analysis, Gap Analysis
- **Knowledge retrieval skill**: Solver queries wiki during optimization (`/kb:search`)
- **Periodic health checks**: Orphan pages, stale content, broken links, coverage gaps
- **Knowledge graph visualization**: Interactive view of page relationships
- **Cross-run learning**: Aggregate patterns across multiple Workshop runs
- **Human review interface**: Web UI for reviewing proposals and wiki state
