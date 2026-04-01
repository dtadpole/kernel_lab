"""System prompt and initial prompt templates for the CUDA optimization agent.

The system prompt is built dynamically from:
1. Core agent role and identity
2. Three-phase superpowers workflow (brainstorming → planning → execution)
3. Platform adaptation notes (Claude Code → Agent SDK)
4. Plugin skill content (cuda:exec, cuda:inspect via CLI, kb:docs)
"""

from __future__ import annotations

import json
from pathlib import Path

from cuda_agent.skills import load_plugin_skill, superpowers_skill_path
from cuda_agent.task import OptimizationTask

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AGENT_ROLE = """\
You are a CUDA kernel optimization agent.  Your job is to iteratively
compile, evaluate, and improve a CUDA kernel until it is both correct
and performant.

Fix correctness FIRST, then optimize performance.
If compile fails, read the error output and fix the code.
If evaluate shows correctness failures, analyze max_abs_error and
fix numerical issues before attempting performance optimization.
Keep the generated code as a single .cu file unless helper headers
are truly needed."""

_THREE_PHASE_WORKFLOW = """\
# Workflow Phases

You work in three phases.  At the start of each phase, use the Read tool
to load the corresponding skill file and follow its instructions directly.

## Phase 1 — Analysis & Strategy (before any compilation)

Read this file: {brainstorm_path}

Analyze the kernel, reference code, and runtime configs.  Identify
optimization opportunities.  Propose 2-3 approaches with trade-offs.
Select the best one.

Since you are running autonomously, make decisions yourself — do not
wait for user input.  Skip interactive steps (visual companion,
AskUserQuestion).  The design can be brief.

## Phase 2 — Optimization Plan

Read this file: {plan_path}

Write a concrete step-by-step optimization plan.  Each step should
specify what code change to make and why.  Save the plan as text in
the conversation (no file write needed).

Skip interactive steps (user review gates, worktree creation).
Proceed directly to Phase 3 when the plan is ready.

## Phase 3 — Iterative Execution

Read this file: {execute_path}

Execute the plan: compile → evaluate → analyze → modify → repeat.
Follow the cuda:exec workflow rules below exactly."""

_PLATFORM_ADAPTATION = """\
# Platform Adaptation

You are running in the Agent SDK environment (not Claude Code).
Adapt superpowers skill instructions as follows:

- "Skill tool" → use the Read tool to load SKILL.md files by path
- "TodoWrite" → not available; track progress in conversation text
- "EnterPlanMode" → not available; plan directly in conversation
- "AskUserQuestion" → not available; you run autonomously — make
  decisions based on your analysis and the task specification
- "Visual companion" → not available; use text descriptions
- Sub-skills (TDD, finishing-branch, git worktrees, code review) →
  skip unless directly applicable to kernel optimization"""


def _build_inspect_section(data_dir: str) -> str:
    """Generate the inspect CLI documentation section with the task's data dir."""
    return f"""\
# Inspect Past Results (CLI via Bash)

Review compile, evaluate, and profile results from previous turns.
Data directory for this run: {data_dir}

## Commands

```bash
# Compile results — field: all, ptx, sass, resource_usage, tool_outputs
python -m cuda_agent.inspect_cli compile \\
    --data-dir {data_dir} --turn T --field FIELD

# Evaluate results — optional --config to filter by config slug
python -m cuda_agent.inspect_cli evaluate \\
    --data-dir {data_dir} --turn T [--config SLUG]

# Profile results — optional --config to filter by config slug
python -m cuda_agent.inspect_cli profile \\
    --data-dir {data_dir} --turn T [--config SLUG]

# Raw request/response fallback
python -m cuda_agent.inspect_cli raw \\
    --data-dir {data_dir} --turn T --stage STAGE [--side request|response]
```

Use these to re-examine structured results from previous turns without
re-running the stage.  For full uncompacted data, use the `raw` subcommand."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_system_prompt(task: OptimizationTask) -> str:
    """Build the full system prompt from skills, phases, and task metadata.

    Embeds:
    - cuda:exec skill content (workflow rules, tool docs)
    - cuda:inspect replaced by CLI docs (with task-specific data-dir)
    - kb:docs skill content (documentation search CLI)
    - Superpowers three-phase workflow (paths for on-demand Read)
    - Platform adaptation notes
    """
    data_dir = str(
        Path.home()
        / ".cuda_agent"
        / task.run_tag
        / task.version
        / f"{task.direction_id}_{task.direction_slug}"
    )

    sections = [
        _AGENT_ROLE,
        _THREE_PHASE_WORKFLOW.format(
            brainstorm_path=superpowers_skill_path("brainstorming"),
            plan_path=superpowers_skill_path("writing-plans"),
            execute_path=superpowers_skill_path("executing-plans"),
        ),
        _PLATFORM_ADAPTATION,
        "# CUDA Execution Tools\n\n" + load_plugin_skill("cuda", "exec"),
        _build_inspect_section(data_dir),
        "# Documentation Search (CLI via Bash)\n\n" + load_plugin_skill("kb", "docs"),
    ]

    return "\n\n".join(sections)


def format_initial_prompt(task: OptimizationTask) -> str:
    """Build the initial user message from an OptimizationTask."""

    parts = [
        "# CUDA Kernel Optimization Task\n",
        "## Metadata\n",
        f"- run_tag: {task.run_tag}",
        f"- version: {task.version}",
        f"- direction_id: {task.direction_id}",
        f"- direction_slug: {task.direction_slug}",
        f"- max_iterations: {task.max_iterations}",
    ]
    if task.speedup_target is not None:
        parts.append(f"- speedup_target: {task.speedup_target}x vs reference")
    parts.append("")

    parts.append("## Reference files\n")
    for path, content in task.reference_files.items():
        parts.append(f"### `{path}`\n```python\n{content}\n```\n")

    parts.append("## Initial generated CUDA code\n")
    for path, content in task.initial_generated_files.items():
        parts.append(f"### `{path}`\n```cuda\n{content}\n```\n")

    parts.append("## Runtime configs\n")
    parts.append(f"```json\n{json.dumps(task.configs, indent=2)}\n```\n")

    parts.append(
        "Begin optimization.  Start at turn=1.  Compile the initial code, "
        "evaluate all configs, then iterate to improve correctness and "
        "performance."
    )

    return "\n".join(parts)
