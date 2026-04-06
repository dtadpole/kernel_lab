"""Steward — provides runtime guidance to Solver and other agents.

Six typed methods, one per scenario. Each delegates to ResponseRouter
for prompt selection and Agent SDK query execution.

Steward is read-only — it never modifies files. It can use WebSearch/WebFetch
for research. All Steward calls are logged automatically by the AgentRunner's
hook/event system (SessionLog → agent_journal).

Current role: operational guidance (answer questions, review sessions).
Future evolution:
  - Mentor: teach from accumulated KB experience
  - Coach: guide optimization methodology
  - Critic: find logical flaws before implementation
  - Strategist: select among competing strategies
  - Consultant: strategic advice for Supervisor
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agents.config import StorageConfig
from agents.response_router import ResponseRouter, ResponseVerdict


# ── Steward Response with intervention level ──

@dataclass
class StewardResponse:
    """Steward's decision with explicit intervention level."""
    action: str              # e.g. ACCEPT, REJECT, INJECT, KILL, ON_TRACK
    detail: str              # text after colon (guidance, reason, minutes)
    reasoning: str           # full analysis text
    intervention_level: int  # 1=inline, 2=inject, 3=restart, 4=kill

    @property
    def needs_solver_interrupt(self) -> bool:
        return self.intervention_level >= 2

    @property
    def needs_solver_restart(self) -> bool:
        return self.intervention_level >= 3

    @property
    def needs_solver_kill(self) -> bool:
        return self.intervention_level >= 4


# ── Action → intervention level mapping ──

_ACTION_LEVELS = {
    # Level 1: inline
    "ALLOW": 1, "DENY": 1, "CONTINUE": 1, "ACCEPT": 1,
    "EXTEND": 1, "ON_TRACK": 1,
    # Level 2: inject
    "INJECT": 2, "WRAP_UP": 2, "DRIFTING": 2, "REDIRECT": 2,
    # Level 3: restart
    "REJECT": 3, "RETRY": 3,
    # Level 4: kill
    "INTERRUPT": 4, "KILL": 4,
}


def _to_steward_response(verdict: ResponseVerdict) -> StewardResponse:
    """Convert a ResponseVerdict to a StewardResponse with intervention level."""
    level = _ACTION_LEVELS.get(verdict.action, 1)
    return StewardResponse(
        action=verdict.action,
        detail=verdict.detail,
        reasoning=verdict.reasoning,
        intervention_level=level,
    )


class Steward:
    """Provides runtime guidance via six typed methods.

    Each method corresponds to a specific scenario. Internally delegates
    to ResponseRouter which loads scenario-specific prompts from
    conf/agent/response_prompts/*.md.
    """

    def __init__(
        self,
        prompts_dir: str | Path = "conf/agent/response_prompts",
        model: str = "claude-sonnet-4-6",
        storage_config: StorageConfig | None = None,
    ):
        self.router = ResponseRouter(
            prompts_dir=Path(prompts_dir), model=model,
            storage_config=storage_config,
        )

    # ── Scenario 1: Solver asks a question ──

    async def answer_question(
        self,
        question: str,
        solver_context: str,
        task: str,
        session_summary: str,
    ) -> str:
        """Solver asks for guidance. Returns free-text answer (intervention_level=1)."""
        return await self.router.respond_raw("ask_question", {
            "task_description": task,
            "session_summary": session_summary,
            "question": question,
            "solver_context": solver_context,
        })

    # ── Scenario 2: Permission check ──

    async def check_permission(
        self,
        tool_name: str,
        tool_input: dict,
        task: str,
        recent_tool_calls: str,
    ) -> StewardResponse:
        """Review a restricted tool call. Returns ALLOW/DENY (intervention_level=1)."""
        verdict = await self.router.respond("permission", {
            "tool_name": tool_name,
            "tool_input": str(tool_input),
            "task_description": task,
            "recent_tool_calls": recent_tool_calls,
        })
        return _to_steward_response(verdict)

    # ── Scenario 3: Solver is stuck ──

    async def handle_stuck(
        self,
        alert_type: str,
        alert_details: str,
        task: str,
        recent_events: str,
        tool_call_counts: str,
        elapsed_time: str,
    ) -> StewardResponse:
        """Solver stalled. Returns CONTINUE/INJECT/INTERRUPT (intervention_level=1/2/4)."""
        verdict = await self.router.respond("stuck", {
            "alert_type": alert_type,
            "alert_details": alert_details,
            "task_description": task,
            "recent_events": recent_events,
            "tool_call_counts": tool_call_counts,
            "elapsed_time": elapsed_time,
        })
        return _to_steward_response(verdict)

    # ── Scenario 4: Session end review ──

    async def review_session_end(
        self,
        result_text: str,
        stop_reason: str,
        task: str,
        elapsed_time: str,
        total_tool_calls: int,
        error_count: int,
        session_summary: str,
    ) -> StewardResponse:
        """Review whether Solver truly finished. Returns ACCEPT/REJECT/RETRY (intervention_level=1/3)."""
        verdict = await self.router.respond("session_end", {
            "task_description": task,
            "result_text": result_text,
            "stop_reason": stop_reason,
            "elapsed_time": elapsed_time,
            "total_tool_calls": str(total_tool_calls),
            "error_count": str(error_count),
            "session_summary": session_summary,
        })
        return _to_steward_response(verdict)

    # ── Scenario 5: Time limit exceeded ──

    async def handle_time_limit(
        self,
        elapsed_time: str,
        time_limit: str,
        task: str,
        recent_progress: str,
        tool_call_trend: str,
    ) -> StewardResponse:
        """Solver exceeded time budget. Returns EXTEND/WRAP_UP/KILL (intervention_level=1/2/4)."""
        verdict = await self.router.respond("time_limit", {
            "elapsed_time": elapsed_time,
            "time_limit": time_limit,
            "task_description": task,
            "recent_progress": recent_progress,
            "tool_call_trend": tool_call_trend,
        })
        return _to_steward_response(verdict)

    # ── Scenario 6: Periodic progress check ──

    async def check_progress(
        self,
        task: str,
        session_summary: str,
        elapsed_time: str,
        recent_events: str,
        tool_call_counts: str,
    ) -> StewardResponse:
        """Periodic progress assessment. Returns ON_TRACK/DRIFTING/REDIRECT (intervention_level=1/2)."""
        verdict = await self.router.respond("progress_check", {
            "task_description": task,
            "session_summary": session_summary,
            "elapsed_time": elapsed_time,
            "recent_events": recent_events,
            "tool_call_counts": tool_call_counts,
        })
        return _to_steward_response(verdict)
