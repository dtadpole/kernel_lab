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
    action: str              # e.g. SUCCESS, CONTINUE, ABORT, INJECT, KILL, ON_TRACK
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
    "ALLOW": 1, "DENY": 1, "SUCCESS": 1,
    "EXTEND": 1, "ON_TRACK": 1,
    # Level 2: inject / continue
    "CONTINUE": 2, "INJECT": 2, "WRAP_UP": 2, "DRIFTING": 2, "REDIRECT": 2,
    # Level 3: abort
    "ABORT": 3,
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
        transcript_path: str,
        question: str,
        solver_context: str = "",
    ) -> str:
        """Solver asks for guidance. Returns free-text answer (intervention_level=1)."""
        full_question = question
        if solver_context:
            full_question = f"{question}\n\nContext: {solver_context}"

        return await self.router.respond_raw("ask_question", {
            "transcript_path": transcript_path,
            "question": full_question,
        })

    # ── Scenario 2: Permission check ──

    async def check_permission(
        self,
        transcript_path: str,
        tool_name: str,
        tool_input: dict,
    ) -> StewardResponse:
        """Review a restricted tool call. Returns ALLOW/DENY (intervention_level=1)."""
        verdict = await self.router.respond("permission", {
            "transcript_path": transcript_path,
            "tool_name": tool_name,
            "tool_input": str(tool_input),
        })
        return _to_steward_response(verdict)

    # ── Scenario 3: Solver is stuck ──

    async def handle_stuck(
        self,
        transcript_path: str,
        alert_type: str,
        alert_details: str,
        elapsed_time: str,
    ) -> StewardResponse:
        """Solver stalled. Returns CONTINUE/INJECT/INTERRUPT (intervention_level=1/2/4)."""
        verdict = await self.router.respond("stuck", {
            "transcript_path": transcript_path,
            "alert_type": alert_type,
            "alert_details": alert_details,
            "elapsed_time": elapsed_time,
        })
        return _to_steward_response(verdict)

    # ── Scenario 4: Session end review ──

    async def review_session_end(
        self,
        transcript_path: str,
        result_text: str,
        stop_reason: str,
        elapsed_time: str,
        total_tool_calls: int,
        error_count: int,
    ) -> StewardResponse:
        """Review whether Solver truly finished. Returns SUCCESS/CONTINUE/ABORT (intervention_level=1/2/3)."""
        verdict = await self.router.respond("session_end", {
            "transcript_path": transcript_path,
            "result_text": result_text,
            "stop_reason": stop_reason,
            "elapsed_time": elapsed_time,
            "total_tool_calls": str(total_tool_calls),
            "error_count": str(error_count),
        })
        return _to_steward_response(verdict)

    # ── Scenario 5: Time limit exceeded ──

    async def handle_time_limit(
        self,
        transcript_path: str,
        elapsed_time: str,
        time_limit: str,
    ) -> StewardResponse:
        """Solver exceeded time budget. Returns EXTEND/WRAP_UP/KILL (intervention_level=1/2/4)."""
        verdict = await self.router.respond("time_limit", {
            "transcript_path": transcript_path,
            "elapsed_time": elapsed_time,
            "time_limit": time_limit,
        })
        return _to_steward_response(verdict)

    # ── Scenario 6: Periodic progress check ──

    async def check_progress(
        self,
        transcript_path: str,
        elapsed_time: str,
    ) -> StewardResponse:
        """Periodic progress assessment. Returns ON_TRACK/DRIFTING/REDIRECT (intervention_level=1/2)."""
        verdict = await self.router.respond("progress_check", {
            "transcript_path": transcript_path,
            "elapsed_time": elapsed_time,
        })
        return _to_steward_response(verdict)
