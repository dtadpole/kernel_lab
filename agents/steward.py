"""Steward — provides runtime guidance to Solver.

A situational methodologist who reads working patterns and tailors
guidance to the current situation. Not a CUDA expert.
Two core responsibilities:
  1. Correctness stuck → guide decomposition
  2. Performance direction drift → redirect to 初心
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agents.config import StorageConfig
from agents.response_router import ResponseRouter, ResponseVerdict


# ── Steward Response with intervention level ──

@dataclass
class StewardResponse:
    """Steward's decision with explicit intervention level."""
    action: str              # e.g. SUCCESS, CONTINUE, ABORT, APPROVED, REVISE
    detail: str              # text after colon (guidance, reason)
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
    "ALLOW": 1, "DENY": 1, "SUCCESS": 1, "APPROVED": 1,
    "EXTEND": 1, "ON_TRACK": 1,
    # Level 2: inject / continue
    "CONTINUE": 2, "INJECT": 2, "REDIRECT": 2, "EXPLORE": 2,
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


def _base_context(transcript_path: str, events_path: str, recent_events: str, mode: str = "exploring") -> dict:
    """Build the common context variables every scenario needs."""
    return {
        "mode": mode,
        "transcript_path": transcript_path,
        "events_path": events_path,
        "recent_events": recent_events,
    }


class Steward:
    """Provides runtime guidance via typed methods.

    Each method corresponds to a specific scenario. Internally delegates
    to ResponseRouter which loads scenario-specific prompts.
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
        events_path: str,
        recent_events: str,
        mode: str,
        question: str,
        solver_context: str = "",
        **kwargs,
    ) -> str:
        """Solver asks for guidance. Returns free-text answer."""
        full_question = question
        if solver_context:
            full_question = f"{question}\n\nContext: {solver_context}"

        ctx = _base_context(transcript_path, events_path, recent_events, mode)
        ctx.update(kwargs)
        ctx["question"] = full_question
        return await self.router.respond_raw("ask_question", ctx)

    # ── Scenario 2: Permission check ──

    async def check_permission(
        self,
        transcript_path: str,
        events_path: str,
        recent_events: str,
        mode: str,
        tool_name: str,
        tool_input: dict,
        **kwargs,
    ) -> StewardResponse:
        """Review a restricted tool call. Returns ALLOW/DENY."""
        ctx = _base_context(transcript_path, events_path, recent_events, mode)
        ctx.update(kwargs)
        ctx["tool_name"] = tool_name
        ctx["tool_input"] = str(tool_input)
        verdict = await self.router.respond("permission", ctx)
        return _to_steward_response(verdict)

    # ── Scenario 3: Session end review ──

    async def review_session_end(
        self,
        transcript_path: str,
        events_path: str,
        recent_events: str,
        mode: str,
        result_text: str,
        stop_reason: str,
        elapsed_time: str,
        total_tool_calls: int,
        error_count: int,
        **kwargs,
    ) -> StewardResponse:
        """Review whether Solver truly finished. Returns SUCCESS/CONTINUE/ABORT."""
        ctx = _base_context(transcript_path, events_path, recent_events, mode)
        ctx.update(kwargs)
        ctx.update({
            "result_text": result_text,
            "stop_reason": stop_reason,
            "elapsed_time": elapsed_time,
            "total_tool_calls": str(total_tool_calls),
            "error_count": str(error_count),
        })
        verdict = await self.router.respond("session_end", ctx)
        return _to_steward_response(verdict)

    # ── Scenario 4: Periodic progress check ──

    async def check_progress(
        self,
        transcript_path: str,
        events_path: str,
        recent_events: str,
        mode: str,
        elapsed_time: str,
        **kwargs,
    ) -> StewardResponse:
        """Periodic progress assessment. Returns ON_TRACK/REDIRECT."""
        ctx = _base_context(transcript_path, events_path, recent_events, mode)
        ctx.update(kwargs)
        ctx["elapsed_time"] = elapsed_time
        verdict = await self.router.respond("progress_check", ctx)
        return _to_steward_response(verdict)

    # ── Scenario 6: Direction review (set_direction) ──

    async def review_direction(
        self,
        transcript_path: str,
        events_path: str,
        recent_events: str,
        mode: str,
        direction: dict,
    ) -> StewardResponse:
        """Review a direction proposal. Returns APPROVED/REDIRECT."""
        ctx = _base_context(transcript_path, events_path, recent_events, mode)
        ctx["direction_json"] = json.dumps(direction, indent=2, ensure_ascii=False)
        verdict = await self.router.respond("set_direction", ctx)
        return _to_steward_response(verdict)

    # ── Scenario 7: Direction check (diffusion) ──

    async def direction_pulse(
        self,
        transcript_path: str,
        events_path: str,
        recent_events: str,
        mode: str,
        direction: dict,
        trigger_type: str,
    ) -> StewardResponse:
        """Check if Solver's work aligns with direction. Returns ON_TRACK/REDIRECT."""
        ctx = _base_context(transcript_path, events_path, recent_events, mode)
        ctx.update({
            "direction_json": json.dumps(direction, indent=2, ensure_ascii=False),
            "trigger_type": trigger_type,
        })
        verdict = await self.router.respond("direction_pulse", ctx)
        return _to_steward_response(verdict)

    # ── Scenario 8: Start brainstorming review ──

    async def review_start_exploring(
        self,
        transcript_path: str,
        events_path: str,
        recent_events: str,
        mode: str,
        direction: dict,
        reason: str,
        **kwargs,
    ) -> StewardResponse:
        """Review Solver's request to abandon direction and brainstorm. Returns APPROVED/REDIRECT."""
        ctx = _base_context(transcript_path, events_path, recent_events, mode)
        ctx.update(kwargs)
        ctx.update({
            "direction_json": json.dumps(direction, indent=2, ensure_ascii=False),
            "reason": reason,
        })
        verdict = await self.router.respond("start_exploring", ctx)
        return _to_steward_response(verdict)
