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


class Steward:
    """Provides runtime guidance via typed methods.

    Each method takes wave_context (dict) as its first argument — the shared
    context built by Workshop._get_steward_context(). Scenario-specific
    parameters follow as explicit keyword arguments.
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
        wave_context: dict,
        question: str,
    ) -> str:
        """Solver asks for guidance. Returns free-text answer."""
        variables = {
            "wave": wave_context,
            "ask_question": {"question": question},
        }
        return await self.router.respond_raw("ask_question", variables)

    # ── Scenario 2: Permission check ──

    async def check_permission(
        self,
        wave_context: dict,
        tool_name: str,
        tool_input: dict,
    ) -> StewardResponse:
        """Review a restricted tool call. Returns ALLOW/DENY."""
        variables = {
            "wave": wave_context,
            "permission": {
                "tool_name": tool_name,
                "tool_input": str(tool_input),
            },
        }
        verdict = await self.router.respond("permission", variables)
        return _to_steward_response(verdict)

    # ── Scenario 3: Session end review ──

    async def review_session_end(
        self,
        wave_context: dict,
        result_text: str,
        stop_reason: str,
        elapsed_time: str,
        total_tool_calls: int,
        error_count: int,
    ) -> StewardResponse:
        """Review whether Solver truly finished. Returns SUCCESS/CONTINUE/ABORT."""
        variables = {
            "wave": wave_context,
            "session_end": {
                "result_text": result_text,
                "stop_reason": stop_reason,
                "elapsed_time": elapsed_time,
                "total_tool_calls": str(total_tool_calls),
                "error_count": str(error_count),
            },
        }
        verdict = await self.router.respond("session_end", variables)
        return _to_steward_response(verdict)

    # ── Scenario 4: Periodic progress check ──

    async def check_progress(
        self,
        wave_context: dict,
        elapsed_time: str,
    ) -> StewardResponse:
        """Periodic progress assessment. Returns ON_TRACK/REDIRECT."""
        variables = {
            "wave": wave_context,
            "progress_check": {"elapsed_time": elapsed_time},
        }
        verdict = await self.router.respond("progress_check", variables)
        return _to_steward_response(verdict)

    # ── Scenario 5: Direction review (set_direction) ──

    async def review_direction(
        self,
        wave_context: dict,
        proposed_direction: dict,
    ) -> StewardResponse:
        """Review a direction proposal. Returns APPROVED/REDIRECT."""
        variables = {
            "wave": wave_context,
            "set_direction": {
                "proposed_direction_json": json.dumps(
                    proposed_direction, indent=2, ensure_ascii=False
                ),
            },
        }
        verdict = await self.router.respond("set_direction", variables)
        return _to_steward_response(verdict)

    # ── Scenario 6: Direction pulse ──

    async def direction_pulse(
        self,
        wave_context: dict,
        trigger_type: str,
    ) -> StewardResponse:
        """Check if Solver's work aligns with direction. Returns ON_TRACK/REDIRECT."""
        variables = {
            "wave": wave_context,
            "direction_pulse": {"trigger_type": trigger_type},
        }
        verdict = await self.router.respond("direction_pulse", variables)
        return _to_steward_response(verdict)

    # ── Scenario 7: Start exploring review ──

    async def review_start_exploring(
        self,
        wave_context: dict,
        reason: str,
    ) -> StewardResponse:
        """Review Solver's request to abandon direction. Returns APPROVED/REDIRECT."""
        variables = {
            "wave": wave_context,
            "start_exploring": {"reason": reason},
        }
        verdict = await self.router.respond("start_exploring", variables)
        return _to_steward_response(verdict)
