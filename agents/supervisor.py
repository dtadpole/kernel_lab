"""Layer 2: Supervisor — orchestrates Solver + Benchmarker + Steward.

Decision loop:
  1. Solver optimizes kernel code
  2. Solver calls request_formal_bench → Supervisor dispatches Benchmarker
  3. Benchmarker runs ik:bench → returns results
  4. If improved → record, stop Solver, start new iteration
  5. If not improved → Solver retries
  6. If 4-hour time limit hit → stop and restart
  7. Every 15 min → Steward progress check
  8. 3 consecutive stuck checks (45 min) → Steward gives new guidance
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from agents.config import SystemConfig
from agents.events import (
    AskEvent,
    DefaultHandler,
    MonitorAlert,
    PermissionEvent,
    StopEvent,
    TextOutputEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from agents.runner import AgentRunner, RunResult
from agents.session_log import SessionLog
from agents.steward import Steward, StewardResponse


def _slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_]+", "_", slug)
    return slug[:60].strip("_")


@dataclass
class SupervisorState:
    phase: str = "idle"
    task: str = ""
    task_slug: str = ""
    run_tag: str = ""
    kernel: str = ""               # kernel name (matmul, fa4, etc.)
    iteration: int = 0
    turns_completed: int = 0
    error_count: int = 0
    current_action: str = ""
    started_at: datetime | None = None
    consecutive_stuck: int = 0     # consecutive stuck checks without progress
    bench_results: list[dict] = field(default_factory=list)
    verdict_history: list[dict] = field(default_factory=list)


@dataclass
class TaskResult:
    success: bool
    result_text: str
    iterations: int
    total_tool_calls: int
    total_errors: int
    elapsed_seconds: float
    verdict_history: list[dict]
    bench_results: list[dict] = field(default_factory=list)
    solver_result: RunResult | None = None


class Supervisor(DefaultHandler):
    """Orchestrates Solver + Benchmarker + Steward."""

    def __init__(
        self,
        config: SystemConfig,
        max_iterations: int = 10,
        response_prompts_dir: str | Path = "conf/agent/response_prompts",
    ):
        self.config = config
        self.max_iterations = max_iterations
        self.state = SupervisorState()

        steward_config = config.steward
        self.steward = Steward(
            prompts_dir=Path(response_prompts_dir),
            model=steward_config.model,
            storage_config=config.storage,
        )

        self._solver_runner: AgentRunner | None = None
        self._current_log: SessionLog | None = None
        self._pending_stop_event: StopEvent | None = None

    # ── Main entry point ──

    async def run_task(
        self,
        task: str,
        kernel: str = "matmul",
        run_tag: str | None = None,
    ) -> TaskResult:
        """Run the full Solver → Benchmarker → decide loop.

        Args:
            task: Task description for the Solver.
            kernel: Kernel name for ik:bench (matmul, fa4, vecadd).
            run_tag: Optional run_tag. Auto-generated if not provided.
        """
        now = datetime.now()
        task_slug = _slugify(task)
        if run_tag is None:
            run_tag = f"supervisor_run_{now.strftime('%Y%m%d_%H%M%S')}"

        self.state = SupervisorState(
            phase="planning",
            task=task,
            task_slug=task_slug,
            run_tag=run_tag,
            kernel=kernel,
            started_at=now,
        )

        solver_config = self.config.get_agent("solver")
        last_result: RunResult | None = None
        verdict: StewardResponse | None = None

        solver_prompt = self._build_initial_prompt(task, run_tag, kernel)

        for iteration in range(self.max_iterations):
            self.state.iteration = iteration
            self.state.turns_completed = 0
            self.state.error_count = 0
            self.state.consecutive_stuck = 0
            self.state.phase = "solving"

            print(f"\n{'='*60}")
            print(f"[Supervisor] Iteration {iteration} — phase: solving")
            print(f"[Supervisor] Run tag: {run_tag}")
            print(f"{'='*60}\n")

            self._solver_runner = AgentRunner(
                agent_config=solver_config,
                storage_config=self.config.storage,
                handler=self,
                monitor_config=self.config.monitor,
                steward=None,
            )

            self._pending_stop_event = None

            last_result = await self._solver_runner.run(
                prompt=solver_prompt,
                task_slug=task_slug,
            )
            self._current_log = last_result.log

            # ── Post-session Steward review ──
            # Now the Solver CLI is fully closed, safe to call Steward
            self.state.phase = "deciding"

            stop_event = self._pending_stop_event
            if stop_event:
                print(f"[Supervisor] Reviewing session end (stop_reason={stop_event.reason})")
                verdict = await self.steward.review_session_end(
                    result_text=stop_event.result_text or last_result.result_text,
                    stop_reason=stop_event.reason,
                    task=self.state.task,
                    elapsed_time=str(self._current_log.elapsed()) if self._current_log else "unknown",
                    total_tool_calls=self.state.turns_completed,
                    error_count=self.state.error_count,
                    session_summary=self._get_session_summary(),
                )
            else:
                verdict = None

            if verdict is None:
                verdict = StewardResponse(
                    action="ACCEPT", detail="", reasoning="No verdict",
                    intervention_level=1,
                )

            self.state.verdict_history.append({
                "iteration": iteration,
                "action": verdict.action,
                "detail": verdict.detail,
                "reasoning": verdict.reasoning[:200],
            })

            print(f"\n[Supervisor] Iteration {iteration} verdict: {verdict.action}")

            if verdict.action == "ACCEPT":
                self.state.phase = "done"
                break
            elif verdict.action in ("RETRY", "REJECT"):
                solver_prompt = self._build_retry_prompt(
                    task, run_tag, kernel, verdict, iteration,
                )
            else:
                self.state.phase = "done"
                break

        elapsed = (datetime.now() - self.state.started_at).total_seconds() if self.state.started_at else 0

        return TaskResult(
            success=(verdict.action == "ACCEPT") if verdict else False,
            result_text=last_result.result_text if last_result else "",
            iterations=self.state.iteration + 1,
            total_tool_calls=self.state.turns_completed,
            total_errors=self.state.error_count,
            elapsed_seconds=elapsed,
            verdict_history=self.state.verdict_history,
            bench_results=self.state.bench_results,
            solver_result=last_result,
        )

    # ── Prompt builders ──

    def _build_initial_prompt(self, task: str, run_tag: str, kernel: str) -> str:
        return f"""Run tag for this session: {run_tag}
Use this run_tag for ALL ik:exec commands (exec.run_tag={run_tag}).
Scratch directory: ~/.cuda_exec/{run_tag}/
Kernel: {kernel}

---

{task}

---

IMPORTANT: Your ik:exec trial results are preliminary — only the formal
benchmark (request_formal_bench) produces official results. Call
request_formal_bench(kernel="{kernel}", reason="...") as soon as your code
compiles and passes correctness. Do not wait for perfection — benchmark
early and often. If it shows no improvement, iterate. If it improves,
a new gem is recorded.

Keep optimizing until the formal benchmark shows improvement or you
exhaust your ideas. Do not stop after a single attempt."""

    def _build_retry_prompt(
        self, task: str, run_tag: str, kernel: str,
        verdict: StewardResponse, iteration: int,
    ) -> str:
        return f"""Run tag for this session: {run_tag}
Use this run_tag for ALL ik:exec commands (exec.run_tag={run_tag}).
Scratch directory: ~/.cuda_exec/{run_tag}/
Kernel: {kernel}

---

ITERATION {iteration + 1}: Previous attempt was not accepted.

Feedback: {verdict.detail or verdict.reasoning[:300]}

Original task: {task}

Please try a different approach. Review what was tried before and
explore an alternative optimization strategy."""

    # ── Benchmarker dispatch ──

    async def _run_benchmarker(self, kernel: str) -> RunResult:
        """Dispatch the Benchmarker agent to run ik:bench."""
        from agents.config import MonitorConfig as MC
        bench_config = self.config.get_agent("benchmarker")
        bench_runner = AgentRunner(
            agent_config=bench_config,
            storage_config=self.config.storage,
            monitor_config=MC.for_benchmarker(),
            steward=None,
        )

        print(f"\n[Supervisor] Dispatching Benchmarker for kernel={kernel}")

        result = await bench_runner.run(
            prompt=f"Run the formal benchmark for kernel={kernel}. Report the results.",
            task_slug=f"bench_{kernel}",
        )

        print(f"[Supervisor] Benchmarker completed: {result.result_text[:200]}")
        return result

    def _parse_bench_improved(self, bench_result: RunResult) -> bool:
        """Check if the benchmark result shows improvement (new gem)."""
        text = bench_result.result_text.lower()
        if "improved" in text and "true" in text:
            return True
        if "new gem" in text or "beats previous" in text:
            return True
        if "gem" in text and ("v002" in text or "v003" in text or "v004" in text):
            return True
        return False

    # ── EventHandler implementation ──

    async def on_tool_call(self, event: ToolCallEvent) -> None:
        self.state.turns_completed += 1
        self.state.current_action = event.tool_name

    async def on_tool_result(self, event: ToolResultEvent) -> None:
        if event.is_error:
            self.state.error_count += 1
        self.state.current_action = ""

    async def on_text(self, event: TextOutputEvent) -> None:
        pass

    async def on_ask(self, event: AskEvent) -> str:
        """Handle Solver questions and benchmark requests."""
        # ── request_formal_bench ──
        if event.question.startswith("REQUEST_FORMAL_BENCH:"):
            return await self._handle_bench_request(event)

        # ── Regular question → Steward ──
        return await self.steward.answer_question(
            question=event.question,
            solver_context=event.context,
            task=self.state.task,
            session_summary=self._get_session_summary(),
        )

    async def _handle_bench_request(self, event: AskEvent) -> str:
        """Solver requested a formal benchmark. Dispatch Benchmarker."""
        kernel = self.state.kernel

        # Parse kernel from the request if specified
        parts = event.question.split("kernel=")
        if len(parts) > 1:
            kernel = parts[1].split()[0].strip()

        try:
            bench_result = await self._run_benchmarker(kernel)
            improved = self._parse_bench_improved(bench_result)

            self.state.bench_results.append({
                "iteration": self.state.iteration,
                "kernel": kernel,
                "improved": improved,
                "summary": bench_result.result_text[:500],
            })

            if improved:
                return f"""BENCHMARK RESULT: IMPROVED ✓

{bench_result.result_text[:1000]}

Your optimization produced a new gem (beat previous best).
You may continue optimizing for further improvements, or
call request_formal_bench again when ready."""
            else:
                return f"""BENCHMARK RESULT: NO IMPROVEMENT

{bench_result.result_text[:1000]}

Your optimization did not beat the previous best.
Please analyze the results, try a different approach,
and call request_formal_bench when ready."""

        except Exception as e:
            return f"BENCHMARK ERROR: {e}\n\nPlease check your code compiles and runs correctly before requesting a benchmark."

    async def on_permission(self, event: PermissionEvent) -> bool:
        response = await self.steward.check_permission(
            tool_name=event.tool_name,
            tool_input=event.tool_input,
            task=self.state.task,
            recent_tool_calls=self._get_recent_tool_calls(),
        )
        return response.action == "ALLOW"

    async def on_stop(self, event: StopEvent) -> None:
        """Solver finished — store stop data for post-session Steward review.

        We do NOT call Steward here because we're still inside the Solver's
        ClaudeSDKClient context. The actual Steward review happens in run_task()
        after the Solver CLI process closes.
        """
        self._pending_stop_event = event

    async def on_monitor_alert(self, event: MonitorAlert) -> str:
        """Handle monitor alerts with stuck-count tracking."""
        log = self._current_log or (self._solver_runner.log if self._solver_runner else None)

        if event.alert_type == "hard_limit":
            return "interrupt"

        if event.alert_type == "total_timeout":
            response = await self.steward.handle_time_limit(
                elapsed_time=str(log.elapsed()) if log else "unknown",
                time_limit=str(self.config.monitor.total_timeout),
                task=self.state.task,
                recent_progress=self._get_recent_events(),
                tool_call_trend=self._get_tool_call_trend(),
            )
        elif event.alert_type in ("idle_timeout", "loop_detected"):
            self.state.consecutive_stuck += 1

            if self.state.consecutive_stuck >= 3:
                # 3 consecutive stuck checks (45 min) → force new guidance
                print(f"[Supervisor] 3 consecutive stuck alerts — forcing Steward guidance")
                response = await self.steward.handle_stuck(
                    alert_type=event.alert_type,
                    alert_details=f"{event.details} (consecutive_stuck={self.state.consecutive_stuck})",
                    task=self.state.task,
                    recent_events=self._get_recent_events(),
                    tool_call_counts=str(log.tool_call_counts()) if log else "{}",
                    elapsed_time=str(log.elapsed()) if log else "unknown",
                )
                self.state.consecutive_stuck = 0
                # Force inject even if Steward says CONTINUE
                if response.action == "CONTINUE":
                    return "inject:You have been stuck for 45 minutes. Please try a completely different optimization approach."
            else:
                response = await self.steward.handle_stuck(
                    alert_type=event.alert_type,
                    alert_details=event.details,
                    task=self.state.task,
                    recent_events=self._get_recent_events(),
                    tool_call_counts=str(log.tool_call_counts()) if log else "{}",
                    elapsed_time=str(log.elapsed()) if log else "unknown",
                )
        else:
            return "continue"

        # Map response to action
        if response.action in ("CONTINUE", "EXTEND", "ON_TRACK"):
            return "continue"
        elif response.action in ("INJECT", "DRIFTING", "REDIRECT"):
            self.state.consecutive_stuck = 0  # reset on intervention
            return f"inject:{response.detail}"
        elif response.action == "WRAP_UP":
            return "inject:Please save your current progress and summarize what you have accomplished, then finish."
        elif response.action in ("INTERRUPT", "KILL"):
            return "interrupt"
        return "continue"

    # ── Status ──

    def get_status(self) -> dict:
        return {
            "phase": self.state.phase,
            "task": self.state.task,
            "kernel": self.state.kernel,
            "run_tag": self.state.run_tag,
            "iteration": self.state.iteration,
            "turns": self.state.turns_completed,
            "errors": self.state.error_count,
            "consecutive_stuck": self.state.consecutive_stuck,
            "current_action": self.state.current_action,
            "elapsed": str(
                (datetime.now() - self.state.started_at) if self.state.started_at else "not started"
            ),
            "bench_results": self.state.bench_results,
            "verdict_history": self.state.verdict_history,
        }

    # ── Context helpers ──

    def _get_session_summary(self) -> str:
        log = self._current_log or (self._solver_runner.log if self._solver_runner else None)
        return log.to_summary() if log else "(no session data)"

    def _get_recent_tool_calls(self) -> str:
        log = self._current_log or (self._solver_runner.log if self._solver_runner else None)
        if not log:
            return "(no data)"
        lines = []
        for e in log.recent(10):
            if isinstance(e, ToolCallEvent):
                inp = str(e.tool_input)[:100]
                lines.append(f"- {e.tool_name}: {inp}")
        return "\n".join(lines) or "(no recent tool calls)"

    def _get_recent_events(self) -> str:
        log = self._current_log or (self._solver_runner.log if self._solver_runner else None)
        if not log:
            return "(no data)"
        lines = []
        for e in log.recent(10):
            ts = e.timestamp.strftime("%H:%M:%S")
            name = type(e).__name__
            if isinstance(e, ToolCallEvent):
                lines.append(f"[{ts}] {name}: {e.tool_name}")
            elif isinstance(e, ToolResultEvent):
                status = "ERR" if e.is_error else "OK"
                lines.append(f"[{ts}] {name}: {e.tool_name} [{status}]")
            elif isinstance(e, TextOutputEvent):
                lines.append(f"[{ts}] {name}: {e.text[:80]}")
            else:
                lines.append(f"[{ts}] {name}")
        return "\n".join(lines) or "(no events)"

    def _get_tool_call_trend(self) -> str:
        log = self._current_log or (self._solver_runner.log if self._solver_runner else None)
        if not log:
            return "(no data)"
        seq = log.recent_tool_sequence(20)
        return " → ".join(seq) if seq else "(no tool calls)"
