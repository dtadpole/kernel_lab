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

PROMPTS_DIR = Path("conf/agent/prompts")


def _load_prompt(name: str) -> str:
    """Load a prompt template from conf/agent/prompts/<name>.md."""
    path = PROMPTS_DIR / f"{name}.md"
    if path.exists():
        return path.read_text().strip()
    raise FileNotFoundError(f"Prompt template not found: {path}")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_]+", "_", slug)
    return slug[:20].strip("_")


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
        max_iterations: int = 0,  # 0 = unlimited (run until SUCCESS or hard_limit)
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

    # ── Continuous loop ──

    async def run_continuous(
        self,
        task: str,
        kernel: str = "matmul",
    ) -> None:
        """Run Solver sessions in an infinite loop until manually stopped.

        Each session is a fresh Solver with a new run_tag, picking up from
        the current state of the KB gen directory. Sessions accumulate experience —
        each new Solver gets a summary of what previous sessions tried.
        """
        session_number = 0
        session_history: list[dict] = []
        now = datetime.now()
        run_tag = f"supervisor_run_{now.strftime('%Y%m%d_%H%M%S')}"

        # Set run_tag on storage config so all agents inherit it
        self.config.storage.run_tag = run_tag

        print(f"\n{'='*60}")
        print(f"[Supervisor] CONTINUOUS MODE — will run indefinitely")
        print(f"[Supervisor] Kernel: {kernel}")
        print(f"[Supervisor] Run tag: {run_tag}")
        print(f"[Supervisor] Kill with Ctrl+C or hard_limit to stop")
        print(f"{'='*60}\n")

        while True:
            session_number += 1

            # Build prompt with history from previous sessions
            session_prompt = self._build_continuous_prompt(
                task, kernel, session_number, session_history,
            )

            print(f"\n{'='*60}")
            print(f"[Supervisor] Session {session_number} starting")
            print(f"[Supervisor] Run tag: {run_tag}")
            if session_history:
                last = session_history[-1]
                print(f"[Supervisor] Previous session: {last.get('verdict', '?')} "
                      f"— {last.get('summary', '')[:100]}")
            print(f"{'='*60}\n")

            try:
                result = await self.run_task(
                    task=session_prompt,
                    kernel=kernel,
                    run_tag=run_tag,
                )

                # Record session result
                session_record = {
                    "session": session_number,
                    "run_tag": run_tag,
                    "success": result.success,
                    "verdict": result.verdict_history[-1]["action"] if result.verdict_history else "?",
                    "iterations": result.iterations,
                    "elapsed": f"{result.elapsed_seconds:.0f}s",
                    "benchmarks": len(result.bench_results),
                    "improved": any(b["improved"] for b in result.bench_results),
                    "summary": result.result_text[:300],
                }
                session_history.append(session_record)

                print(f"\n[Supervisor] Session {session_number} ended: "
                      f"verdict={session_record['verdict']}, "
                      f"improved={session_record['improved']}, "
                      f"elapsed={session_record['elapsed']}")

                # Save session history to journal
                self._save_continuous_log(kernel, session_history)

            except KeyboardInterrupt:
                print(f"\n[Supervisor] Interrupted by user after {session_number} sessions")
                break
            except Exception as e:
                print(f"\n[Supervisor] Session {session_number} error: {e}")
                session_history.append({
                    "session": session_number,
                    "run_tag": run_tag,
                    "success": False,
                    "verdict": "ERROR",
                    "summary": str(e)[:300],
                })
                # Continue to next session despite error

    def _build_continuous_prompt(
        self, task: str, kernel: str,
        session_number: int, history: list[dict],
    ) -> str:
        """Build prompt for a new session, including history from previous sessions."""
        parts = [task]

        if history:
            parts.append("\n---\n")
            parts.append(f"## Previous Sessions ({len(history)} completed)\n")
            # Show last 5 sessions in detail, earlier ones summarized
            recent = history[-5:]
            for h in recent:
                improved = "✓ IMPROVED" if h.get("improved") else "✗ no improvement"
                parts.append(
                    f"- Session {h['session']} ({h['elapsed']}): "
                    f"verdict={h['verdict']}, {improved}\n"
                    f"  {h.get('summary', '')[:200]}\n"
                )
            parts.append(
                "\nBuild on what previous sessions learned. "
                "Do NOT repeat approaches that already failed. "
                "Try something fundamentally different if prior approaches "
                "did not produce improvement.\n"
            )

        return "\n".join(parts)

    def _save_continuous_log(self, kernel: str, history: list[dict]) -> None:
        """Save the continuous session history to the run's journal."""
        import json
        log_path = self.config.storage.journal_path / f"continuous_{kernel}.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            for h in history:
                f.write(json.dumps(h, default=str) + "\n")

    # ── Single task ──

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

        # Set run_tag on storage so journal goes under runs/<run_tag>/journal/
        # and CUDA_EXEC_RUN_TAG env var is propagated to all agents
        self.config.storage.run_tag = run_tag

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

        # Create AgentRunner once — CONTINUE will resume the same session
        self._solver_runner = AgentRunner(
            agent_config=solver_config,
            storage_config=self.config.storage,
            handler=self,
            monitor_config=self.config.monitor,
            steward=None,
        )

        solver_prompt = self._build_initial_prompt(task, run_tag, kernel)

        iteration = 0
        while True:  # run until SUCCESS or hard_limit
            if self.max_iterations > 0 and iteration >= self.max_iterations:
                break
            self.state.iteration = iteration
            self.state.consecutive_stuck = 0
            self.state.phase = "solving"

            print(f"\n{'='*60}")
            print(f"[Supervisor] Iteration {iteration} — phase: solving")
            print(f"[Supervisor] Run tag: {run_tag}")
            print(f"{'='*60}\n")

            self._pending_stop_event = None

            if iteration == 0:
                # First iteration — fresh session
                last_result = await self._solver_runner.run(
                    prompt=solver_prompt,
                    task_slug=task_slug,
                )
            else:
                # CONTINUE — resume the same session with Steward guidance
                last_result = await self._solver_runner.resume(solver_prompt)

            self._current_log = last_result.log

            # ── Post-session Steward review ──
            # Solver has stopped (end_turn), safe to call Steward.
            # On CONTINUE, we resume the same session with guidance.
            self.state.phase = "deciding"

            stop_event = self._pending_stop_event
            if stop_event:
                print(f"[Supervisor] Reviewing session end (stop_reason={stop_event.reason})")
                verdict = await self.steward.review_session_end(
                    transcript_path=self._get_transcript_path(),
                    result_text=stop_event.result_text or last_result.result_text,
                    stop_reason=stop_event.reason,
                    elapsed_time=str(self._current_log.elapsed()) if self._current_log else "unknown",
                    total_tool_calls=self.state.turns_completed,
                    error_count=self.state.error_count,
                )
            else:
                verdict = None

            if verdict is None or not verdict.action:
                # No verdict or empty action — default to CONTINUE so we don't
                # silently accept incomplete work
                verdict = StewardResponse(
                    action="CONTINUE", detail="Steward could not produce a verdict",
                    reasoning="No verdict available — continuing with fresh approach",
                    intervention_level=2,
                )

            self.state.verdict_history.append({
                "iteration": iteration,
                "action": verdict.action,
                "detail": verdict.detail,
                "reasoning": verdict.reasoning[:200],
            })

            print(f"\n[Supervisor] Iteration {iteration} verdict: {verdict.action}")

            if verdict.action == "SUCCESS":
                self.state.phase = "done"
                break
            elif verdict.action == "ABORT":
                self.state.phase = "done"
                break
            elif verdict.action == "CONTINUE":
                solver_prompt = self._build_continue_prompt(verdict)
            else:
                # Unknown verdict — treat as CONTINUE
                solver_prompt = self._build_continue_prompt(verdict)

            iteration += 1

        elapsed = (datetime.now() - self.state.started_at).total_seconds() if self.state.started_at else 0

        return TaskResult(
            success=(verdict.action == "SUCCESS") if verdict else False,
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
        template = _load_prompt("supervisor_initial")
        result = template.format(run_tag=run_tag, kernel=kernel, task=task)
        return result.replace("<run_tag>", run_tag)

    def _build_continue_prompt(self, verdict: StewardResponse) -> str:
        """Build the resume prompt for CONTINUE — injected into the same session."""
        guidance = verdict.detail or verdict.reasoning[:500]
        template = _load_prompt("supervisor_continue")
        return template.format(guidance=guidance)

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
            transcript_path=self._get_transcript_path(),
            question=event.question,
            solver_context=event.context,
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
                template = _load_prompt("supervisor_bench_improved")
            else:
                template = _load_prompt("supervisor_bench_no_improvement")
            return template.format(bench_result_text=bench_result.result_text[:1000])

        except Exception as e:
            return f"BENCHMARK ERROR: {e}\n\nPlease check your code compiles and runs correctly before requesting a benchmark."

    async def on_permission(self, event: PermissionEvent) -> bool:
        response = await self.steward.check_permission(
            transcript_path=self._get_transcript_path(),
            tool_name=event.tool_name,
            tool_input=event.tool_input,
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

        tp = self._get_transcript_path()
        elapsed = str(log.elapsed()) if log else "unknown"

        if event.alert_type == "total_timeout":
            response = await self.steward.handle_time_limit(
                transcript_path=tp,
                elapsed_time=elapsed,
                time_limit=str(self.config.monitor.total_timeout),
            )
        elif event.alert_type in ("idle_timeout", "loop_detected"):
            self.state.consecutive_stuck += 1

            if self.state.consecutive_stuck >= 3:
                # 3 × 10 min = 30 min no progress → force interrupt + new guidance
                print(f"[Supervisor] 3 consecutive stuck alerts (30 min) — forcing interrupt")
                response = await self.steward.handle_stuck(
                    transcript_path=tp,
                    alert_type=event.alert_type,
                    alert_details=f"{event.details} (consecutive_stuck={self.state.consecutive_stuck})",
                    elapsed_time=elapsed,
                )
                self.state.consecutive_stuck = 0
                if response.action == "CONTINUE":
                    return "interrupt"
            else:
                response = await self.steward.handle_stuck(
                    transcript_path=tp,
                    alert_type=event.alert_type,
                    alert_details=event.details,
                    elapsed_time=elapsed,
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

    def _get_transcript_path(self) -> str:
        """Return the path to the current Solver's transcript.md."""
        if self._solver_runner and self._solver_runner._storage:
            return str(self._solver_runner._storage.transcript_path)
        return "(no transcript available)"
