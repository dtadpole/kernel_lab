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

import asyncio
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
    gpu: int = 4                   # GPU index for exec/trial/bench
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
        gpu: int = 4,
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
        print(f"[Supervisor] GPU: {gpu}")
        print(f"[Supervisor] Run tag: {run_tag}")
        print(f"[Supervisor] Kill with Ctrl+C or hard_limit to stop")
        print(f"{'='*60}\n")

        consecutive_errors = 0
        max_consecutive_errors = 5
        error_cooldown_seconds = 60
        recent_session_starts: list[float] = []  # timestamps
        max_sessions_per_minute = 10

        while True:
            session_number += 1

            # Rate limit: prevent crash loops from creating millions of sessions
            import time
            now = time.time()
            recent_session_starts.append(now)
            recent_session_starts = [t for t in recent_session_starts if now - t < 60]
            if len(recent_session_starts) > max_sessions_per_minute:
                print(f"[Supervisor] CRASH LOOP DETECTED: {len(recent_session_starts)} sessions in 60s. "
                      f"Cooling down {error_cooldown_seconds}s...")
                await asyncio.sleep(error_cooldown_seconds)
                recent_session_starts.clear()
                consecutive_errors = 0

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
                    gpu=gpu,
                    run_tag=run_tag,
                )

                consecutive_errors = 0  # reset on success

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
            except BaseException as e:
                consecutive_errors += 1
                print(f"\n[Supervisor] Session {session_number} error ({consecutive_errors}/{max_consecutive_errors}): "
                      f"{type(e).__name__}: {e}")
                session_history.append({
                    "session": session_number,
                    "run_tag": run_tag,
                    "success": False,
                    "verdict": "ERROR",
                    "elapsed": "0s",
                    "improved": False,
                    "summary": f"{type(e).__name__}: {str(e)[:280]}",
                })

                if consecutive_errors >= max_consecutive_errors:
                    print(f"[Supervisor] {max_consecutive_errors} consecutive errors — "
                          f"cooling down {error_cooldown_seconds}s before retry")
                    await asyncio.sleep(error_cooldown_seconds)
                    consecutive_errors = 0

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
        gpu: int = 4,
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

        # Create run directories: KB run dir + scratch dir
        kb_run_dir = Path(self.config.storage.kb_root).expanduser() / "runs" / run_tag
        scratch_dir = Path.home() / ".cuda_exec" / run_tag
        kb_run_dir.mkdir(parents=True, exist_ok=True)
        scratch_dir.mkdir(parents=True, exist_ok=True)

        self.state = SupervisorState(
            phase="planning",
            task=task,
            task_slug=task_slug,
            run_tag=run_tag,
            kernel=kernel,
            gpu=gpu,
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

        solver_prompt = self._build_initial_prompt(task, run_tag, kernel, gpu)

        iteration = 0
        consecutive_quick_exits = 0
        max_quick_exits = 3
        min_session_seconds = 30  # sessions shorter than this are "quick exits"

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

            # Detect quick exits (Solver stops almost immediately)
            session_duration = last_result.log.elapsed().total_seconds() if last_result.log else 0
            if iteration > 0 and session_duration < min_session_seconds:
                consecutive_quick_exits += 1
                print(f"[Supervisor] Quick exit detected ({session_duration:.0f}s < {min_session_seconds}s) "
                      f"— {consecutive_quick_exits}/{max_quick_exits}")
                if consecutive_quick_exits >= max_quick_exits:
                    print(f"[Supervisor] {max_quick_exits} consecutive quick exits — aborting run_task")
                    break
            else:
                consecutive_quick_exits = 0

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

    def _build_initial_prompt(self, task: str, run_tag: str, kernel: str, gpu: int = 4) -> str:
        template = _load_prompt("supervisor_initial")
        result = template.format(run_tag=run_tag, kernel=kernel, task=task, gpu=gpu)
        return result.replace("<run_tag>", run_tag)

    def _build_continue_prompt(self, verdict: StewardResponse) -> str:
        """Build the resume prompt for CONTINUE — injected into the same session."""
        guidance = verdict.detail or verdict.reasoning[:500]
        template = _load_prompt("supervisor_continue")
        return template.format(guidance=guidance)

    # ── Benchmarker dispatch ──

    async def _run_benchmarker(
        self,
        kernel: str,
        arch: str = "",
        impls: str = "",
        timeout: int = 0,
    ) -> RunResult:
        """Run formal benchmark directly via .venv/bin/python subprocess.

        No Benchmarker agent — just run formal.py and parse JSON output.
        GPU and run_tag are always overridden by Supervisor (not from Solver).
        """
        import asyncio
        import subprocess

        gpu = self.state.gpu       # Supervisor controls GPU
        run_tag = self.state.run_tag
        cwd = self.config.defaults.get("cwd", str(Path.cwd()))
        venv_python = str(Path(cwd) / ".venv" / "bin" / "python")

        cmd = [
            venv_python, "-m", "cuda_exec.formal",
            f"bench.kernel={kernel}",
            f"bench.gpu={gpu}",
            f"bench.run_tag={run_tag}",
        ]
        if arch:
            cmd.append(f"bench.arch={arch}")
        if impls:
            cmd.append(f"bench.impls=[{impls}]")
        if timeout > 0:
            cmd.append(f"bench.timeout={timeout}")

        print(f"\n[Supervisor] Running formal bench: {' '.join(cmd)}")

        # Run in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        proc_result = await loop.run_in_executor(None, lambda: subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=1200,  # 20 min max
        ))

        # stdout = JSON result, stderr = Markdown table + source paths
        table_output = proc_result.stderr.strip()
        json_output = proc_result.stdout.strip()

        print(f"[Supervisor] Bench exit code: {proc_result.returncode}")
        if table_output:
            print(f"[Supervisor] Bench table:\n{table_output[:500]}")

        # Parse JSON for structured data (gems, improved, etc.)
        bench_data = {}
        if json_output:
            try:
                bench_data = json.loads(json_output)
            except json.JSONDecodeError:
                pass

        # result_text = stderr table (human-readable, returned to Solver)
        result_text = table_output or "(no output)"
        if proc_result.returncode != 0:
            result_text = f"BENCHMARK FAILED (exit code {proc_result.returncode})\n\n{result_text}"

        result = RunResult(result_text=result_text)
        result.usage = {"bench_data": bench_data}
        return result

    def _check_correctness_failure(self, bench_data: dict, result_text: str) -> bool:
        """Check if any gen-* impl has correctness failures.

        Primary: structured JSON from formal.py stdout.
        Fallback: ✗ character in stderr table.
        """
        summary = bench_data.get("summary", {})
        for impl_slug, impl_info in summary.get("impls", {}).items():
            if impl_slug.startswith("gen-"):
                for cfg_slug, cfg_info in impl_info.get("configs", {}).items():
                    if cfg_info.get("correct") is False:
                        return True
        # Fallback: check stderr table
        if "✗" in result_text:
            return True
        return False

    def _parse_bench_improved(self, bench_result: RunResult) -> bool:
        """Check if the benchmark result shows improvement (new gem)."""
        # Check structured data first
        bench_data = bench_result.usage.get("bench_data", {})
        if bench_data.get("improved"):
            return True
        gems = bench_data.get("gems", {})
        if gems:
            return True
        # Fallback to text parsing
        text = bench_result.result_text.lower()
        if "improved: true" in text or "improved=true" in text:
            return True
        if "new gem" in text:
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

        # ── submit_bench_reflection ──
        if event.question == "SUBMIT_BENCH_REFLECTION":
            return self._handle_bench_reflection(event.context)

        # ── Regular question → Steward ──
        return await self.steward.answer_question(
            transcript_path=self._get_transcript_path(),
            question=event.question,
            solver_context=event.context,
        )

    def _handle_bench_reflection(self, context: str) -> str:
        """Save bench reflection and optional gem notes via cuda_exec.reflection."""
        from cuda_exec.reflection import save_bench_reflection

        try:
            data = json.loads(context)
        except json.JSONDecodeError:
            return "Invalid context — expected JSON with reflection_md field."

        gem_id = data.get("gem_id", "")
        gem_notes_md = data.get("gem_notes_md", "")
        reflection_md = data.get("reflection_md", "")

        if not reflection_md:
            return "Missing reflection_md. Please write your reflection."

        # Get latest bench timestamp from state
        bench_ts = ""
        if self.state.bench_results:
            last = self.state.bench_results[-1]
            bench_data = last.get("bench_data", {})
            bench_ts = bench_data.get("bench_timestamp", "")
        if not bench_ts:
            # Fallback: use current time
            bench_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        result = save_bench_reflection(
            run_tag=self.state.run_tag,
            bench_ts=bench_ts,
            kernel=self.state.kernel,
            reflection_md=reflection_md,
            gem_id=gem_id,
            gem_notes_md=gem_notes_md,
        )

        files = result.get("files_written", [])
        print(f"[Supervisor] Reflection saved: {files}")

        parts = [f"Reflection saved to {result.get('reflection_path', '?')}."]
        if result.get("gem_notes_path"):
            parts.append(f"Gem notes saved to {result['gem_notes_path']}.")
        if result.get("gem_error"):
            parts.append(f"Warning: {result['gem_error']}")
        parts.append("Continue optimizing.")

        return " ".join(parts)

    async def _handle_bench_request(self, event: AskEvent) -> str:
        """Solver requested a formal benchmark. Run via subprocess."""
        kernel = self.state.kernel

        # Parse kernel from the request string
        query = event.question
        kernel_m = re.search(r"kernel=(\S+)", query)
        if kernel_m:
            kernel = kernel_m.group(1)

        try:
            # Supervisor controls GPU, impls (always "all"), and run_tag
            bench_result = await self._run_benchmarker(kernel)
            improved = self._parse_bench_improved(bench_result)

            bench_data = bench_result.usage.get("bench_data", {})
            bench_ts = bench_data.get("bench_timestamp", "")

            self.state.bench_results.append({
                "iteration": self.state.iteration,
                "kernel": kernel,
                "improved": improved,
                "summary": bench_result.result_text[:500],
                "bench_data": bench_data,
            })

            has_correctness_failure = self._check_correctness_failure(
                bench_data, bench_result.result_text
            )

            # Build template variables
            run_tag = self.state.run_tag
            kb_root = Path.home() / "kernel_lab_kb"
            impls_dir = kb_root / "runs" / run_tag / "impls" / bench_ts

            # Find gem info
            gems = bench_data.get("gems", {})
            gem_id = ""
            n_improved = 0
            for slug, info in gems.items():
                ver = info.get("version", "?")
                n_improved = len(info.get("improved_configs", []))
                gem_id = f"{slug}/v{ver:03d}"

            # Find previous/best gem path
            import glob
            gem_pattern = str(kb_root / "runs" / run_tag / "gems" / kernel / "gen-cuda" / "v*")
            gem_matches = sorted(glob.glob(gem_pattern))
            if improved and len(gem_matches) >= 2:
                prev_gem_path = gem_matches[-2]
            elif gem_matches:
                prev_gem_path = gem_matches[-1]
            else:
                prev_gem_path = "(none — first iteration)"

            template_vars = {
                "bench_result_text": bench_result.result_text,
                "gem_id": gem_id,
                "n_improved": n_improved,
                "impls_dir": str(impls_dir),
                "kernel": kernel,
                "prev_gem_path": prev_gem_path,
                "best_gem_path": gem_matches[-1] if gem_matches else "(none)",
            }

            if improved:
                template = _load_prompt("supervisor_bench_improved")
            else:
                template = _load_prompt("supervisor_bench_no_improvement")

            # Escape braces in bench_result_text to prevent format() errors
            # (formal.py output may contain { } in log lines)
            safe_bench_text = bench_result.result_text.replace("{", "{{").replace("}", "}}")
            template_vars["bench_result_text"] = safe_bench_text
            result_text = template.format(**template_vars)
            # Unescape for final output
            result_text = result_text.replace("{{", "{").replace("}}", "}")

            if has_correctness_failure:
                correctness_warning = (
                    "⚠️ CORRECTNESS FAILURE DETECTED ⚠️\n\n"
                    "One or more configs show ✗ in the benchmark table above.\n"
                    "✗ means your kernel produced WRONG RESULTS vs the golden reference.\n"
                    "You MUST fix correctness before doing any performance optimization.\n"
                    "Do NOT request another formal_bench until ALL configs show ✓.\n\n"
                )
                result_text = correctness_warning + result_text

            return result_text

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
            return "terminate"

        tp = self._get_transcript_path()
        elapsed = str(log.elapsed()) if log else "unknown"

        if event.alert_type == "total_timeout":
            # Code decision: auto-continue until hard_limit
            print(f"[Supervisor] Time limit at {elapsed} — auto-continuing")
            return "continue"
        elif event.alert_type == "progress_check":
            # Code decision: just log, no Steward needed
            print(f"[Supervisor] Progress check at {elapsed} — heartbeat OK")
            return "continue"
        elif event.alert_type in ("idle_timeout", "loop_detected"):
            self.state.consecutive_stuck += 1

            if self.state.consecutive_stuck >= 3:
                print(f"[Supervisor] 3 consecutive stuck alerts — forcing interrupt")
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
            "gpu": self.state.gpu,
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
