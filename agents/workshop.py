"""Layer 2: Workshop — orchestrates Solver + Benchmarker + Steward.

Decision loop:
  1. Solver optimizes kernel code
  2. Solver calls request_formal_bench → Workshop dispatches Benchmarker
  3. Benchmarker runs ik:bench → returns results
  4. If improved → record, stop Solver, start new wave
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

import os
import subprocess

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
from agents.api_server import WorkshopAPIServer
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


async def _run_subprocess_async(
    cmd: list[str],
    cwd: str,
    log_path: Path,
    timeout: int = 1800,
) -> subprocess.CompletedProcess:
    """Run a subprocess asynchronously with real-time stdout/stderr logging.

    Uses asyncio.create_subprocess_exec — no threads. stdout/stderr are
    streamed line-by-line to log_path, prefixed with [stdout]/[stderr].

    Returns CompletedProcess with captured stdout and stderr (unprefixed).
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    async def _stream(stream: asyncio.StreamReader, prefix: str, collector: list[str]):
        async for raw_line in stream:
            line = raw_line.decode("utf-8", errors="replace")
            collector.append(line)

    log_file = open(log_path, "w")

    async def _stream_to_log(stream: asyncio.StreamReader, prefix: str, collector: list[str]):
        async for raw_line in stream:
            line = raw_line.decode("utf-8", errors="replace")
            log_file.write(f"[{prefix}] {line}")
            log_file.flush()
            collector.append(line)

    try:
        await asyncio.wait_for(
            asyncio.gather(
                _stream_to_log(proc.stdout, "stdout", stdout_lines),
                _stream_to_log(proc.stderr, "stderr", stderr_lines),
            ),
            timeout=timeout,
        )
        await proc.wait()
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        log_file.close()
        raise subprocess.TimeoutExpired(cmd, timeout)

    log_file.close()

    return subprocess.CompletedProcess(
        cmd, proc.returncode, "".join(stdout_lines), "".join(stderr_lines)
    )


def _slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_]+", "_", slug)
    return slug[:20].strip("_")


@dataclass
class WorkshopState:
    phase: str = "idle"
    task: str = ""
    task_slug: str = ""
    run_tag: str = ""
    kernel: str = ""               # kernel name (matmul, fa4, etc.)
    gpu: int = 4                   # GPU index for exec/trial/bench
    wave: int = 0
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
    waves: int
    total_tool_calls: int
    total_errors: int
    elapsed_seconds: float
    verdict_history: list[dict]
    bench_results: list[dict] = field(default_factory=list)
    solver_result: RunResult | None = None


class Workshop(DefaultHandler):
    """Orchestrates Solver + Benchmarker + Steward."""

    def __init__(
        self,
        config: SystemConfig,
        response_prompts_dir: str | Path = "conf/agent/response_prompts",
    ):
        self.config = config
        self.state = WorkshopState()

        steward_config = config.steward
        self.steward = Steward(
            prompts_dir=Path(response_prompts_dir),
            model=steward_config.model,
            storage_config=config.storage,
        )

        self._solver_runner: AgentRunner | None = None
        self._current_log: SessionLog | None = None
        self._pending_stop_event: StopEvent | None = None
        self._api_server: WorkshopAPIServer | None = None

    # ── Main loop ──

    async def run_continuous(
        self,
        task: str,
        kernel: str = "matmul",
        gpu: int = 4,
    ) -> None:
        """Run forever: Task → Wave → Session.

        Task: one optimization job, runs until external kill.
        Wave: one subprocess (one PID). Fresh AgentRunner per Wave.
        Session: one dialogue round within a Wave (query → ResultMessage).
        """
        import time as _time

        now = datetime.now()
        run_tag = f"workshop_run_{now.strftime('%Y%m%d_%H%M%S')}"
        task_slug = _slugify(task)

        self.config.storage.run_tag = run_tag

        self.state = WorkshopState(
            phase="starting",
            task=task,
            task_slug=task_slug,
            run_tag=run_tag,
            kernel=kernel,
            gpu=gpu,
            started_at=now,
        )

        # Create run directories
        kb_run_dir = Path(self.config.storage.kb_root).expanduser() / "runs" / run_tag
        scratch_dir = Path.home() / ".cuda_exec" / run_tag
        kb_run_dir.mkdir(parents=True, exist_ok=True)
        scratch_dir.mkdir(parents=True, exist_ok=True)

        # Start API server
        self._api_server = WorkshopAPIServer(self)
        try:
            api_port = await self._api_server.start()
        except Exception as e:
            print(f"[Workshop] API server failed to start: {e}")
            self._api_server = None

        print(f"\n{'='*60}")
        print(f"[Workshop] Task: {kernel} on GPU {gpu}")
        print(f"[Workshop] Run tag: {run_tag}")
        if self._api_server:
            print(f"[Workshop] API: http://127.0.0.1:{api_port}")
        print(f"[Workshop] Kill with Ctrl+C to stop")
        print(f"{'='*60}\n")

        wave = 0
        wave_history: list[dict] = []
        consecutive_errors = 0
        max_consecutive_errors = 5
        recent_wave_starts: list[float] = []

        while True:  # Task loop — runs forever
            # ── Crash loop protection ──
            now_ts = _time.time()
            recent_wave_starts.append(now_ts)
            recent_wave_starts = [t for t in recent_wave_starts if now_ts - t < 60]
            if len(recent_wave_starts) > 10:
                print(f"[Workshop] CRASH LOOP: {len(recent_wave_starts)} waves in 60s — cooling down 60s")
                await asyncio.sleep(60)
                recent_wave_starts.clear()
                consecutive_errors = 0

            # ── Create fresh Runner for this Wave ──
            solver_config = self.config.get_agent("solver")
            self._solver_runner = AgentRunner(
                agent_config=solver_config,
                storage_config=self.config.storage,
                handler=self,
                monitor_config=self.config.monitor,
                wave=wave,
                task_slug=task_slug,
            )

            self.state.wave = wave
            self.state.phase = "solving"
            self._gem_produced = False
            self._reflection_received = False

            wave_start = datetime.now()
            initial_prompt = self._build_initial_prompt(task, run_tag, kernel, gpu)
            if wave_history:
                initial_prompt += self._build_wave_history_prompt(wave_history)

            print(f"\n{'='*60}")
            print(f"[Workshop] Wave {wave} starting")
            print(f"{'='*60}\n")

            try:
                await self._solver_runner.start(initial_prompt)
                session = 0

                while True:  # Session loop within Wave
                    self.state.phase = "solving"
                    result = await self._solver_runner.run_until_result()
                    self._current_log = result.log

                    # ── Gem + reflection → SUCCESS → end Wave ──
                    if self._gem_produced and self._reflection_received:
                        print(f"[Workshop] Wave {wave} — gem produced + reflection received → SUCCESS")
                        await self._solver_runner.send_message(
                            "SUCCESS. Thank you for your contribution. This wave is complete."
                        )
                        await asyncio.sleep(10)
                        self.state.verdict_history.append({
                            "wave": wave, "session": session,
                            "action": "SUCCESS", "detail": "gem + reflection",
                        })
                        break

                    # ── Steward review ──
                    self.state.phase = "deciding"
                    stop_event = self._pending_stop_event
                    if stop_event:
                        print(f"[Workshop] Wave {wave} session {session} ended (stop_reason={stop_event.reason})")
                        verdict = await self.steward.review_session_end(
                            transcript_path=self._get_transcript_path(),
                            result_text=stop_event.result_text or result.result_text,
                            stop_reason=stop_event.reason,
                            elapsed_time=str(self._current_log.elapsed()) if self._current_log else "unknown",
                            total_tool_calls=self.state.turns_completed,
                            error_count=self.state.error_count,
                        )
                    else:
                        verdict = None

                    if verdict is None or not verdict.action:
                        verdict = StewardResponse(
                            action="CONTINUE", detail="Steward could not produce a verdict",
                            reasoning="No verdict available", intervention_level=2,
                        )

                    self.state.verdict_history.append({
                        "wave": wave, "session": session,
                        "action": verdict.action,
                        "detail": verdict.detail,
                        "reasoning": verdict.reasoning[:200],
                    })

                    print(f"[Workshop] Wave {wave} session {session} verdict: {verdict.action}")

                    if verdict.action == "ABORT":
                        break  # End Wave

                    # ── CONTINUE → new Session in same Wave ──
                    session += 1
                    self._pending_stop_event = None
                    continue_prompt = self._build_continue_prompt(verdict)
                    await self._solver_runner.send_message(continue_prompt)

                consecutive_errors = 0

            except KeyboardInterrupt:
                print(f"\n[Workshop] Interrupted by user at wave {wave}")
                await self._solver_runner.stop()
                await self._stop_api_server()
                break

            except Exception as e:
                consecutive_errors += 1
                print(f"[Workshop] Wave {wave} error ({consecutive_errors}/{max_consecutive_errors}): "
                      f"{type(e).__name__}: {e}")
                if consecutive_errors >= max_consecutive_errors:
                    print(f"[Workshop] {max_consecutive_errors} consecutive errors — cooling down 60s")
                    await asyncio.sleep(60)
                    consecutive_errors = 0

            finally:
                # Always clean up subprocess
                try:
                    await self._solver_runner.stop()
                except Exception as e:
                    print(f"[Workshop] Wave {wave} cleanup error: {e}")

            # ── Record wave result ──
            elapsed = (datetime.now() - wave_start).total_seconds()
            wave_record = {
                "wave": wave,
                "elapsed": f"{elapsed:.0f}s",
                "benchmarks": len(self.state.bench_results),
                "improved": self._gem_produced,
                "verdict": self.state.verdict_history[-1]["action"] if self.state.verdict_history else "?",
            }
            wave_history.append(wave_record)
            self._save_wave_log(kernel, wave_history)

            print(f"[Workshop] Wave {wave} ended ({wave_record['elapsed']}, "
                  f"verdict={wave_record['verdict']}, improved={wave_record['improved']})")

            wave += 1

    # ── Helpers ──

    def _build_wave_history_prompt(self, history: list[dict]) -> str:
        """Append previous wave summaries to the initial prompt."""
        parts = ["\n---\n", f"## Previous Waves ({len(history)} completed)\n"]
        for h in history[-5:]:
            improved = "✓ IMPROVED" if h.get("improved") else "✗ no improvement"
            parts.append(
                f"- Wave {h['wave']} ({h['elapsed']}): "
                f"verdict={h['verdict']}, {improved}\n"
            )
        parts.append(
            "\nBuild on what previous waves learned. "
            "Do NOT repeat approaches that already failed.\n"
        )
        return "\n".join(parts)

    def _save_wave_log(self, kernel: str, history: list[dict]) -> None:
        """Save wave history to the run's journal."""
        log_path = self.config.storage.journal_path / f"waves_{kernel}.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            for h in history:
                f.write(json.dumps(h, default=str) + "\n")

    async def _stop_api_server(self) -> None:
        """Stop the API server if running."""
        if self._api_server:
            await self._api_server.stop()
            self._api_server = None

    # ── Prompt builders ──

    def _build_initial_prompt(self, task: str, run_tag: str, kernel: str, gpu: int = 4) -> str:
        template = _load_prompt("workshop_initial")
        result = template.format(run_tag=run_tag, kernel=kernel, task=task, gpu=gpu)
        return result.replace("<run_tag>", run_tag)

    def _build_continue_prompt(self, verdict: StewardResponse) -> str:
        """Build the resume prompt for CONTINUE — injected into the same session."""
        guidance = verdict.detail or verdict.reasoning[:500]
        template = _load_prompt("workshop_continue")
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
        GPU and run_tag are always overridden by Workshop (not from Solver).
        """
        gpu = self.state.gpu       # Workshop controls GPU
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

        print(f"\n[Workshop] Running formal bench: {' '.join(cmd)}")

        # Run async subprocess — no threads
        bench_log_dir = Path.home() / ".cuda_exec" / run_tag
        bench_log_dir.mkdir(parents=True, exist_ok=True)
        bench_log_path = bench_log_dir / "formal_bench.log"

        proc_result = await _run_subprocess_async(cmd, cwd, bench_log_path, timeout=1800)

        # stdout = JSON result, stderr = Markdown table + source paths
        table_output = proc_result.stderr.strip()
        json_output = proc_result.stdout.strip()

        print(f"[Workshop] Bench exit code: {proc_result.returncode}")
        if table_output:
            print(f"[Workshop] Bench table:\n{table_output[:500]}")

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
        print(f"[Workshop] Reflection saved: {files}")

        self._reflection_received = True

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
            # Workshop controls GPU, impls (always "all"), and run_tag
            bench_result = await self._run_benchmarker(kernel)
            improved = self._parse_bench_improved(bench_result)

            bench_data = bench_result.usage.get("bench_data", {})
            bench_ts = bench_data.get("bench_timestamp", "")

            self.state.bench_results.append({
                "wave": self.state.wave,
                "kernel": kernel,
                "improved": improved,
                "summary": bench_result.result_text[:500],
                "bench_data": bench_data,
            })

            if improved:
                self._gem_produced = True

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
                prev_gem_path = "(none — first wave)"

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
                template = _load_prompt("workshop_bench_improved")
            else:
                template = _load_prompt("workshop_bench_no_improvement")

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
            print(f"[Workshop] Time limit at {elapsed} — auto-continuing")
            return "continue"
        elif event.alert_type == "progress_check":
            # Code decision: just log, no Steward needed
            print(f"[Workshop] Progress check at {elapsed} — heartbeat OK")
            return "continue"
        elif event.alert_type in ("idle_timeout", "loop_detected"):
            self.state.consecutive_stuck += 1

            if self.state.consecutive_stuck >= 3:
                print(f"[Workshop] 3 consecutive stuck alerts — forcing interrupt")
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
            "wave": self.state.wave,
            "turns": self.state.turns_completed,
            "errors": self.state.error_count,
            "consecutive_stuck": self.state.consecutive_stuck,
            "current_action": self.state.current_action,
            "started_at": self.state.started_at.isoformat() if self.state.started_at else None,
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
