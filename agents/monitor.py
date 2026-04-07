"""Parallel watchdog that monitors agent health.

Runs as an independent asyncio task alongside the agent.
Detects: idle timeout, total timeout, tool call loops.
Can interrupt or inject guidance into the running Solver.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from agents.config import MonitorConfig
from agents.events import EventHandler, InjectEvent, MonitorAlert, TextOutputEvent
from agents.session_log import SessionLog

if TYPE_CHECKING:
    from agents.runner import AgentRunner


class AgentMonitor:
    """Async watchdog — runs alongside the agent, checks SessionLog periodically.

    Has a reference to the AgentRunner so it can interrupt or inject guidance.
    """

    def __init__(
        self,
        log: SessionLog,
        handler: EventHandler,
        runner: AgentRunner | None = None,
        config: MonitorConfig | None = None,
    ):
        self.log = log
        self.handler = handler
        self.runner = runner
        self.config = config or MonitorConfig()
        self._running = False
        self._task: asyncio.Task | None = None
        self._pending_inject: str | None = None
        self._last_progress_check: float = 0.0  # elapsed time at last progress check

    async def start(self) -> None:
        """Start the monitor as a background task."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the monitor."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.config.check_interval)
            if not self._running:
                break
            alert = self._check_health()
            if alert:
                self.log.append(alert)
                action = await self.handler.on_monitor_alert(alert)
                await self._execute_action(action, alert)

    async def _execute_action(self, action: str, alert: MonitorAlert) -> None:
        """Execute the action returned by the Supervisor."""
        if action == "continue":
            # Log that we checked and decided to continue
            return

        elif action == "terminate":
            # Hard kill — terminate the Solver process, Supervisor will restart
            event = TextOutputEvent(
                text=f"[monitor] TERMINATE — {alert.alert_type}: {alert.details}"
            )
            self.log.append(event)
            if self.runner:
                await self.runner.terminate()
            self._running = False

        elif action == "interrupt":
            # Soft interrupt — ask Solver to stop gracefully
            event = TextOutputEvent(
                text=f"[monitor] INTERRUPT — {alert.alert_type}: {alert.details}"
            )
            self.log.append(event)
            if self.runner:
                await self.runner.interrupt()
            self._running = False

        elif action.startswith("inject:"):
            # Extract guidance, log it.
            # Inject is stored and will be delivered to the Solver via the
            # next PreToolUse hook as additionalContext. We do NOT interrupt
            # the Solver — inject is a soft nudge, not a kill.
            guidance = action[len("inject:"):]
            event = InjectEvent(
                guidance=guidance,
                source=f"monitor_{alert.alert_type}",
            )
            self.log.append(event)
            # Store for delivery via next hook callback
            self._pending_inject = guidance

        else:
            # Unknown action — log it
            event = TextOutputEvent(
                text=f"[monitor] unknown action: {action}"
            )
            self.log.append(event)

    def consume_pending_inject(self) -> str | None:
        """Consume and return any pending inject guidance. Called by runner's PreToolUse hook."""
        guidance = self._pending_inject
        self._pending_inject = None
        return guidance

    def _check_health(self) -> MonitorAlert | None:
        """Check for anomalies. Returns an alert or None."""
        elapsed = self.log.elapsed().total_seconds()

        # Hard limit — non-negotiable, always interrupt
        if elapsed > self.config.hard_limit:
            return MonitorAlert(
                alert_type="hard_limit",
                details=f"Session hit hard limit: {elapsed:.0f}s (max: {self.config.hard_limit}s)",
            )

        # Total timeout — triggers Steward (can be extended)
        if elapsed > self.config.total_timeout:
            return MonitorAlert(
                alert_type="total_timeout",
                details=f"Session running for {elapsed:.0f}s (limit: {self.config.total_timeout}s)",
            )

        # Idle timeout
        idle = self.log.last_event_age().total_seconds()
        if idle > self.config.idle_timeout:
            return MonitorAlert(
                alert_type="idle_timeout",
                details=f"No activity for {idle:.0f}s (limit: {self.config.idle_timeout}s)",
            )

        # Loop detection
        seq = self.log.recent_tool_sequence(self.config.loop_threshold)
        if len(seq) >= self.config.loop_threshold and len(set(seq)) == 1:
            return MonitorAlert(
                alert_type="loop_detected",
                details=f"Tool '{seq[0]}' called {len(seq)} times consecutively",
            )

        # Periodic progress check
        if (self.config.progress_check_interval > 0
                and elapsed - self._last_progress_check >= self.config.progress_check_interval):
            self._last_progress_check = elapsed
            return MonitorAlert(
                alert_type="progress_check",
                details=f"Periodic progress check at {elapsed:.0f}s",
            )

        return None
