"""Parallel watchdog that monitors agent health.

Runs as an independent asyncio task alongside the agent.
Detects: idle timeout, total timeout, tool call loops.
"""

from __future__ import annotations

import asyncio

from agents.config import MonitorConfig
from agents.events import EventHandler, MonitorAlert
from agents.session_log import SessionLog


class AgentMonitor:
    """Async watchdog — runs alongside the agent, checks SessionLog periodically."""

    def __init__(
        self,
        log: SessionLog,
        handler: EventHandler,
        config: MonitorConfig | None = None,
    ):
        self.log = log
        self.handler = handler
        self.config = config or MonitorConfig()
        self._running = False
        self._task: asyncio.Task | None = None

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
                if action == "interrupt":
                    self._running = False
                    break

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

        return None
