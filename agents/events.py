"""Event types and handler protocol for the Agent Runner.

Events are emitted by SDK hooks and message stream processing.
Upper layers (Supervisor) implement EventHandler to react to them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol
from uuid import uuid4


# ── Base ──

@dataclass
class AgentEvent:
    """Base class for all agent events."""
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: uuid4().hex[:8])

    def to_dict(self) -> dict:
        d = {
            "ts": self.timestamp.isoformat(),
            "id": self.event_id,
            "type": type(self).__name__,
        }
        for k, v in self.__dict__.items():
            if k not in ("timestamp", "event_id"):
                d[k] = v
        return d


# ── Interaction events (need response) ──

@dataclass
class AskEvent(AgentEvent):
    """Agent asks a question via ask_supervisor MCP tool."""
    question: str = ""
    context: str = ""


@dataclass
class PermissionEvent(AgentEvent):
    """Agent needs permission for a tool call."""
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)


# ── Observation events ──

@dataclass
class ToolCallEvent(AgentEvent):
    """Agent is about to call a tool (PreToolUse)."""
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)
    tool_use_id: str = ""


@dataclass
class ToolResultEvent(AgentEvent):
    """Tool call completed (PostToolUse)."""
    tool_name: str = ""
    tool_use_id: str = ""
    result_summary: str = ""
    is_error: bool = False


@dataclass
class TextOutputEvent(AgentEvent):
    """Agent produced text output."""
    text: str = ""


@dataclass
class SubagentEvent(AgentEvent):
    """Subagent started or stopped."""
    agent_id: str = ""
    agent_type: str = ""
    action: str = ""  # "start" | "stop"


# ── Lifecycle events ──

@dataclass
class StartEvent(AgentEvent):
    """Agent session started."""
    session_id: str = ""


@dataclass
class StopEvent(AgentEvent):
    """Agent session stopped."""
    reason: str = ""  # end_turn | max_turns | interrupted | error
    result_text: str = ""


@dataclass
class MonitorAlert(AgentEvent):
    """Monitor detected an anomaly."""
    alert_type: str = ""  # idle_timeout | total_timeout | loop_detected
    details: str = ""


@dataclass
class InjectEvent(AgentEvent):
    """Supervisor injected guidance into the Solver."""
    guidance: str = ""
    source: str = ""  # monitor_stuck | monitor_time_limit | user


# ── Handler protocol ──

class EventHandler(Protocol):
    """Upper layers implement this to handle agent events."""

    async def on_permission(self, event: PermissionEvent) -> bool:
        """Return True to allow, False to deny."""
        ...

    async def on_ask(self, event: AskEvent) -> str:
        """Return answer text."""
        ...

    async def on_tool_call(self, event: ToolCallEvent) -> None:
        ...

    async def on_tool_result(self, event: ToolResultEvent) -> None:
        ...

    async def on_text(self, event: TextOutputEvent) -> None:
        ...

    async def on_stop(self, event: StopEvent) -> None:
        ...

    async def on_monitor_alert(self, event: MonitorAlert) -> str:
        """Return: 'continue' | 'interrupt' | 'inject:<guidance>'."""
        ...


class DefaultHandler:
    """No-op handler. Allows everything, logs nothing."""

    async def on_permission(self, event: PermissionEvent) -> bool:
        return True

    async def on_ask(self, event: AskEvent) -> str:
        return "No guidance available."

    async def on_tool_call(self, event: ToolCallEvent) -> None:
        pass

    async def on_tool_result(self, event: ToolResultEvent) -> None:
        pass

    async def on_text(self, event: TextOutputEvent) -> None:
        pass

    async def on_stop(self, event: StopEvent) -> None:
        pass

    async def on_monitor_alert(self, event: MonitorAlert) -> str:
        return "continue"
