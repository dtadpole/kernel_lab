"""Append-only structured log of agent events.

Bridges events to persistent storage (events.jsonl + transcript.md).
Provides querying and summarization for Monitor and Steward.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

from agents.events import (
    AgentEvent,
    AskEvent,
    StartEvent,
    StopEvent,
    SubagentEvent,
    ToolCallEvent,
    ToolResultEvent,
    TextOutputEvent,
)
from agents.storage import SessionStorage


class SessionLog:
    """In-memory event log with optional persistence to SessionStorage."""

    def __init__(self, storage: SessionStorage | None = None):
        self.events: list[AgentEvent] = []
        self._storage = storage
        self._start_time: datetime | None = None

    def append(self, event: AgentEvent) -> None:
        """Record an event. Persists to storage if available."""
        self.events.append(event)
        if self._start_time is None:
            self._start_time = event.timestamp

        # Persist to disk
        if self._storage:
            self._storage.append_event(event.to_dict())
            self._append_transcript_line(event)

    def elapsed(self) -> timedelta:
        if self._start_time is None:
            return timedelta(0)
        return datetime.now() - self._start_time

    def last_event_age(self) -> timedelta:
        """Time since the last event was recorded."""
        if not self.events:
            return timedelta(0)
        return datetime.now() - self.events[-1].timestamp

    def recent(self, n: int = 20) -> list[AgentEvent]:
        return self.events[-n:]

    def tool_call_counts(self) -> dict[str, int]:
        """Count how many times each tool was called."""
        counts: Counter[str] = Counter()
        for e in self.events:
            if isinstance(e, ToolCallEvent):
                counts[e.tool_name] += 1
        return dict(counts)

    def recent_tool_sequence(self, n: int = 10) -> list[str]:
        """Last N tool names called, for loop detection."""
        tools = [e.tool_name for e in self.events if isinstance(e, ToolCallEvent)]
        return tools[-n:]

    def to_summary(self, max_chars: int = 4000) -> str:
        """Generate a text summary for Steward context."""
        parts = []
        parts.append(f"Session duration: {self.elapsed()}")
        parts.append(f"Total events: {len(self.events)}")
        parts.append(f"Tool calls: {self.tool_call_counts()}")
        parts.append("")

        # Recent events
        parts.append("Recent activity:")
        for event in self.recent(15):
            ts = event.timestamp.strftime("%H:%M:%S")
            if isinstance(event, ToolCallEvent):
                inp = str(event.tool_input)
                if len(inp) > 100:
                    inp = inp[:100] + "..."
                parts.append(f"  [{ts}] Tool: {event.tool_name} {inp}")
            elif isinstance(event, ToolResultEvent):
                status = "ERROR" if event.is_error else "OK"
                parts.append(f"  [{ts}] Result: {event.tool_name} [{status}] {event.result_summary[:80]}")
            elif isinstance(event, TextOutputEvent):
                text = event.text[:120] + "..." if len(event.text) > 120 else event.text
                parts.append(f"  [{ts}] Output: {text}")
            elif isinstance(event, AskEvent):
                parts.append(f"  [{ts}] Ask: {event.question[:100]}")
            elif isinstance(event, StopEvent):
                parts.append(f"  [{ts}] Stop: {event.reason}")

        summary = "\n".join(parts)
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "\n...(truncated)"
        return summary

    def _append_transcript_line(self, event: AgentEvent) -> None:
        """Format event as human-readable transcript line."""
        assert self._storage is not None
        ts = event.timestamp.strftime("%H:%M:%S")

        if isinstance(event, StartEvent):
            self._storage.append_transcript(f"\n### Session started ({ts})\n")
        elif isinstance(event, ToolCallEvent):
            inp = str(event.tool_input)
            if len(inp) > 200:
                inp = inp[:200] + "..."
            self._storage.append_transcript(f"**[{ts}] Tool: {event.tool_name}**")
            self._storage.append_transcript(f"> {inp}\n")
        elif isinstance(event, ToolResultEvent):
            status = "ERROR" if event.is_error else "OK"
            self._storage.append_transcript(f"**[{ts}] Result ({status}):** {event.result_summary[:200]}\n")
        elif isinstance(event, TextOutputEvent):
            self._storage.append_transcript(f"**[{ts}] Agent:**")
            self._storage.append_transcript(f"> {event.text[:500]}\n")
        elif isinstance(event, AskEvent):
            self._storage.append_transcript(f"**[{ts}] Ask → Supervisor:** {event.question}\n")
        elif isinstance(event, SubagentEvent):
            self._storage.append_transcript(f"**[{ts}] Subagent {event.action}:** {event.agent_type} ({event.agent_id})\n")
        elif isinstance(event, StopEvent):
            self._storage.append_transcript(f"\n### Session ended ({ts}) — {event.reason}\n")
