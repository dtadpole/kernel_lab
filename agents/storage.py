"""Runtime storage for agent waves.

Manages wave directories in the KB repo's journal/.
Each wave = one subprocess = one directory with all logs.

Directory structure:
    journal/w000_<timestamp>/
    ├── meta.json
    ├── heartbeat.json
    ├── solver/
    │   ├── events.jsonl
    │   ├── transcript.md
    │   ├── stdin.log
    │   ├── stdout.log
    │   └── stderr.log
    ├── steward/
    │   └── <scenario>_<timestamp>/
    └── directions/
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from agents.config import StorageConfig


class WaveStorage:
    """Manages a single wave's files.

    One wave = one subprocess = one PID.
    A wave may contain multiple sessions (CONTINUE).
    """

    def __init__(self, config: StorageConfig, agent_name: str, task_slug: str = "", wave: int = 0):
        self.agent_name = agent_name
        self.task_slug = task_slug  # kept for backward compat (e.g. init_transcript)
        self.wave = wave

        now = datetime.now()
        wave_id = f"w{wave:03d}_{now.strftime('%Y%m%d_%H%M%S')}"
        self.wave_dir = config.journal_path / wave_id
        self.wave_dir.mkdir(parents=True, exist_ok=True)
        (self.wave_dir / self._subdir).mkdir(exist_ok=True)
        (self.wave_dir / "directions").mkdir(exist_ok=True)

        # Log file handles (opened lazily, closed in close_logs)
        self._stdin_log: open | None = None
        self._stdout_log: open | None = None
        self._stderr_log: open | None = None

    # ── Path properties ──

    @property
    def _subdir(self) -> str:
        """Agent subdirectory based on agent_name.

        steward_* → steward/, solver → solver/, others → their own name.
        The directory is created on first use if it doesn't exist.
        """
        if self.agent_name.startswith("steward"):
            return "steward"
        return self.agent_name

    @property
    def events_path(self) -> Path:
        return self.wave_dir / self._subdir / "events.jsonl"

    @property
    def transcript_path(self) -> Path:
        return self.wave_dir / self._subdir / "transcript.md"

    @property
    def directions_path(self) -> Path:
        return self.wave_dir / "directions"

    @property
    def meta_path(self) -> Path:
        return self.wave_dir / "meta.json"

    # ── Process lifecycle ──

    def write_process_start(self, pid: int, agent: str, model: str) -> None:
        """Write process_start.json immediately after subprocess starts."""
        data = {
            "pid": pid,
            "ts": datetime.now().isoformat(),
            "wave": self.wave,
            "agent": agent,
            "model": model,
        }
        path = self.wave_dir / "process_start.json"
        path.write_text(json.dumps(data) + "\n")

    def write_process_end(self, pid: int, exit_code: int) -> None:
        """Write process_end.json after subprocess exits.

        Minimal — only PID, timestamp, exit_code. No parsing, no reason.
        Must succeed even in error conditions.
        """
        data = {
            "pid": pid,
            "ts": datetime.now().isoformat(),
            "exit_code": exit_code,
        }
        path = self.wave_dir / "process_end.json"
        path.write_text(json.dumps(data) + "\n")

    # ── Raw pipe logging ──

    def log_stdin(self, raw_data: str) -> None:
        """Log a raw message written to subprocess stdin."""
        if self._stdin_log is None:
            self._stdin_log = open(self.wave_dir / self._subdir / "stdin.log", "a")
        ts = datetime.now().isoformat(timespec="seconds")
        self._stdin_log.write(f"[{ts}] {raw_data}\n")
        self._stdin_log.flush()

    def log_stdout(self, raw_data: str) -> None:
        """Log a raw message read from subprocess stdout."""
        if self._stdout_log is None:
            self._stdout_log = open(self.wave_dir / self._subdir / "stdout.log", "a")
        ts = datetime.now().isoformat(timespec="seconds")
        self._stdout_log.write(f"[{ts}] {raw_data}\n")
        self._stdout_log.flush()

    def log_stderr(self, line: str) -> None:
        """Log a line from subprocess stderr."""
        if self._stderr_log is None:
            self._stderr_log = open(self.wave_dir / self._subdir / "stderr.log", "a")
        ts = datetime.now().isoformat(timespec="seconds")
        self._stderr_log.write(f"[{ts}] {line}\n")
        self._stderr_log.flush()

    # ── Structured events ──

    def append_event(self, event_dict: dict) -> None:
        """Append a structured event to events.jsonl."""
        with open(self.events_path, "a") as f:
            f.write(json.dumps(event_dict, ensure_ascii=False, default=str) + "\n")

    def append_transcript(self, line: str) -> None:
        """Append a line to transcript.md."""
        with open(self.transcript_path, "a") as f:
            f.write(line + "\n")

    # ── Metadata ──

    def write_meta(self, meta: dict) -> None:
        """Write session metadata."""
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    def write_heartbeat(self, ts: datetime, source: str) -> None:
        """Write heartbeat JSON. source is required."""
        if not source:
            raise ValueError("heartbeat source must not be empty")
        path = self.wave_dir / "heartbeat.json"
        data = {"ts": ts.isoformat(), "source": source}
        path.write_text(json.dumps(data))

    def read_heartbeat(self) -> dict | None:
        """Read heartbeat JSON."""
        path = self.wave_dir / "heartbeat.json"
        if path.exists():
            text = path.read_text().strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"ts": text, "source": "unknown"}
        return None

    def init_transcript(self, agent_name: str, task: str) -> None:
        """Write transcript header."""
        now = datetime.now()
        wave_id = self.wave_dir.name
        header = f"""---
wave: {self.wave}
wave_id: {wave_id}
agent: {agent_name}
task: {self.task_slug}
---

# Wave {self.wave}: {agent_name} @ {now.strftime('%Y-%m-%d %H:%M')}

## Task
{task}

## Transcript
"""
        self.transcript_path.write_text(header)

    # ── Steward ──

    def steward_storage(self, scenario: str) -> Path:
        """Create a Steward sub-directory for a specific scenario call."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.wave_dir / "steward" / f"{scenario}_{ts}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ── Cleanup ──

    def close_logs(self) -> None:
        """Close all open log file handles. Each step independent."""
        for name, fh in [("stdin", self._stdin_log), ("stdout", self._stdout_log), ("stderr", self._stderr_log)]:
            if fh:
                try:
                    fh.close()
                except Exception as e:
                    print(f"[WaveStorage] Failed to close {name}.log: {e}")
        self._stdin_log = None
        self._stdout_log = None
        self._stderr_log = None


# Legacy alias for backward compatibility with tests
SessionStorage = WaveStorage


def ensure_log_dir(config: StorageConfig) -> Path:
    """Create and return the process-level log directory."""
    log_dir = config.log_path
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir
