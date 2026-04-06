"""Runtime storage for agent sessions.

Manages session directories in the KB repo's agent_journal/.
Process-level logs go to ~/.kernel_lab/logs/.
"""

from __future__ import annotations

import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path

from agents.config import StorageConfig


class SessionStorage:
    """Manages a single session's files in agent_journal/."""

    def __init__(self, config: StorageConfig, agent_name: str, task_slug: str):
        self.agent_name = agent_name
        self.task_slug = task_slug
        self.session_id = self._generate_session_id(agent_name)
        self.session_dir = config.journal_path / agent_name / task_slug / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _generate_session_id(agent_name: str) -> str:
        now = datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M%S")
        hash_str = sha256(f"{agent_name}:{now.isoformat()}".encode()).hexdigest()[:6]
        return f"s001_{date_str}_{hash_str}"

    @property
    def events_path(self) -> Path:
        return self.session_dir / "events.jsonl"

    @property
    def transcript_path(self) -> Path:
        return self.session_dir / "transcript.md"

    @property
    def meta_path(self) -> Path:
        return self.session_dir / "meta.json"

    @property
    def config_snapshot_path(self) -> Path:
        return self.session_dir / "config_snapshot.yaml"

    @property
    def artifacts_dir(self) -> Path:
        d = self.session_dir / "artifacts"
        d.mkdir(exist_ok=True)
        return d

    def append_event(self, event_dict: dict) -> None:
        """Append a single event to events.jsonl."""
        with open(self.events_path, "a") as f:
            f.write(json.dumps(event_dict, ensure_ascii=False, default=str) + "\n")

    def append_transcript(self, line: str) -> None:
        """Append a line to transcript.md."""
        with open(self.transcript_path, "a") as f:
            f.write(line + "\n")

    def write_meta(self, meta: dict) -> None:
        """Write session metadata."""
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    def write_heartbeat(self, ts: datetime) -> None:
        """Write last stream event timestamp to heartbeat file."""
        heartbeat_path = self.session_dir / "heartbeat"
        heartbeat_path.write_text(ts.isoformat())

    def read_heartbeat(self) -> str | None:
        """Read last heartbeat timestamp."""
        heartbeat_path = self.session_dir / "heartbeat"
        if heartbeat_path.exists():
            return heartbeat_path.read_text().strip()
        return None

    def write_config_snapshot(self, config_text: str) -> None:
        """Save the configuration used for this session."""
        self.config_snapshot_path.write_text(config_text)

    def init_transcript(self, agent_name: str, task: str) -> None:
        """Write transcript header with frontmatter."""
        now = datetime.now()
        header = f"""---
slug: {self.session_id}-{agent_name}-{self.task_slug}
type: journal
agent: {agent_name}
task: {self.task_slug}
session_id: {self.session_id}
---

# Session: {agent_name} @ {now.strftime('%Y-%m-%d %H:%M')}

## Task
{task}

## Transcript
"""
        self.transcript_path.write_text(header)


def ensure_log_dir(config: StorageConfig) -> Path:
    """Create and return the process-level log directory."""
    log_dir = config.log_path
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir
