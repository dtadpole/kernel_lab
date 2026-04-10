"""Library — host process for the knowledge base pipeline.

Manages the proposal queue and orchestrates Librarian, Information Analyst,
Taxonomist, and Auditor agents.

Usage:
    python -m agents.library

Main loop:
    1. Check injection queue (priority)
    2. Pick next proposal from queue (timestamp order)
    3. Process one proposal (spawn Librarian agent)
    4. Repeat
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from agents.config import SystemConfig
from agents.events import (
    AskEvent,
    DefaultHandler,
    StopEvent,
    TextOutputEvent,
    ToolCallEvent,
    ToolResultEvent,
)


# ── Constants ──

WIKI_ROOT = Path.home() / "kernel_lab_kb" / "wiki"
PROPOSALS_DIR = WIKI_ROOT / "_proposals"
INJECT_DIR = PROPOSALS_DIR / "_inject"
DONE_DIR = PROPOSALS_DIR / "_done"

POLL_INTERVAL = 10  # seconds between queue scans


# ── State ──

@dataclass
class LibraryState:
    """Runtime state for the Library host process."""
    phase: str = "idle"               # idle | processing | consulting
    run_tag: str = ""
    proposals_processed: int = 0
    current_proposal: str | None = None
    wiki_writes: int = 0
    expert_consults: int = 0
    errors: int = 0
    started_at: datetime | None = None


# ── Library ──

class Library(DefaultHandler):
    """Host process for the knowledge base pipeline.

    Manages a proposal queue with two priority levels:
    - Injection queue (_proposals/_inject/): processed first, non-preemptive
    - Regular queue (_proposals/): processed in timestamp order

    Each proposal is processed by spawning a Librarian LLM agent that can
    optionally consult Taxonomist and Auditor experts via MCP tools.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.state = LibraryState()

    # ── Main loop ──

    async def run_continuous(self) -> None:
        """Run forever: scan queues → process proposals → repeat."""
        now = datetime.now()
        self.state.run_tag = f"library_run_{now.strftime('%Y%m%d_%H%M%S')}"
        self.state.started_at = now

        # Ensure directories exist
        PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
        INJECT_DIR.mkdir(parents=True, exist_ok=True)
        DONE_DIR.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[Library] Knowledge base pipeline started")
        print(f"[Library] Run tag: {self.state.run_tag}")
        print(f"[Library] Proposals dir: {PROPOSALS_DIR}")
        print(f"[Library] Inject dir: {INJECT_DIR}")
        print(f"[Library] Wiki root: {WIKI_ROOT}")
        print(f"[Library] Poll interval: {POLL_INTERVAL}s")
        print(f"[Library] Ctrl+C to stop")
        print(f"{'='*60}\n")

        try:
            while True:
                # Priority 1: Injected proposals (drain all before resuming queue)
                while True:
                    injected = self._next_injection()
                    if injected is None:
                        break
                    await self._process_proposal(injected, source="inject")

                # Priority 2: Regular queue (one at a time, then re-check inject)
                proposal = self._next_proposal()
                if proposal:
                    await self._process_proposal(proposal, source="queue")
                    continue  # re-check inject before next queue item

                # Nothing to do — wait
                self.state.phase = "idle"
                await asyncio.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print(f"\n[Library] Interrupted. Processed {self.state.proposals_processed} proposals.")

    # ── Queue scanning ──

    def _next_injection(self) -> Path | None:
        """Return the oldest injected proposal, or None."""
        return self._oldest_yaml(INJECT_DIR)

    def _next_proposal(self) -> Path | None:
        """Return the oldest queued proposal, or None."""
        return self._oldest_yaml(PROPOSALS_DIR)

    @staticmethod
    def _oldest_yaml(directory: Path) -> Path | None:
        """Find the oldest .yaml file in a directory (by filename timestamp).

        Proposal files are named: YYYYMMDD_HHMMSS_<slug>.yaml
        Returns the one with the earliest timestamp prefix.
        """
        yamls = sorted(
            f for f in directory.iterdir()
            if f.is_file() and f.suffix in (".yaml", ".yml")
        )
        return yamls[0] if yamls else None

    # ── Proposal processing ──

    async def _process_proposal(self, proposal_path: Path, source: str = "queue") -> None:
        """Process one proposal end-to-end.

        Phase 1: Just log and move to _done/ (no LLM agent yet).
        Phase 2+: Spawn Librarian agent to read, consult experts, write wiki.
        """
        self.state.phase = "processing"
        self.state.current_proposal = proposal_path.name
        start = datetime.now()

        print(f"[Library] Processing ({source}): {proposal_path.name}")

        try:
            # TODO Phase 2: Spawn Librarian agent here
            # For now, just read and log
            content = proposal_path.read_text(encoding="utf-8")
            print(f"[Library]   Read {len(content)} chars from {proposal_path.name}")

            # Move to _done/
            done_path = DONE_DIR / proposal_path.name
            proposal_path.rename(done_path)
            print(f"[Library]   Moved to _done/")

            self.state.proposals_processed += 1

        except Exception as e:
            self.state.errors += 1
            print(f"[Library]   ERROR: {e}")

        elapsed = (datetime.now() - start).total_seconds()
        print(f"[Library]   Done ({elapsed:.1f}s). "
              f"Total processed: {self.state.proposals_processed}")
        self.state.current_proposal = None

    # ── EventHandler callbacks (for future LLM agent integration) ──

    async def on_ask(self, event: AskEvent) -> str:
        """Handle Librarian MCP tool calls.

        Dispatches:
        - CONSULT_TAXONOMIST: → spawn Taxonomist agent
        - CONSULT_AUDITOR: → spawn Auditor agent
        """
        if event.question.startswith("CONSULT_TAXONOMIST:"):
            self.state.expert_consults += 1
            # TODO Phase 3: spawn Taxonomist agent
            return "(Taxonomist not available yet)"

        if event.question.startswith("CONSULT_AUDITOR:"):
            self.state.expert_consults += 1
            # TODO Phase 3: spawn Auditor agent
            return "(Auditor not available yet)"

        return "(Unknown request)"

    async def on_tool_call(self, event: ToolCallEvent) -> None:
        pass

    async def on_tool_result(self, event: ToolResultEvent) -> None:
        if event.is_error:
            self.state.errors += 1

    async def on_stop(self, event: StopEvent) -> None:
        pass

    # ── Status ──

    def get_status(self) -> dict:
        """Return current Library state for API/monitoring."""
        return {
            "phase": self.state.phase,
            "run_tag": self.state.run_tag,
            "proposals_processed": self.state.proposals_processed,
            "current_proposal": self.state.current_proposal,
            "wiki_writes": self.state.wiki_writes,
            "expert_consults": self.state.expert_consults,
            "errors": self.state.errors,
            "started_at": self.state.started_at.isoformat() if self.state.started_at else None,
            "queue_size": len(list(PROPOSALS_DIR.glob("*.yaml"))),
            "inject_size": len(list(INJECT_DIR.glob("*.yaml"))),
        }


# ── CLI entry point ──

def main():
    """Run the Library process."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Library (knowledge base pipeline)")
    parser.add_argument("--config", default="conf/agent/agents.yaml",
                        help="Config file path")
    args = parser.parse_args()

    config = SystemConfig.from_yaml(args.config)
    library = Library(config=config)

    print(f"[Library] Config: {args.config}")
    asyncio.run(library.run_continuous())


if __name__ == "__main__":
    main()
