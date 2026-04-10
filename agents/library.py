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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from agents.config import MonitorConfig, SystemConfig
from agents.events import (
    AskEvent,
    DefaultHandler,
    StopEvent,
    TextOutputEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from agents.runner import AgentRunner


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
        """Process one proposal by spawning a Librarian LLM agent.

        1. Read proposal YAML
        2. Build prompt with proposal content
        3. Spawn Librarian agent via AgentRunner
        4. Librarian reads wiki, decides action, writes/updates pages
        5. Move processed proposal to _done/
        """
        self.state.phase = "processing"
        self.state.current_proposal = proposal_path.name
        start = datetime.now()

        print(f"[Library] Processing ({source}): {proposal_path.name}")

        try:
            # Read proposal content
            content = proposal_path.read_text(encoding="utf-8")
            print(f"[Library]   Read {len(content)} chars")

            # Build Librarian prompt with proposal
            prompt = self._build_librarian_prompt(proposal_path.name, content)

            # Spawn Librarian agent
            librarian_config = self.config.get_agent("librarian")
            runner = AgentRunner(
                agent_config=librarian_config,
                storage_config=self.config.storage,
                handler=self,
                monitor_config=MonitorConfig(
                    hard_limit=600,           # 10 min per proposal
                    total_timeout=300,        # 5 min soft limit
                    idle_timeout=120,         # 2 min idle
                    check_interval=30,
                ),
                wave=self.state.proposals_processed,
                task_slug="librarian",
            )

            print(f"[Library]   Starting Librarian agent...")
            await runner.start(prompt)

            # Librarian may need multiple sessions:
            # Session 0: analyze proposal, plan actions
            # Session 1+: write wiki pages (if not done in session 0)
            max_sessions = 3
            for session in range(max_sessions):
                result = await runner.run_until_result()
                print(f"[Library]   Session {session} done (stop_reason={result.stop_reason})")

                # Check if Librarian actually wrote wiki pages
                if result.result_text and ("wrote" in result.result_text.lower()
                                           or "created" in result.result_text.lower()
                                           or "updated" in result.result_text.lower()):
                    break

                # If first session just analyzed, prompt to continue writing
                if session < max_sessions - 1:
                    await runner.send_message(
                        "Continue. Write the wiki pages now. "
                        "Use the Write tool to create the files."
                    )

            await runner.stop()

            if result.result_text:
                print(f"[Library]   Result: {result.result_text[:200]}")

            # Move to _done/
            done_path = DONE_DIR / proposal_path.name
            proposal_path.rename(done_path)

            self.state.proposals_processed += 1

        except Exception as e:
            self.state.errors += 1
            print(f"[Library]   ERROR: {type(e).__name__}: {e}")

        elapsed = (datetime.now() - start).total_seconds()
        print(f"[Library]   Done ({elapsed:.1f}s). "
              f"Total processed: {self.state.proposals_processed}")
        self.state.current_proposal = None

    def _build_librarian_prompt(self, filename: str, content: str) -> str:
        """Build the prompt for the Librarian agent."""
        return f"""Process the following knowledge proposal and update the wiki accordingly.

## Proposal: {filename}

```yaml
{content}
```

## Instructions

1. Read the proposal carefully — understand the knowledge claims and evidence.
2. Check the existing wiki at ~/kernel_lab_kb/wiki/ for related pages.
3. Decide: create a new page, update an existing page, or defer.
4. If creating/updating, write the wiki page with proper YAML frontmatter.
5. Update any related pages that should link to/from this new content.

## Wiki Location

Write pages to: ~/kernel_lab_kb/wiki/{{category}}/{{page-slug}}.md
Categories: concepts, patterns, problems, decisions, references

## Page Format

Every wiki page MUST have this frontmatter:
```yaml
---
id: "page-slug"
title: "Human Readable Title"
category: "concepts"
tags: ["tag1", "tag2"]
status: active
created: {datetime.now().strftime('%Y-%m-%d')}
updated: {datetime.now().strftime('%Y-%m-%d')}
sources:
  - "evidence reference"
---
```

Use [[page-slug]] for cross-references between wiki pages.

After writing, report what you did: created/updated which pages, and why.
"""

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
