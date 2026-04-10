"""Workshop HTTP API server — status, transcript, and human inject.

Runs as an async background task inside the Workshop process.
Binds to 127.0.0.1 on an OS-assigned port. Writes connection info
to <kb_run_dir>/supervisor_api.json for client discovery.

Endpoints:
    GET  /status      — Workshop state + Solver metrics
    GET  /transcript  — Solver transcript (supports ?tail=N)
    POST /inject      — Soft-inject human guidance into Solver
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
import uvicorn

if TYPE_CHECKING:
    from agents.workshop import Workshop


class InjectRequest(BaseModel):
    message: str


def _find_free_port() -> int:
    """Find a free port by binding to port 0 and releasing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class WorkshopAPIServer:
    """FastAPI server for Workshop introspection and control."""

    def __init__(self, workshop: Workshop):
        self.workshop = supervisor
        self._port: int = 0
        self._api_json_path: Path | None = None
        self._serve_task: asyncio.Task | None = None
        self._server: uvicorn.Server | None = None
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="Workshop API", docs_url=None, redoc_url=None)

        @app.get("/status")
        async def status():
            return self._build_status()

        @app.get("/transcript")
        async def transcript(tail: int | None = Query(None)):
            return self._build_transcript(tail)

        @app.post("/inject")
        async def inject(req: InjectRequest):
            return self._do_inject(req.message.strip())

        return app

    async def start(self) -> int:
        """Start the API server on a random port. Returns the assigned port."""
        self._port = _find_free_port()

        config = uvicorn.Config(
            self.app,
            host="127.0.0.1",
            port=self._port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)

        self._serve_task = asyncio.create_task(self._server.serve())

        # Wait briefly for server to be ready
        for _ in range(50):
            if self._server.started:
                break
            await asyncio.sleep(0.1)

        self._write_api_json()
        print(f"[Workshop API] Listening on http://127.0.0.1:{self._port}")
        return self._port

    async def stop(self) -> None:
        """Stop the API server and clean up."""
        if self._server:
            self._server.should_exit = True
        if self._serve_task:
            try:
                await asyncio.wait_for(self._serve_task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._serve_task.cancel()
        self._server = None
        self._serve_task = None
        self._remove_api_json()

    @property
    def port(self) -> int:
        return self._port

    # ── Route implementations ──

    def _build_status(self) -> dict:
        """Build the full status dict."""
        status = self.workshop.get_status()

        transcript_path = self.workshop._get_transcript_path()
        has_transcript = transcript_path != "(no transcript available)"
        session_dir = Path(transcript_path).parent if has_transcript else None

        event_count = 0
        tool_call_count = 0
        bench_request_count = 0
        last_event_ts = ""
        last_tool_call_ts = ""

        events_path = session_dir / "events.jsonl" if session_dir else None
        if events_path and events_path.exists():
            with open(events_path) as f:
                for line in f:
                    event_count += 1
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("type") == "ToolCallEvent":
                        tool_call_count += 1
                        last_tool_call_ts = ev.get("ts", "")
                    if (
                        ev.get("type") == "AskEvent"
                        and "REQUEST_FORMAL_BENCH" in ev.get("question", "")
                    ):
                        bench_request_count += 1
                    last_event_ts = ev.get("ts", "")

        heartbeat = {}
        heartbeat_path = session_dir / "heartbeat" if session_dir else None
        if heartbeat_path and heartbeat_path.exists():
            text = heartbeat_path.read_text().strip()
            try:
                heartbeat = json.loads(text)
            except json.JSONDecodeError:
                heartbeat = {"ts": text, "source": "unknown"}

        kb_root = Path(self.workshop.config.storage.kb_root).expanduser()
        run_tag = self.workshop.state.run_tag
        gems_dir = kb_root / "runs" / run_tag / "gems"
        gem_count = sum(1 for d in gems_dir.rglob("v*") if d.is_dir()) if gems_dir.exists() else 0

        kernel_name = status.get("kernel", "")
        kernel_file = kb_root / "runs" / run_tag / "gen" / "sm90" / kernel_name / "cuda" / "cuda.cu"
        kernel_lines = sum(1 for _ in open(kernel_file)) if kernel_file.exists() else 0

        status.update({
            "heartbeat": heartbeat,
            "events": event_count,
            "tool_calls": tool_call_count,
            "bench_requests": bench_request_count,
            "last_event_ts": last_event_ts,
            "last_tool_call_ts": last_tool_call_ts,
            "gems": gem_count,
            "kernel_lines": kernel_lines,
            "api_port": self._port,
        })
        return status

    def _build_transcript(self, tail: int | None) -> PlainTextResponse:
        """Build transcript response."""
        transcript_path = self.workshop._get_transcript_path()
        if transcript_path == "(no transcript available)":
            return JSONResponse({"error": "no transcript available"}, status_code=404)

        path = Path(transcript_path)
        if not path.exists():
            return JSONResponse({"error": f"not found: {path}"}, status_code=404)

        content = path.read_text(encoding="utf-8", errors="replace")

        if tail is not None:
            lines = content.splitlines()
            content = "\n".join(lines[-tail:])

        return PlainTextResponse(content)

    def _do_inject(self, message: str) -> dict | JSONResponse:
        """Execute inject — send a user message directly to the Solver."""
        if not message:
            return JSONResponse({"error": "empty message"}, status_code=400)

        runner = self.workshop._solver_runner
        if not runner:
            return JSONResponse({"error": "no Solver running"}, status_code=409)

        client = runner._client
        if not client:
            return JSONResponse({"error": "no SDK client active"}, status_code=409)

        # Send as a user message via the SDK — Solver sees it immediately,
        # no need to wait for the next tool call.
        guidance = f"[Human guidance from Workshop API]: {message}"
        asyncio.create_task(self._send_inject(client, guidance))

        print(f"[Workshop API] Human inject sent: {message[:100]}")

        return {
            "ok": True,
            "message": message,
            "delivery": "sent — Solver will see it as a user message",
        }

    @staticmethod
    async def _send_inject(client, guidance: str) -> None:
        """Send inject message to the Solver via SDK client.query()."""
        try:
            await client.query(guidance)
        except Exception as e:
            print(f"[Workshop API] Inject send failed: {e}")

    # ── API JSON file ──

    def _write_api_json(self) -> None:
        """Write supervisor_api.json to the run's KB directory."""
        run_tag = self.workshop.state.run_tag
        if not run_tag:
            return

        kb_root = Path(self.workshop.config.storage.kb_root).expanduser()
        api_json_path = kb_root / "runs" / run_tag / "supervisor_api.json"
        api_json_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "port": self._port,
            "pid": os.getpid(),
            "kernel": self.workshop.state.kernel,
            "gpu": self.workshop.state.gpu,
            "started_at": (
                self.workshop.state.started_at.isoformat()
                if self.workshop.state.started_at
                else None
            ),
            "run_tag": run_tag,
        }

        api_json_path.write_text(json.dumps(data, indent=2) + "\n")
        self._api_json_path = api_json_path
        print(f"[Workshop API] Wrote {api_json_path}")

    def _remove_api_json(self) -> None:
        """Remove supervisor_api.json on shutdown."""
        if self._api_json_path and self._api_json_path.exists():
            self._api_json_path.unlink()
            print(f"[Workshop API] Removed {self._api_json_path}")
