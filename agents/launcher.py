"""Process manager for Workshop and Library.

Launches an agent group as a subprocess with full I/O capture.

Responsibilities:
  - Create run_tag and run directory
  - Start subprocess with real-time stdout/stderr logging
  - Write process_start.json and process_end.json to run directory
  - Forward SIGTERM/SIGINT to child process

Usage:
    cd /home/zhenc/kernel_lab
    .venv/bin/python -m agents.launcher run=workshop kernel=matmul gpu=6
    .venv/bin/python -m agents.launcher run=library kernel=matmul
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from datetime import datetime
from pathlib import Path


class Launcher:
    """Manages one subprocess (Workshop or Library) with full I/O capture."""

    def __init__(self, run_tag: str, run_dir: Path, cmd: list[str]):
        self.run_tag = run_tag
        self.run_dir = run_dir
        self.cmd = cmd
        self._proc: asyncio.subprocess.Process | None = None
        self._stdout_file = None
        self._stderr_file = None
        self._stopping = False

    async def start(self) -> int:
        """Start subprocess, capture I/O, wait for exit. Returns exit code."""
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Open log files
        self._stdout_file = open(self.run_dir / "stdout.log", "a")
        self._stderr_file = open(self.run_dir / "stderr.log", "a")

        # Start subprocess
        self._proc = await asyncio.create_subprocess_exec(
            *self.cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        pid = self._proc.pid
        started_at = datetime.now()

        # Write process_start.json
        start_info = {
            "pid": pid,
            "started_at": started_at.isoformat(),
            "run_tag": self.run_tag,
            "command": self.cmd,
        }
        (self.run_dir / "process_start.json").write_text(
            json.dumps(start_info, indent=2) + "\n"
        )

        print(f"[Launcher] Started PID {pid}: {' '.join(self.cmd[:6])}...")
        print(f"[Launcher] Run tag: {self.run_tag}")
        print(f"[Launcher] Logs: {self.run_dir}")

        # Stream stdout/stderr concurrently
        await asyncio.gather(
            self._stream(self._proc.stdout, self._stdout_file, "stdout"),
            self._stream(self._proc.stderr, self._stderr_file, "stderr"),
        )

        # Wait for exit
        await self._proc.wait()
        exit_code = self._proc.returncode
        ended_at = datetime.now()
        duration = (ended_at - started_at).total_seconds()

        # Detect signal
        sig = None
        if exit_code is not None and exit_code < 0:
            sig = -exit_code
            try:
                sig_name = signal.Signals(sig).name
            except (ValueError, AttributeError):
                sig_name = str(sig)
            sig = sig_name

        # Write process_end.json
        end_info = {
            "pid": pid,
            "exit_code": exit_code,
            "ended_at": ended_at.isoformat(),
            "duration_s": round(duration, 1),
            "signal": sig,
        }
        (self.run_dir / "process_end.json").write_text(
            json.dumps(end_info, indent=2) + "\n"
        )

        # Close log files
        self._stdout_file.close()
        self._stderr_file.close()

        print(f"[Launcher] PID {pid} exited (code={exit_code}, duration={duration:.0f}s, signal={sig})")
        return exit_code

    async def stop(self):
        """Send SIGTERM to child, wait up to 10s, then SIGKILL."""
        if not self._proc or self._proc.returncode is not None:
            return
        if self._stopping:
            return
        self._stopping = True

        print(f"[Launcher] Sending SIGTERM to PID {self._proc.pid}")
        try:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                print(f"[Launcher] SIGTERM timeout, sending SIGKILL")
                self._proc.kill()
                await self._proc.wait()
        except ProcessLookupError:
            pass

    async def _stream(self, pipe, file, label: str):
        """Read from pipe line by line, write to file in real-time."""
        async for raw_line in pipe:
            line = raw_line.decode("utf-8", errors="replace")
            file.write(line)
            file.flush()


def _create_run_tag(run: str) -> str:
    """Generate a run tag from agent group and current timestamp."""
    return f"{run}_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


async def async_main(cfg) -> int:
    """Main async entry point."""
    run = cfg.get("run")
    kernel = cfg.get("kernel", "matmul")
    gpu = cfg.get("gpu", 4)
    run_tag = cfg.get("run_tag") or _create_run_tag(run)
    kb_root = Path(cfg.get("kb_root", "~/kernel_lab_kb")).expanduser()
    config_file = cfg.get("config", "conf/agent/agents.yaml")
    task = cfg.get("task")

    run_dir = kb_root / "runs" / run_tag

    # Build command for the subprocess
    module = f"agents.{run}"  # agents.workshop or agents.library
    cmd = [
        sys.executable, "-m", module,
        "--kernel", kernel,
        "--gpu", str(gpu),
        "--run-tag", run_tag,
        "--config", config_file,
    ]
    if task:
        cmd.extend(["--task", task])

    launcher = Launcher(run_tag=run_tag, run_dir=run_dir, cmd=cmd)

    # Forward signals to child
    def _signal_handler(sig, frame):
        print(f"\n[Launcher] Received {signal.Signals(sig).name}")
        asyncio.ensure_future(launcher.stop())

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    exit_code = await launcher.start()
    return exit_code


def main():
    """CLI entry point."""

    # Parse key=value args from sys.argv
    args = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, _, val = arg.partition("=")
            args[key] = val

    if "run" not in args:
        print(
            "Usage: .venv/bin/python -m agents.launcher run=<GROUP> [OPTIONS]\n"
            "\n"
            "Required:\n"
            "  run=<name>             Agent group to run (workshop, library)\n"
            "\n"
            "Options:\n"
            "  kernel=<name>          Kernel to optimize (default: matmul)\n"
            "  gpu=<N>                GPU index (default: 4)\n"
            "  run_tag=<tag>          Run tag (default: auto-generate)\n"
            "  config=<path>          Config file (default: conf/agent/agents.yaml)\n"
            "  task=<text>            Custom task description\n"
            "  kb_root=<path>         Knowledge base root (default: ~/kernel_lab_kb)\n"
            "\n"
            "Examples:\n"
            "  .venv/bin/python -m agents.launcher run=workshop kernel=matmul gpu=6\n"
            "  .venv/bin/python -m agents.launcher run=library kernel=fa4",
            file=sys.stderr,
        )
        sys.exit(1)

    # Apply defaults
    args.setdefault("kernel", "matmul")
    args.setdefault("gpu", 4)
    args.setdefault("config", "conf/agent/agents.yaml")
    args["gpu"] = int(args["gpu"])

    exit_code = asyncio.run(async_main(args))
    sys.exit(exit_code or 0)


if __name__ == "__main__":
    main()
