"""Entry point for running the Supervisor.

Usage:
    cd /home/zhenc/kernel_lab
    .venv/bin/python -m agents.main --kernel matmul --gpu 4
    .venv/bin/python -m agents.main --kernel fa4 --gpu 0
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path

from agents.config import SystemConfig
from agents.workshop import Workshop


TASKS_DIR = Path("conf/agent/tasks")


def _load_task(kernel: str) -> str:
    """Load task description from conf/agent/tasks/<kernel>.md."""
    task_file = TASKS_DIR / f"{kernel}.md"
    if task_file.exists():
        return task_file.read_text().strip()
    raise FileNotFoundError(f"No task file for kernel '{kernel}' at {task_file}")


def _setup_process_group():
    """Create a new process group so all children can be killed together."""
    os.setpgrp()

    def _kill_group(signum, frame):
        print(f"\n[Main] Received signal {signum} ({signal.Signals(signum).name}), killing process group...")
        try:
            os.killpg(os.getpgrp(), signal.SIGTERM)
        except ProcessLookupError:
            pass
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, _kill_group)
    signal.signal(signal.SIGINT, _kill_group)


def main():
    _setup_process_group()
    parser = argparse.ArgumentParser(description="Run Supervisor")
    parser.add_argument("--kernel", default="matmul", choices=["matmul", "fa4", "vecadd"],
                        help="Kernel to optimize (default: matmul)")
    parser.add_argument("--gpu", type=int, default=4,
                        help="GPU index for exec/trial/bench (default: 4)")
    parser.add_argument("--task", default=None,
                        help="Custom task description (default: auto-generated)")
    parser.add_argument("--config", default="conf/agent/agents.yaml",
                        help="Config file path")
    parser.add_argument("--prompts-dir", default="conf/agent/response_prompts",
                        help="Steward prompts directory")
    args = parser.parse_args()

    config = SystemConfig.from_yaml(args.config)
    workshop = Workshop(
        config=config,
        response_prompts_dir=args.prompts_dir,
    )

    task = args.task or _load_task(args.kernel)

    print(f"[Main] Kernel: {args.kernel}")
    print(f"[Main] GPU: {args.gpu}")
    print(f"[Main] Task: {task[:100]}...")
    print(f"[Main] Config: {args.config}")

    asyncio.run(workshop.run_continuous(task=task, kernel=args.kernel, gpu=args.gpu))


if __name__ == "__main__":
    main()
