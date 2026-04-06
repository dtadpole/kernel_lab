"""Entry point for running the Supervisor → Solver → Benchmarker loop.

Usage:
    # Continuous mode (default) — runs forever, spawning new Solvers
    cd /home/zhenc/kernel_lab
    .venv/bin/python -m agents.main --kernel matmul

    # Single session mode
    .venv/bin/python -m agents.main --kernel matmul --single
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from agents.config import SystemConfig
from agents.supervisor import Supervisor, TaskResult


DEFAULT_TASKS = {
    "matmul": (
        "Optimize the CUDA matmul kernel in data/gen/sm90/matmul/cuda.cu. "
        "Use ik:exec to compile, trial, and profile. "
        "When you believe your optimization is ready, call request_formal_bench. "
        "Target: beat the current best gem in kernel_lab_kb."
    ),
    "fa4": (
        "Optimize the Flash Attention 4 kernel in data/gen/sm90/fa4/cuda.cu. "
        "Use ik:exec to compile, trial, and profile. "
        "When ready, call request_formal_bench. "
        "Target: beat the current best gem."
    ),
}


def print_result(result: TaskResult) -> None:
    print(f"\n{'='*60}")
    print(f"  SESSION RESULT")
    print(f"{'='*60}")
    print(f"  Success:    {result.success}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Tool calls: {result.total_tool_calls}")
    print(f"  Errors:     {result.total_errors}")
    print(f"  Elapsed:    {result.elapsed_seconds:.0f}s ({result.elapsed_seconds/60:.1f}m)")
    print(f"  Benchmarks: {len(result.bench_results)}")
    for i, br in enumerate(result.bench_results):
        improved = "✓ IMPROVED" if br["improved"] else "✗ no improvement"
        print(f"    [{i}] {br['kernel']} — {improved}")
    print(f"\n  Verdict history:")
    for v in result.verdict_history:
        print(f"    iter {v['iteration']}: {v['action']} — {v['detail'][:80]}")
    print(f"\n  Result (first 500 chars):")
    print(f"  {result.result_text[:500]}")
    print(f"{'='*60}\n")


async def run_continuous(args: argparse.Namespace) -> None:
    """Continuous mode — run Solver sessions forever."""
    config = SystemConfig.from_yaml(args.config)

    supervisor = Supervisor(
        config=config,
        max_iterations=args.max_iterations,
        response_prompts_dir=args.prompts_dir,
    )

    task = args.task or DEFAULT_TASKS.get(args.kernel, DEFAULT_TASKS["matmul"])

    print(f"[Main] Starting Supervisor — CONTINUOUS MODE")
    print(f"[Main] Kernel: {args.kernel}")
    print(f"[Main] Task: {task[:100]}...")
    print(f"[Main] Config: {args.config}")

    await supervisor.run_continuous(task=task, kernel=args.kernel)


async def run_single(args: argparse.Namespace) -> TaskResult:
    """Single session mode — run one Solver session."""
    config = SystemConfig.from_yaml(args.config)

    supervisor = Supervisor(
        config=config,
        max_iterations=args.max_iterations,
        response_prompts_dir=args.prompts_dir,
    )

    task = args.task or DEFAULT_TASKS.get(args.kernel, DEFAULT_TASKS["matmul"])

    print(f"[Main] Starting Supervisor — SINGLE SESSION")
    print(f"[Main] Kernel: {args.kernel}")
    print(f"[Main] Task: {task[:100]}...")
    print(f"[Main] Config: {args.config}")

    result = await supervisor.run_task(
        task=task,
        kernel=args.kernel,
        run_tag=args.run_tag,
    )

    print_result(result)

    # Save result to journal
    journal_dir = Path(config.storage.kb_root).expanduser() / config.storage.journal_dir
    result_file = journal_dir / "supervisor_results.jsonl"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "kernel": args.kernel,
            "success": result.success,
            "iterations": result.iterations,
            "elapsed_seconds": result.elapsed_seconds,
            "bench_results": result.bench_results,
            "verdict_history": result.verdict_history,
        }, default=str) + "\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run Supervisor → Solver → Benchmarker loop")
    parser.add_argument("--kernel", default="matmul", choices=["matmul", "fa4", "vecadd"],
                        help="Kernel to optimize (default: matmul)")
    parser.add_argument("--task", default=None,
                        help="Custom task description (default: auto-generated)")
    parser.add_argument("--config", default="conf/agent/agents.yaml",
                        help="Config file path")
    parser.add_argument("--prompts-dir", default="conf/agent/response_prompts",
                        help="Steward prompts directory")
    parser.add_argument("--max-iterations", type=int, default=0,
                        help="Max iterations per session (0 = unlimited)")
    parser.add_argument("--run-tag", default=None,
                        help="Custom run_tag (single mode only)")
    parser.add_argument("--single", action="store_true",
                        help="Run a single session instead of continuous mode")
    args = parser.parse_args()

    if args.single:
        result = asyncio.run(run_single(args))
        sys.exit(0 if result.success else 1)
    else:
        asyncio.run(run_continuous(args))


if __name__ == "__main__":
    main()
