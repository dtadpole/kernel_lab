"""Entry point for running the Supervisor → Solver → Benchmarker loop.

Usage:
    cd /home/zhenc/kernel_lab
    .venv/bin/python -m agents.main --kernel matmul --task "Optimize the matmul kernel for SM90"
    .venv/bin/python -m agents.main --kernel matmul  # uses default task
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
    print(f"  SUPERVISOR RESULT")
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


async def run(args: argparse.Namespace) -> TaskResult:
    config = SystemConfig.from_yaml(args.config)

    supervisor = Supervisor(
        config=config,
        max_iterations=args.max_iterations,
        response_prompts_dir=args.prompts_dir,
    )

    task = args.task or DEFAULT_TASKS.get(args.kernel, DEFAULT_TASKS["matmul"])

    print(f"[Main] Starting Supervisor")
    print(f"[Main] Kernel: {args.kernel}")
    print(f"[Main] Task: {task[:100]}...")
    print(f"[Main] Max iterations: {args.max_iterations}")
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
                        help="Max solve iterations (0 = unlimited, run until ACCEPT or hard_limit)")
    parser.add_argument("--run-tag", default=None,
                        help="Custom run_tag (default: auto-generated)")
    args = parser.parse_args()

    result = asyncio.run(run(args))
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
