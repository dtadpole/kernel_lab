"""CLI entry point for the CUDA optimization agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from cuda_agent.agent import run_optimization
from cuda_agent.config import load_config
from cuda_agent.task import OptimizationTask


def _read_dir_files(dir_path: Path) -> dict[str, str]:
    """Read all files in a directory into a {relative_path: content} map."""

    skip_dirs = {"__pycache__", ".git"}
    files: dict[str, str] = {}
    for p in sorted(dir_path.rglob("*")):
        if p.is_file() and not (skip_dirs & set(p.parts)):
            try:
                rel = str(p.relative_to(dir_path))
                files[rel] = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
    return files


def _read_single_file(file_path: Path) -> dict[str, str]:
    """Read a single file into a {filename: content} map."""

    return {file_path.name: file_path.read_text(encoding="utf-8")}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cuda_agent",
        description="Iteratively optimize a CUDA kernel via the cuda_exec service.",
    )
    p.add_argument("--run-tag", required=True, help="Agent run namespace tag")
    p.add_argument("--version", required=True, help="Agent/API version tag")
    p.add_argument("--direction-id", type=int, required=True, help="Stable integer id for a research direction")
    p.add_argument("--direction-slug", required=True, help="Readable slug for a research direction")
    p.add_argument("--reference-dir", type=Path, required=True, help="Directory containing reference Python file(s)")
    p.add_argument("--generated-file", type=Path, required=True, help="Path to the initial generated .cu file")
    p.add_argument("--configs-file", type=Path, required=True, help="JSON file with slug-keyed runtime configs")
    p.add_argument("--max-iterations", type=int, default=None, help="Max optimization iterations (default: from config)")
    p.add_argument("--speedup-target", type=float, default=None, help="Optional speedup target vs reference")
    p.add_argument("--cuda-exec-url", default=None, help="cuda_exec service URL (default: from config)")
    p.add_argument("--config-override", action="append", default=[], help="Hydra config override, e.g. agent.model=claude-opus-4")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    overrides = tuple(args.config_override)
    cfg = load_config(overrides)

    if not args.reference_dir.is_dir():
        print(f"error: --reference-dir is not a directory: {args.reference_dir}", file=sys.stderr)
        raise SystemExit(1)
    if not args.generated_file.is_file():
        print(f"error: --generated-file not found: {args.generated_file}", file=sys.stderr)
        raise SystemExit(1)
    if not args.configs_file.is_file():
        print(f"error: --configs-file not found: {args.configs_file}", file=sys.stderr)
        raise SystemExit(1)

    reference_files = _read_dir_files(args.reference_dir)
    generated_files = _read_single_file(args.generated_file)
    configs = json.loads(args.configs_file.read_text(encoding="utf-8"))

    task = OptimizationTask(
        run_tag=args.run_tag,
        version=args.version,
        direction_id=args.direction_id,
        direction_slug=args.direction_slug,
        reference_files=reference_files,
        initial_generated_files=generated_files,
        configs=configs,
        max_iterations=args.max_iterations if args.max_iterations is not None else 10,
        speedup_target=args.speedup_target,
        cuda_exec_url=args.cuda_exec_url or cfg.service.cuda_exec_url,
    )

    result = asyncio.run(run_optimization(task, overrides=overrides))
    print(result)
