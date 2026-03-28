#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _cli_common import (
    add_metadata_args,
    emit_result,
    ensure_repo_root_on_path,
    parse_env_assignments,
    print_command_preview,
    resolve_workspace_from_args,
)

ensure_repo_root_on_path()

from cuda_exec.runner import run_profile  # noqa: E402

CUDA_BIN = Path("/usr/local/cuda/bin")
NCU = CUDA_BIN / "ncu"
NSYS = Path("/usr/local/bin/nsys")


def add_common_execution_args(parser: argparse.ArgumentParser) -> None:
    add_metadata_args(parser)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout in seconds")
    parser.add_argument("--json", action="store_true", help="Emit result as JSON")
    parser.add_argument(
        "--dry-run-only",
        action="store_true",
        help="Only print the resolved profiler command without executing it",
    )


def build_ncu_args(args: argparse.Namespace) -> list[str]:
    profiler_args: list[str] = []
    if args.set_name:
        profiler_args.extend(["--set", args.set_name])
    if args.target_processes:
        profiler_args.extend(["--target-processes", args.target_processes])
    if args.export:
        profiler_args.extend(["--export", args.export])
    if args.kernel_name:
        profiler_args.extend(["--kernel-name", args.kernel_name])
    if args.launch_count is not None:
        profiler_args.extend(["--launch-count", str(args.launch_count)])
    if args.force_overwrite:
        profiler_args.append("--force-overwrite")
    for section in args.section:
        profiler_args.extend(["--section", section])
    profiler_args.extend(args.profiler_arg)
    return profiler_args


def build_nsys_args(args: argparse.Namespace) -> list[str]:
    profiler_args: list[str] = ["profile"]
    if args.trace:
        profiler_args.extend(["--trace", args.trace])
    if args.output:
        profiler_args.extend(["-o", args.output])
    if args.sample:
        profiler_args.extend(["--sample", args.sample])
    if args.force_overwrite:
        profiler_args.append("--force-overwrite=true")
    profiler_args.extend(args.profiler_arg)
    return profiler_args


def normalize_target_command(command: list[str], parser: argparse.ArgumentParser) -> list[str]:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("Missing target command. Example: profile.py ncu --run-tag alpha --version v1 --direction-id 1 --direction-slug test --turn 0 -- ./a.out")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profiling helper for cuda_exec. Shows and optionally runs NCU/NSYS profiler commands.",
    )
    subparsers = parser.add_subparsers(dest="profiler", required=True)

    ncu_parser = subparsers.add_parser("ncu", help="Profile with Nsight Compute")
    ncu_parser.add_argument("--set-name", default="default", help="NCU section set name")
    ncu_parser.add_argument("--section", action="append", default=[], help="Additional NCU section")
    ncu_parser.add_argument("--target-processes", default="all", help="NCU --target-processes value")
    ncu_parser.add_argument("--export", default=None, help="NCU export prefix")
    ncu_parser.add_argument("--kernel-name", default=None, help="Optional kernel filter")
    ncu_parser.add_argument("--launch-count", type=int, default=None, help="Optional launch count filter")
    ncu_parser.add_argument("--force-overwrite", action="store_true", help="Add --force-overwrite")
    ncu_parser.add_argument("--profiler-arg", action="append", default=[], help="Extra argument passed to ncu")
    add_common_execution_args(ncu_parser)
    ncu_parser.add_argument("target_command", nargs=argparse.REMAINDER, help="Target command, usually passed after --")

    nsys_parser = subparsers.add_parser("nsys", help="Profile with Nsight Systems")
    nsys_parser.add_argument("--trace", default="cuda,nvtx,osrt", help="NSYS trace domains")
    nsys_parser.add_argument("--output", default=None, help="NSYS report output prefix")
    nsys_parser.add_argument("--sample", default=None, help="Optional NSYS sampling mode")
    nsys_parser.add_argument("--force-overwrite", action="store_true", help="Add --force-overwrite=true")
    nsys_parser.add_argument("--profiler-arg", action="append", default=[], help="Extra argument passed to nsys")
    add_common_execution_args(nsys_parser)
    nsys_parser.add_argument("target_command", nargs=argparse.REMAINDER, help="Target command, usually passed after --")

    args = parser.parse_args()
    workspace_path = resolve_workspace_from_args(args)
    env = parse_env_assignments(args.env)
    target_command = normalize_target_command(list(args.target_command), parser)

    if args.profiler == "ncu":
        profiler_args = build_ncu_args(args)
        preview_command = [str(NCU), *profiler_args, *target_command]
    else:
        profiler_args = build_nsys_args(args)
        preview_command = [str(NSYS), *profiler_args, *target_command]

    print_command_preview(preview_command, workspace_path)
    if args.dry_run_only:
        return 0

    result = run_profile(
        profiler=args.profiler,
        target_command=target_command,
        profiler_args=profiler_args,
        workspace_path=workspace_path,
        env=env,
        timeout_seconds=args.timeout,
    )
    emit_result(result, as_json=args.json)
    return 0 if result["ok"] else result["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
