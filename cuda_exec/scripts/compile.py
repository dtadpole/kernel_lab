#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _cli_common import default_workdir, emit_result, ensure_repo_root_on_path, parse_env_assignments, print_command_preview

ensure_repo_root_on_path()

from cuda_exec.runner import run_generic_command  # noqa: E402

CUDA_BIN = Path("/usr/local/cuda/bin")
NVCC = CUDA_BIN / "nvcc"
PTXAS = CUDA_BIN / "ptxas"


def build_nvcc_command(args: argparse.Namespace) -> list[str]:
    command = [str(NVCC)]
    if args.arch:
        command.append(f"-arch={args.arch}")
    if args.std:
        command.append(f"-std={args.std}")
    if args.opt:
        command.append(f"-O{args.opt}")
    if args.lineinfo:
        command.append("-lineinfo")

    mode_flags = {
        "binary": [],
        "object": ["-dc"],
        "ptx": ["-ptx"],
        "cubin": ["-cubin"],
        "preprocess": ["-E"],
    }
    command.extend(mode_flags[args.mode])
    command.extend(args.extra_arg)
    command.append(args.source)
    command.extend(["-o", args.output])
    return command


def build_ptxas_command(args: argparse.Namespace) -> list[str]:
    command = [str(PTXAS), f"-arch={args.arch}"]
    if args.verbose:
        command.append("-v")
    command.extend(args.extra_arg)
    command.extend([args.input, "-o", args.output])
    return command


def add_common_execution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workdir", default=None, help="Working directory for command execution")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument("--json", action="store_true", help="Emit result as JSON")
    parser.add_argument(
        "--dry-run-only",
        action="store_true",
        help="Only print the resolved toolkit command without executing it",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile helper for cuda_exec. Shows and optionally runs CUDA Toolkit compile commands.",
    )
    subparsers = parser.add_subparsers(dest="tool", required=True)

    nvcc_parser = subparsers.add_parser("nvcc", help="Compile with nvcc")
    nvcc_parser.add_argument("--source", required=True, help="Source file passed to nvcc")
    nvcc_parser.add_argument("--output", required=True, help="Output path")
    nvcc_parser.add_argument(
        "--mode",
        choices=["binary", "object", "ptx", "cubin", "preprocess"],
        default="binary",
        help="nvcc compilation mode",
    )
    nvcc_parser.add_argument("--arch", default="native", help="nvcc -arch value")
    nvcc_parser.add_argument("--std", default="c++17", help="C++ standard")
    nvcc_parser.add_argument("--opt", default="3", help="Optimization level without the leading O")
    nvcc_parser.add_argument("--lineinfo", action="store_true", help="Add -lineinfo")
    nvcc_parser.add_argument("--extra-arg", action="append", default=[], help="Extra argument passed to nvcc")
    add_common_execution_args(nvcc_parser)

    ptxas_parser = subparsers.add_parser("ptxas", help="Assemble PTX with ptxas")
    ptxas_parser.add_argument("--input", required=True, help="PTX input file")
    ptxas_parser.add_argument("--output", required=True, help="Output cubin path")
    ptxas_parser.add_argument("--arch", default="sm_120", help="ptxas -arch value")
    ptxas_parser.add_argument("--verbose", action="store_true", help="Add -v")
    ptxas_parser.add_argument("--extra-arg", action="append", default=[], help="Extra argument passed to ptxas")
    add_common_execution_args(ptxas_parser)

    args = parser.parse_args()
    workdir = default_workdir(args.workdir)
    env = parse_env_assignments(args.env)

    if args.tool == "nvcc":
        command = build_nvcc_command(args)
    else:
        command = build_ptxas_command(args)

    print_command_preview(command, workdir)
    if args.dry_run_only:
        return 0

    result = run_generic_command(
        kind="compile",
        command=command,
        workdir=workdir,
        env=env,
        timeout_seconds=args.timeout,
    )
    emit_result(result, as_json=args.json)
    return 0 if result["ok"] else result["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
