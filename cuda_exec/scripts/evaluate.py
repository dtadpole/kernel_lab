#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _cli_common import (
    add_metadata_args,
    emit_result,
    ensure_repo_root_on_path,
    parse_env_assignments,
    print_command_preview,
    resolve_workspace_from_args,
)

ensure_repo_root_on_path()

from cuda_exec.runner import run_generic_command  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluation helper for cuda_exec. Runs an evaluator command and captures stdout/stderr.",
    )
    add_metadata_args(parser)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument("--json", action="store_true", help="Emit result as JSON")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run, usually passed after --",
    )
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("Missing evaluator command. Example: evaluate.py --run-tag alpha --version v1 --direction-id 1 --direction-slug test --turn 0 -- bash -lc 'python eval.py'")

    workspace_path = resolve_workspace_from_args(args)
    env = parse_env_assignments(args.env)

    print_command_preview(command, workspace_path)
    result = run_generic_command(
        kind="evaluate",
        command=command,
        workspace_path=workspace_path,
        env=env,
        timeout_seconds=args.timeout,
    )
    emit_result(result, as_json=args.json)
    return 0 if result["ok"] else result["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
