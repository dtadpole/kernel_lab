from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Dict, Iterable, List


def ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def parse_env_assignments(values: Iterable[str]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid env assignment: {item!r}. Expected KEY=VALUE")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"Invalid env assignment with empty key: {item!r}")
        env[key] = value
    return env


def shell_join(command: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def print_command_preview(command: List[str], workspace_path: str) -> None:
    print("toolkit_command:")
    print(f"  workspace_path: {workspace_path}")
    print(f"  cmd: {shell_join(command)}")


def emit_result(result: dict, *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=False))
        return

    print("result:")
    print(f"  ok: {result['ok']}")
    print(f"  kind: {result['kind']}")
    print(f"  returncode: {result['returncode']}")
    print(f"  duration_seconds: {result['duration_seconds']:.6f}")
    print(f"  workspace_path: {result['workspace_path']}")
    print("  command:")
    print(f"    {shell_join(result['command'])}")
    print("  stdout:")
    stdout = result.get("output", {}).get("stdout", "")
    print(stdout if stdout else "<empty>")
    print("  stderr:")
    stderr = result.get("output", {}).get("stderr", "")
    print(stderr if stderr else "<empty>")


def add_metadata_args(parser) -> None:
    parser.add_argument("--run-tag", required=True, help="Run namespace tag used under ~/.cuda_exec")
    parser.add_argument("--version", required=True, help="Version component used under ~/.cuda_exec")
    parser.add_argument("--direction-id", type=int, required=True, help="Direction id")
    parser.add_argument("--direction-slug", required=True, help="Direction slug")
    parser.add_argument("--turn", type=int, required=True, help="Turn number")


def resolve_workspace_from_args(args) -> str:
    from cuda_exec.runner import resolve_workspace_bundle

    bundle = resolve_workspace_bundle(
        run_tag=args.run_tag,
        version=args.version,
        direction_id=args.direction_id,
        direction_slug=args.direction_slug,
        turn=args.turn,
    )
    return bundle["workspace_path"]
