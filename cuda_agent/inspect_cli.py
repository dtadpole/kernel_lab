"""CLI for inspecting local CUDA agent data store.

Reads compile/evaluate/profile results from the local data store
that the MCP server writes during optimization runs.

Usage:
    python -m cuda_agent.inspect_cli compile --data-dir DIR --turn T [--field FIELD] [--attempt N]
    python -m cuda_agent.inspect_cli evaluate --data-dir DIR --turn T [--config SLUG] [--attempt N]
    python -m cuda_agent.inspect_cli profile --data-dir DIR --turn T [--config SLUG] [--attempt N]
    python -m cuda_agent.inspect_cli raw --data-dir DIR --turn T --stage STAGE [--attempt N] [--side SIDE]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _find_latest_attempt(turn_dir: Path, stage: str) -> int:
    """Find the highest attempt number for a stage in a turn directory."""
    if not turn_dir.is_dir():
        return 0
    highest = 0
    for f in turn_dir.glob(f"{stage}.attempt_*.response.json"):
        # Extract attempt number from filename like "compile.attempt_003.response.json"
        parts = f.stem.split(".")  # ["compile", "attempt_003", "response"]
        if len(parts) >= 2:
            try:
                n = int(parts[1].split("_")[1])
                highest = max(highest, n)
            except (IndexError, ValueError):
                pass
    return highest


def _read_response(data_dir: str, turn: int, stage: str, attempt: int | None) -> dict | None:
    """Read a stored response JSON file."""
    turn_dir = Path(data_dir) / f"turn_{turn}"
    if attempt is None:
        attempt = _find_latest_attempt(turn_dir, stage)
        if attempt == 0:
            return None
    fp = turn_dir / f"{stage}.attempt_{attempt:03d}.response.json"
    if not fp.is_file():
        return None
    return json.loads(fp.read_text(encoding="utf-8"))


def _read_file(data_dir: str, turn: int, stage: str, attempt: int | None, side: str) -> dict | None:
    """Read a stored request or response JSON file."""
    turn_dir = Path(data_dir) / f"turn_{turn}"
    if attempt is None:
        attempt = _find_latest_attempt(turn_dir, stage)
        if attempt == 0:
            return None
    fp = turn_dir / f"{stage}.attempt_{attempt:03d}.{side}.json"
    if not fp.is_file():
        return None
    return json.loads(fp.read_text(encoding="utf-8"))


def _list_available(data_dir: str, turn: int) -> list[str]:
    """List available data point prefixes in a turn directory."""
    turn_dir = Path(data_dir) / f"turn_{turn}"
    if not turn_dir.is_dir():
        return []
    seen: set[str] = set()
    for f in sorted(turn_dir.glob("*.json")):
        name = f.name
        for suffix in (".request.json", ".response.json"):
            if name.endswith(suffix):
                seen.add(name[: -len(suffix)])
                break
    return sorted(seen)


def _error(msg: str, data_dir: str | None = None, turn: int | None = None) -> str:
    """Build a JSON error response."""
    result: dict = {"error": msg}
    if data_dir is not None and turn is not None:
        result["available"] = _list_available(data_dir, turn)
    return json.dumps(result, indent=2)


def cmd_compile(args: argparse.Namespace) -> str:
    data = _read_response(args.data_dir, args.turn, "compile", args.attempt)
    if data is None:
        return _error(f"No compile response found for turn {args.turn}.",
                      args.data_dir, args.turn)

    all_ok = data.get("all_ok")
    field = args.field

    if field == "all":
        return json.dumps(data, indent=2)

    if field in ("ptx", "sass", "resource_usage"):
        artifact = data.get("artifacts", {}).get(field, {})
        content = artifact.get("content", "")
        return json.dumps({"all_ok": all_ok, field: content}, indent=2)

    if field == "tool_outputs":
        raw_outputs = data.get("tool_outputs", {})
        outputs: dict[str, str] = {}
        for name, payload in raw_outputs.items():
            if isinstance(payload, dict):
                outputs[name] = payload.get("content", "")
            else:
                outputs[name] = str(payload)
        return json.dumps({"all_ok": all_ok, "tool_outputs": outputs}, indent=2)

    return _error(f"Unknown field: {field}")


def cmd_evaluate(args: argparse.Namespace) -> str:
    data = _read_response(args.data_dir, args.turn, "evaluate", args.attempt)
    if data is None:
        return _error(f"No evaluate response found for turn {args.turn}.",
                      args.data_dir, args.turn)

    all_ok = data.get("all_ok")
    configs = data.get("configs", {})

    if args.config is not None:
        if args.config not in configs:
            return json.dumps({
                "error": f"Config '{args.config}' not found.",
                "available_configs": sorted(configs.keys()),
            }, indent=2)
        configs = {args.config: configs[args.config]}

    return json.dumps({"all_ok": all_ok, "configs": configs}, indent=2)


def cmd_profile(args: argparse.Namespace) -> str:
    data = _read_response(args.data_dir, args.turn, "profile", args.attempt)
    if data is None:
        return _error(f"No profile response found for turn {args.turn}.",
                      args.data_dir, args.turn)

    all_ok = data.get("all_ok")
    configs = data.get("configs", {})

    if args.config is not None:
        if args.config not in configs:
            return json.dumps({
                "error": f"Config '{args.config}' not found.",
                "available_configs": sorted(configs.keys()),
            }, indent=2)
        configs = {args.config: configs[args.config]}

    return json.dumps({"all_ok": all_ok, "configs": configs}, indent=2)


def cmd_raw(args: argparse.Namespace) -> str:
    data = _read_file(args.data_dir, args.turn, args.stage, args.attempt, args.side)
    if data is None:
        return _error(
            f"No {args.side} found for {args.stage} in turn {args.turn}.",
            args.data_dir, args.turn,
        )
    return json.dumps(data, indent=2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cuda_agent.inspect_cli",
        description="Inspect local CUDA agent data store.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Common arguments
    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-dir", required=True, help="Path to run data directory")
        p.add_argument("--turn", type=int, required=True, help="Turn number")
        p.add_argument("--attempt", type=int, default=None, help="Attempt number (default: latest)")

    # compile
    p_compile = sub.add_parser("compile", help="Retrieve compile results")
    add_common(p_compile)
    p_compile.add_argument("--field", default="all",
                           choices=["all", "ptx", "sass", "resource_usage", "tool_outputs"],
                           help="Which field to extract")
    p_compile.set_defaults(func=cmd_compile)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Retrieve evaluate results")
    add_common(p_eval)
    p_eval.add_argument("--config", default=None, help="Filter to a single config slug")
    p_eval.set_defaults(func=cmd_evaluate)

    # profile
    p_prof = sub.add_parser("profile", help="Retrieve profile results")
    add_common(p_prof)
    p_prof.add_argument("--config", default=None, help="Filter to a single config slug")
    p_prof.set_defaults(func=cmd_profile)

    # raw
    p_raw = sub.add_parser("raw", help="Retrieve raw request/response data")
    add_common(p_raw)
    p_raw.add_argument("--stage", required=True,
                       choices=["compile", "evaluate", "profile", "execute"],
                       help="Which stage to retrieve")
    p_raw.add_argument("--side", default="response",
                       choices=["request", "response"],
                       help="Which side to return (default: response)")
    p_raw.set_defaults(func=cmd_raw)

    args = parser.parse_args(argv)
    output = args.func(args)
    print(output)


if __name__ == "__main__":
    main()
