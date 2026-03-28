#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from _cli_common import add_metadata_args, ensure_repo_root_on_path

ensure_repo_root_on_path()

from cuda_exec.models import Metadata  # noqa: E402
from cuda_exec.tasks import run_evaluate_task  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hardened evaluate helper for cuda_exec. Evaluates the convention-selected executable artifact.",
    )
    add_metadata_args(parser)
    parser.add_argument("--target-file", action="append", default=[], help="Target artifact to evaluate")
    parser.add_argument("--return-file", action="append", default=[], help="Additional file to return")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    args = parser.parse_args()

    metadata = Metadata(
        run_tag=args.run_tag,
        version=args.version,
        direction_id=args.direction_id,
        direction_slug=args.direction_slug,
        turn=args.turn,
    )
    result = run_evaluate_task(
        metadata=metadata,
        timeout_seconds=args.timeout,
        target_files=args.target_file,
        return_files=args.return_file,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else result["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
