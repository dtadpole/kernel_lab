#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _cli_common import add_metadata_args, ensure_repo_root_on_path

ensure_repo_root_on_path()

from cuda_exec.models import Metadata  # noqa: E402
from cuda_exec.tasks import run_compile_task  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hardened compile helper for cuda_exec. Stages original/generated source files and compiles one CUDA source with nvcc.",
    )
    add_metadata_args(parser)
    parser.add_argument("--original-file", action="append", default=[], help="Original source artifact")
    parser.add_argument("--generated-file", action="append", default=[], help="Generated/candidate source artifact")
    parser.add_argument("--artifact", action="append", default=[], help="Additional artifact to return")
    parser.add_argument("--return-file", action="append", default=[], help="Additional file to return")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument("--json", action="store_true", help="Emit result as JSON")
    args = parser.parse_args()

    metadata = Metadata(
        run_tag=args.run_tag,
        version=args.version,
        direction_id=args.direction_id,
        direction_slug=args.direction_slug,
        turn=args.turn,
    )
    result = run_compile_task(
        metadata=metadata,
        timeout_seconds=args.timeout,
        original_files=args.original_file,
        generated_files=args.generated_file,
        artifacts=args.artifact,
        return_files=args.return_file,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else result["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
