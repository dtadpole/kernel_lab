#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from _cli_common import add_metadata_args, ensure_repo_root_on_path

ensure_repo_root_on_path()

from cuda_exec.models import Metadata, RuntimeConfig  # noqa: E402
from cuda_exec.tasks import run_evaluate_task  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hardened evaluate helper for cuda_exec. Compile must already have run once for this turn.",
    )
    add_metadata_args(parser)
    parser.add_argument("--config-id", required=True, help="Runtime config id")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--embedding-size", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--causal", action="store_true", help="Enable causal mode")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    args = parser.parse_args()

    metadata = Metadata(
        run_tag=args.run_tag,
        version=args.version,
        direction_id=args.direction_id,
        direction_slug=args.direction_slug,
        turn=args.turn,
    )
    config = RuntimeConfig(
        config_id=args.config_id,
        num_layers=args.num_layers,
        embedding_size=args.embedding_size,
        num_heads=args.num_heads,
        causal=args.causal,
    )
    result = run_evaluate_task(
        metadata=metadata,
        timeout_seconds=args.timeout,
        configs=[config],
    )
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else result["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
