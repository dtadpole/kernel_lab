#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

TIMESTAMP_PREFIX_FORMAT = "%Y-%m-%d-%H-%M"
DEFAULT_ROOT = Path.home() / "temp"
RUN_PREFIX = "cuda-exec-"
KEEP_MARKERS = ("KEEP", "keep", ".keep")


@dataclass
class Candidate:
    path: Path
    created_at: datetime
    keep_reason: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prune old preserved cuda_exec run directories under ~/temp by default. "
            "Default behavior deletes runs older than 7 days unless they are marked keep."
        )
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help=f"Parent directory containing preserved run directories (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Delete runs older than this many days (default: 7)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without removing anything",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print kept and skipped directories as well as deletions",
    )
    return parser.parse_args()


def _timestamp_from_name(path: Path) -> datetime | None:
    parts = path.name.split("-", 5)
    if len(parts) < 5:
        return None
    prefix = "-".join(parts[:5])
    try:
        return datetime.strptime(prefix, TIMESTAMP_PREFIX_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        return None


def _created_at(path: Path) -> datetime:
    stamped = _timestamp_from_name(path)
    if stamped is not None:
        return stamped
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


def _keep_reason(path: Path) -> str | None:
    lowered = path.name.lower()
    if "keep" in lowered:
        return "directory name contains keep"
    for marker in KEEP_MARKERS:
        if (path / marker).exists():
            return f"marker file present: {marker}"
    return None


def _iter_candidates(root: Path) -> Iterable[Candidate]:
    if not root.exists():
        return []
    items: list[Candidate] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if RUN_PREFIX not in child.name:
            continue
        items.append(Candidate(path=child, created_at=_created_at(child), keep_reason=_keep_reason(child)))
    return items


def main() -> int:
    args = _parse_args()
    root = Path(args.root).expanduser().resolve()
    cutoff = datetime.now(UTC) - timedelta(days=args.days)

    print(f"root={root}")
    print(f"days={args.days}")
    print(f"dry_run={args.dry_run}")
    print(f"cutoff={cutoff.isoformat()}")

    deleted = 0
    kept = 0
    skipped = 0

    for candidate in _iter_candidates(root):
        if candidate.keep_reason is not None:
            kept += 1
            if args.verbose:
                print(f"KEEP  {candidate.path}  ({candidate.keep_reason})")
            continue

        if candidate.created_at >= cutoff:
            skipped += 1
            if args.verbose:
                print(f"SKIP  {candidate.path}  (newer than cutoff: {candidate.created_at.isoformat()})")
            continue

        deleted += 1
        action = "DELETE" if not args.dry_run else "DRYRUN"
        print(f"{action} {candidate.path}  (created_at={candidate.created_at.isoformat()})")
        if not args.dry_run:
            shutil.rmtree(candidate.path)

    print(f"summary: deleted={deleted} kept={kept} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
