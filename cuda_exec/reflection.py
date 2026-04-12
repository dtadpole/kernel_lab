"""Bench reflection storage.

Saves reflection and gem notes after each formal benchmark.
Can be used from Supervisor (programmatic) or CLI (standalone).

Usage (CLI):
    # Reflection only (no gem)
    .venv/bin/python -m cuda_exec.reflection \
        --run-tag supervisor_run_20260407_234041 \
        --bench-ts 20260408_003040 \
        --kernel fa4 \
        --reflection-md "## Discovery\n1. ..."

    # With gem
    .venv/bin/python -m cuda_exec.reflection \
        --run-tag supervisor_run_20260407_234041 \
        --bench-ts 20260408_003040 \
        --kernel fa4 \
        --gem-id gen-cuda/v003 \
        --gem-notes-md "## Implementation\n- ..." \
        --reflection-md "## Discovery\n1. ..."

Storage:
    reflection_md → impls/<bench_ts>/reflection.md
    gem_notes_md  → gems/v00N_*/notes.md
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


_KB_REPO = Path.home() / "kernel_lab_kb"


def save_bench_reflection(
    *,
    run_tag: str,
    bench_ts: str,
    kernel: str,
    reflection_md: str,
    gem_id: str = "",
    gem_notes_md: str = "",
    kb_repo: Path | None = None,
) -> dict:
    """Save bench reflection and optional gem notes.

    Args:
        run_tag: Supervisor run tag (e.g. supervisor_run_20260407_234041)
        bench_ts: Bench timestamp from formal.py (e.g. 20260408_003040)
        kernel: Kernel name (e.g. fa4)
        reflection_md: Markdown reflection (required)
        gem_id: Optional gem ID (e.g. gen-cuda/v003)
        gem_notes_md: Optional Markdown gem notes (required if gem_id set)
        kb_repo: KB repo path (default ~/kernel_lab_kb)

    Returns:
        dict with paths written and status
    """
    repo = kb_repo or _KB_REPO
    run_dir = repo / "runs" / run_tag
    result = {"status": "ok", "files_written": []}

    # 1. Save reflection to impls/<bench_ts>/reflection.md
    impls_dir = run_dir / "impls" / bench_ts
    if impls_dir.exists():
        reflection_path = impls_dir / "reflection.md"
        reflection_path.write_text(reflection_md.strip() + "\n")
        result["files_written"].append(str(reflection_path))
        result["reflection_path"] = str(reflection_path)
    else:
        # impls dir might not exist if bench failed; write to run root
        fallback = run_dir / "reflections"
        fallback.mkdir(parents=True, exist_ok=True)
        reflection_path = fallback / f"{bench_ts}.md"
        reflection_path.write_text(reflection_md.strip() + "\n")
        result["files_written"].append(str(reflection_path))
        result["reflection_path"] = str(reflection_path)
        result["note"] = f"impls/{bench_ts}/ not found, wrote to reflections/"

    # 2. Save gem notes if gem_id provided
    if gem_id and gem_notes_md:
        # gem_id can be "gen-cuda/v005" (legacy) or just "v005"
        parts = gem_id.split("/")
        version = parts[-1]  # always the last part
        gem_pattern = str(run_dir / "gems" / f"{version}_*")
        matches = sorted(glob.glob(gem_pattern))
        if matches:
            gem_dir = Path(matches[-1])
            notes_path = gem_dir / "notes.md"
            notes_path.write_text(gem_notes_md.strip() + "\n")
            result["files_written"].append(str(notes_path))
            result["gem_notes_path"] = str(notes_path)
        else:
            result["gem_error"] = f"Gem not found: {gem_pattern}"

    return result


def cli_main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Save bench reflection and gem notes")
    parser.add_argument("--run-tag", required=True, help="Supervisor run tag")
    parser.add_argument("--bench-ts", required=True, help="Bench timestamp")
    parser.add_argument("--kernel", required=True, help="Kernel name")
    parser.add_argument("--reflection-md", required=True, help="Markdown reflection text")
    parser.add_argument("--gem-id", default="", help="Gem ID (e.g. gen-cuda/v003)")
    parser.add_argument("--gem-notes-md", default="", help="Markdown gem notes")
    parser.add_argument("--kb-repo", default=None, help="KB repo path")
    args = parser.parse_args()

    kb = Path(args.kb_repo) if args.kb_repo else None
    result = save_bench_reflection(
        run_tag=args.run_tag,
        bench_ts=args.bench_ts,
        kernel=args.kernel,
        reflection_md=args.reflection_md,
        gem_id=args.gem_id,
        gem_notes_md=args.gem_notes_md,
        kb_repo=kb,
    )

    print(json.dumps(result, indent=2))
    if result.get("gem_error"):
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
