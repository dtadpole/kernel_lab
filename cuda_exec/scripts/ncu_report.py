#!/usr/bin/env python3
"""Generate a curated text report from an NCU .ncu-rep binary report.

Runs ``ncu --import <report> --page raw`` and deduplicates per-invocation
``device__`` metrics (keeping only the first copy).  All other metric
categories are kept for every invocation.

Usage::

    python ncu_report.py --input profile.ncu-rep --output report.txt
    python ncu_report.py --input profile.ncu-rep          # stdout
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

NCU_BINARY = "/usr/local/cuda/bin/ncu"


def _generate_report(ncu_rep_path: str) -> str:
    """Import .ncu-rep and return the deduplicated raw-page text."""
    result = subprocess.run(
        [NCU_BINARY, "--import", ncu_rep_path, "--page", "raw"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        return f"ncu --import failed (rc={result.returncode}): {stderr}\n"

    raw = result.stdout
    if not raw.strip():
        return "ncu --import produced no output.\n"

    return _deduplicate_device_metrics(raw)


def _deduplicate_device_metrics(raw: str) -> str:
    """Keep only the first invocation block's device__ lines.

    The raw-page output repeats every metric for each kernel invocation.
    ``device__`` metrics are GPU hardware constants — identical across all
    invocations.  We keep them in the first block and strip them from all
    subsequent blocks.
    """
    lines = raw.split("\n")
    out: list[str] = []
    block_index = -1  # incremented to 0 at first "Metric Name" header

    for line in lines:
        stripped = line.lstrip()

        # Detect invocation block boundary: "Metric Name" header line.
        if stripped.startswith("Metric Name"):
            block_index += 1
            out.append(line)
            continue

        # After the first block, skip device__ lines.
        if block_index > 0 and stripped.startswith("device__"):
            continue

        out.append(line)

    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate curated NCU text report")
    parser.add_argument("--input", required=True, help="Path to .ncu-rep file")
    parser.add_argument("--output", default=None, help="Output text file (default: stdout)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    report = _generate_report(args.input)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding="utf-8")
    else:
        sys.stdout.write(report)


if __name__ == "__main__":
    main()
