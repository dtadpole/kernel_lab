"""Direction tracking — read/write direction files.

A Direction is a JSON file with 5 fields:
  name, description, opportunity, evidence, ideas

Stored under each wave's directions/ subdirectory with per-wave sequence numbers:
  w000/directions/001_tma-pipeline.json
  w000/directions/002_warp-specialization.json
"""

from __future__ import annotations

import json
from pathlib import Path


def write_direction(directions_dir: Path, seq: int, direction: dict) -> Path:
    """Write a direction file. Returns the path written."""
    name = direction.get("name", "unknown")
    # Sanitize name for filename
    safe_name = name.replace(" ", "-").replace("/", "-")[:30]
    path = directions_dir / f"{seq:03d}_{safe_name}.json"
    directions_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(direction, indent=2, ensure_ascii=False) + "\n")
    return path


def read_active_direction(directions_dir: Path) -> dict | None:
    """Read the latest direction from a wave's directions/ folder.

    Returns None if no directions exist.
    """
    if not directions_dir.exists():
        return None
    files = sorted(directions_dir.glob("*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())


def next_seq(directions_dir: Path) -> int:
    """Next sequence number for a direction file. Starts at 1."""
    if not directions_dir.exists():
        return 1
    return len(list(directions_dir.glob("*.json"))) + 1


def inherit_direction(
    source_directions_dir: Path,
    target_directions_dir: Path,
) -> Path | None:
    """Copy the active direction from a previous wave to a new wave.

    Returns the path of the new file, or None if no direction to inherit.
    """
    direction = read_active_direction(source_directions_dir)
    if direction is None:
        return None
    seq = next_seq(target_directions_dir)
    return write_direction(target_directions_dir, seq, direction)
