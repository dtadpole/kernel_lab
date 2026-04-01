"""Path restriction utilities for agent hooks."""

from __future__ import annotations

import os


def is_path_allowed(file_path: str, allowed_dir: str) -> bool:
    """Check whether *file_path* resolves to somewhere inside *allowed_dir*."""
    allowed = os.path.realpath(allowed_dir)
    real = os.path.realpath(file_path)
    return real == allowed or real.startswith(allowed + os.sep)
