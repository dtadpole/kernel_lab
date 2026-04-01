"""Load skill content from plugin directories and superpowers."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PLUGIN_DIR = _REPO_ROOT / "plugins"

_SUPERPOWERS_MARKER = "superpowers@claude-plugins-official"
_INSTALLED_PLUGINS = Path.home() / ".claude" / "plugins" / "installed_plugins.json"


def load_plugin_skill(plugin: str, skill: str) -> str:
    """Read a plugin SKILL.md and return its full text content."""
    path = _PLUGIN_DIR / plugin / "skills" / skill / "SKILL.md"
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def superpowers_base_dir() -> Path:
    """Discover the superpowers install path from installed_plugins.json."""
    data = json.loads(_INSTALLED_PLUGINS.read_text(encoding="utf-8"))
    entries = data["plugins"].get(_SUPERPOWERS_MARKER)
    if not entries:
        raise FileNotFoundError(
            f"superpowers plugin not found in {_INSTALLED_PLUGINS}"
        )
    return Path(entries[0]["installPath"])


def superpowers_skill_path(skill: str) -> str:
    """Return the absolute path to a superpowers SKILL.md."""
    return str(superpowers_base_dir() / "skills" / skill / "SKILL.md")
