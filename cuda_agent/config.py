"""Centralized Hydra-based configuration for cuda_agent.

Loads and merges YAML configs from the top-level ``conf/`` directory
using Hydra's compose API.  Environment variable interpolation
(``${oc.env:VAR,default}``) is resolved eagerly at load time.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# conf/ lives at the repo root, two levels above this file
# (cuda_agent/config.py -> cuda_agent/ -> repo_root/conf/)
_CONF_DIR = str(Path(__file__).resolve().parents[1] / "conf")


@lru_cache(maxsize=1)
def load_config(overrides: tuple[str, ...] = ()) -> DictConfig:
    """Load the merged Hydra configuration from ``conf/``.

    Uses ``initialize_config_dir`` with an absolute path so that the
    config directory is resolved independently of the Python package
    location.

    Args:
        overrides: Hydra override strings, e.g.
            ``("agent.model=claude-opus-4",)``.
            Must be a tuple (not list) for ``lru_cache`` hashability.

    Returns:
        Resolved, read-only ``DictConfig``.
    """
    with initialize_config_dir(config_dir=_CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=list(overrides))
    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)
    return cfg
