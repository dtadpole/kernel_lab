"""Shared fixtures for cuda plugin tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONF_HOSTS = _REPO_ROOT / "conf" / "hosts" / "default.yaml"


@pytest.fixture(scope="session")
def hosts_config() -> dict[str, Any]:
    """Load host configuration from conf/hosts/default.yaml."""
    with _CONF_HOSTS.open() as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def all_host_names(hosts_config) -> list[str]:
    """All configured host names."""
    return sorted(hosts_config.get("hosts", {}).keys())


def _resolve_host(hosts_config: dict[str, Any], name: str) -> dict[str, Any]:
    defaults = hosts_config.get("host_defaults", {})
    host = hosts_config["hosts"][name]
    return {**defaults, **host}


@pytest.fixture(scope="session")
def host_one(hosts_config) -> dict[str, Any]:
    return _resolve_host(hosts_config, "_one")


@pytest.fixture(scope="session")
def host_two(hosts_config) -> dict[str, Any]:
    return _resolve_host(hosts_config, "_two")


@pytest.fixture()
def sample_metadata() -> dict[str, Any]:
    """Metadata dict for smoke tests."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return {
        "run_tag": f"test_{ts}",
        "version": "v1",
        "direction_id": 99,
        "direction_slug": "test-smoke",
        "turn": 1,
    }


@pytest.fixture(scope="session")
def vecadd_fixtures() -> dict[str, Any]:
    """Load vecadd test fixtures."""
    fixtures_dir = _REPO_ROOT / "conf" / "fixtures" / "sm120" / "vecadd"
    generated_dir = _REPO_ROOT / "data" / "generated" / "sm120" / "vecadd"
    return {
        "reference": (fixtures_dir / "cutedsl.py").read_text(),
        "generated": (generated_dir / "generated.cu").read_text(),
        "configs": json.loads((fixtures_dir / "configs.json").read_text()),
    }


def ssh_run(host: str, cmd: str) -> subprocess.CompletedProcess:
    """Run a command on a remote host. Helper for integration tests."""
    return subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, cmd],
        capture_output=True, text=True,
    )
