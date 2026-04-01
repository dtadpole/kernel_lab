"""Service skill tests: health, status, lifecycle."""

import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import ssh_run

_CLI = Path(__file__).resolve().parents[1] / "deploy" / "cli.py"


def _cli(*args: str) -> subprocess.CompletedProcess:
    """Run the deploy CLI using the current interpreter."""
    return subprocess.run(
        [sys.executable, str(_CLI), *args],
        capture_output=True, text=True,
    )


# ---------------------------------------------------------------------------
# Quick tests — health check only (fast, but needs SSH)
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_health_one(host_one):
    """_one API is reachable."""
    r = ssh_run(host_one["ssh_host"], f"curl -sf http://127.0.0.1:{host_one['port']}/healthz")
    assert r.returncode == 0
    assert '"ok":true' in r.stdout


@pytest.mark.quick
def test_health_two(host_two):
    """_two API is reachable."""
    r = ssh_run(host_two["ssh_host"], f"curl -sf http://127.0.0.1:{host_two['port']}/healthz")
    assert r.returncode == 0
    assert '"ok":true' in r.stdout


@pytest.mark.quick
def test_health_cli_all():
    """CLI health --all reports both healthy."""
    r = _cli("health", "--all")
    assert "healthy" in r.stdout
    assert "unhealthy" not in r.stdout


# ---------------------------------------------------------------------------
# Integration tests — full lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_status_one():
    """CLI status _one returns full report."""
    r = _cli("status", "_one")
    assert r.returncode == 0
    assert "Service:" in r.stdout
    assert "Health:" in r.stdout
    assert "GPU:" in r.stdout


@pytest.mark.integration
def test_status_all():
    """CLI status --all covers both hosts."""
    r = _cli("status", "--all")
    assert "_one" in r.stdout
    assert "_two" in r.stdout


@pytest.mark.integration
def test_stop_start_cycle_one():
    """Stop → start cycle on _one."""
    r = _cli("stop", "_one")
    assert "Stopped" in r.stdout

    r = _cli("start", "_one")
    assert "Health: OK" in r.stdout

    r = _cli("health", "_one")
    assert "healthy" in r.stdout


@pytest.mark.integration
def test_deploy_start_cycle_two():
    """Deploy → start cycle on _two (idempotent)."""
    r = _cli("deploy", "_two")
    assert "Deploy complete" in r.stdout

    r = _cli("start", "_two")
    assert "Health: OK" in r.stdout
