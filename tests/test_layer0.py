"""Tests for Layer 0: SDK + Infra verification."""

import anyio
import pytest

from agents.layer0_infra import check_api, check_cli, check_sdk


@pytest.mark.quick
def test_cli_exists():
    result = check_cli()
    assert result["ok"], f"CLI check failed: {result.get('error')}"
    assert result["version"], "No version string returned"
    print(f"  CLI: {result['path']} → {result['version']}")


@pytest.mark.quick
def test_sdk_import():
    result = check_sdk()
    assert result["ok"], f"SDK import failed: {result.get('error')}"
    print("  SDK: all expected exports importable")


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_api_connectivity():
    result = anyio.run(check_api)
    assert result["ok"], f"API check failed: {result.get('error')}"
    print(f"  API response: {result['response'][:80]}")
