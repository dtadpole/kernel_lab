"""Tests for the dispatch module — target validation and host resolution.

Quick tests only: no GPU, no remote service required.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# validate_target
# ---------------------------------------------------------------------------


@pytest.mark.quick
class TestValidateTarget:
    """Tests for validate_target()."""

    def test_valid_local(self):
        from plugins.cuda.dispatch import validate_target

        # Should not raise
        validate_target({"mode": "local", "gpu_index": 0})
        validate_target({"mode": "local", "gpu_index": 3})

    def test_valid_remote(self):
        from plugins.cuda.dispatch import validate_target

        validate_target({"mode": "remote", "host": "_one"})
        validate_target({"mode": "remote", "host": "some_host"})

    def test_missing_mode(self):
        from plugins.cuda.dispatch import validate_target

        with pytest.raises(ValueError, match="mode"):
            validate_target({"gpu_index": 0})

    def test_invalid_mode(self):
        from plugins.cuda.dispatch import validate_target

        with pytest.raises(ValueError, match="mode"):
            validate_target({"mode": "cloud", "host": "x"})

    def test_local_missing_gpu_index(self):
        from plugins.cuda.dispatch import validate_target

        with pytest.raises(ValueError, match="gpu_index"):
            validate_target({"mode": "local"})

    def test_local_negative_gpu_index(self):
        from plugins.cuda.dispatch import validate_target

        with pytest.raises(ValueError, match="gpu_index"):
            validate_target({"mode": "local", "gpu_index": -1})

    def test_remote_missing_host(self):
        from plugins.cuda.dispatch import validate_target

        with pytest.raises(ValueError, match="host"):
            validate_target({"mode": "remote"})

    def test_remote_empty_host(self):
        from plugins.cuda.dispatch import validate_target

        with pytest.raises(ValueError, match="host"):
            validate_target({"mode": "remote", "host": ""})


# ---------------------------------------------------------------------------
# resolve_host_url
# ---------------------------------------------------------------------------


@pytest.mark.quick
class TestResolveHostUrl:
    """Tests for resolve_host_url()."""

    def test_resolve_one(self):
        from plugins.cuda.dispatch import resolve_host_url

        url = resolve_host_url("_one")
        assert url == "http://66.179.249.233:41980"

    def test_resolve_two(self):
        from plugins.cuda.dispatch import resolve_host_url

        url = resolve_host_url("_two")
        assert url == "http://66.179.249.233:42980"

    def test_resolve_no_internet_host(self):
        from plugins.cuda.dispatch import resolve_host_url

        url = resolve_host_url("h8_3")
        assert url == "http://devvm8490.cco0.facebook.com:8980"

    def test_unknown_host_raises(self):
        from plugins.cuda.dispatch import resolve_host_url

        with pytest.raises(ValueError, match="not found"):
            resolve_host_url("nonexistent_host")
