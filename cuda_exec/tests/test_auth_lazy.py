"""Tests for lazy bearer-token loading in cuda_exec.auth.

Verifies that:
- Importing the auth module does NOT crash when the key file is missing
- Calling load_key() or verify_bearer_token() without a key file raises RuntimeError
- load_key() works correctly when a valid key file exists
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# ---------------------------------------------------------------------------
# Mark every test in this module as "quick" (fast, no remote, <5 s).
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.quick


def _fresh_import_auth():
    """Force a fresh import of cuda_exec.auth (drop cached module)."""
    mod_name = "cuda_exec.auth"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


class TestLazyImport:
    """Importing the module must succeed even when no key file exists."""

    def test_import_without_key_file(self, tmp_path, monkeypatch):
        """Module import must not crash when the key file is missing."""
        monkeypatch.setenv("CUDA_EXEC_KEY_PATH", str(tmp_path / "nonexistent.key"))
        auth = _fresh_import_auth()
        # The module object should exist and expose expected names.
        assert hasattr(auth, "load_key")
        assert hasattr(auth, "verify_bearer_token")


class TestLoadKey:
    """load_key() should raise RuntimeError when the key file is missing,
    and return the token when it exists."""

    def test_missing_key_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CUDA_EXEC_KEY_PATH", str(tmp_path / "missing.key"))
        auth = _fresh_import_auth()
        with pytest.raises(RuntimeError, match="key file not found"):
            auth.load_key()

    def test_empty_key_raises(self, tmp_path, monkeypatch):
        key_file = tmp_path / "empty.key"
        key_file.write_text("   \n")
        monkeypatch.setenv("CUDA_EXEC_KEY_PATH", str(key_file))
        auth = _fresh_import_auth()
        with pytest.raises(RuntimeError, match="key file is empty"):
            auth.load_key()

    def test_valid_key_returns_token(self, tmp_path, monkeypatch):
        key_file = tmp_path / "valid.key"
        key_file.write_text("  my-secret-token\n")
        monkeypatch.setenv("CUDA_EXEC_KEY_PATH", str(key_file))
        auth = _fresh_import_auth()
        assert auth.load_key() == "my-secret-token"


class TestVerifyBearerToken:
    """verify_bearer_token() must raise RuntimeError when there is no key,
    and HTTPException(401) on a bad token."""

    @pytest.mark.asyncio
    async def test_verify_without_key_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CUDA_EXEC_KEY_PATH", str(tmp_path / "missing.key"))
        auth = _fresh_import_auth()
        # Build a fake credentials object.
        creds = type("Creds", (), {"credentials": "anything"})()
        with pytest.raises(RuntimeError, match="key file not found"):
            await auth.verify_bearer_token(creds)

    @pytest.mark.asyncio
    async def test_verify_rejects_bad_token(self, tmp_path, monkeypatch):
        key_file = tmp_path / "good.key"
        key_file.write_text("real-token")
        monkeypatch.setenv("CUDA_EXEC_KEY_PATH", str(key_file))
        auth = _fresh_import_auth()
        from fastapi import HTTPException

        creds = type("Creds", (), {"credentials": "wrong-token"})()
        with pytest.raises(HTTPException) as exc_info:
            await auth.verify_bearer_token(creds)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_verify_accepts_good_token(self, tmp_path, monkeypatch):
        key_file = tmp_path / "good.key"
        key_file.write_text("real-token")
        monkeypatch.setenv("CUDA_EXEC_KEY_PATH", str(key_file))
        auth = _fresh_import_auth()
        creds = type("Creds", (), {"credentials": "real-token"})()
        # Should return None (no exception).
        result = await auth.verify_bearer_token(creds)
        assert result is None
