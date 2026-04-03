"""Bearer token authentication for cuda_exec.

Reads a static token from a key file **lazily on first use** and exposes a
FastAPI dependency that validates incoming ``Authorization: Bearer <token>``
headers.

The token is NOT loaded at module import time, so importing this module is
safe even when no key file exists (e.g. local-only execution).
"""

from __future__ import annotations

import hmac
import os
import sys
from pathlib import Path

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_DEFAULT_KEY_PATH = Path.home() / ".keys" / "cuda_exec.key"

_bearer_scheme = HTTPBearer()

# Sentinel indicating the token has not been loaded yet.
_UNSET = object()
_cached_token: object | str = _UNSET


def load_key() -> str:
    """Read the bearer token from disk (lazy, cached after first call).

    Raises ``RuntimeError`` when the key file is missing, unreadable, or
    empty — the caller decides how to handle the error.
    """
    return _get_token()


def _get_token() -> str:
    """Load the bearer token on first call and cache it.

    Raises ``RuntimeError`` instead of ``sys.exit`` so callers can handle
    the failure gracefully.
    """
    global _cached_token
    if _cached_token is not _UNSET:
        assert isinstance(_cached_token, str)
        return _cached_token

    key_path = Path(os.environ.get("CUDA_EXEC_KEY_PATH") or str(_DEFAULT_KEY_PATH))
    try:
        token = key_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise RuntimeError(f"cuda_exec: key file not found: {key_path}")
    except OSError as exc:
        raise RuntimeError(f"cuda_exec: cannot read key file {key_path}: {exc}")
    if not token:
        raise RuntimeError(f"cuda_exec: key file is empty: {key_path}")

    _cached_token = token
    return token


async def verify_bearer_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> None:
    """FastAPI dependency — rejects requests whose bearer token does not match
    the key loaded on first use."""

    if not hmac.compare_digest(credentials.credentials, _get_token()):
        raise HTTPException(status_code=401, detail="invalid bearer token")
