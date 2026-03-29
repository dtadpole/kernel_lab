"""Bearer token authentication for cuda_exec.

Reads a static token from a key file at startup and exposes a FastAPI
dependency that validates incoming ``Authorization: Bearer <token>`` headers.
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


def load_key() -> str:
    """Read the bearer token from disk.  Exits the process when the key is
    missing or empty so the service never runs unauthenticated."""

    key_path = Path(os.environ.get("CUDA_EXEC_KEY_PATH") or str(_DEFAULT_KEY_PATH))
    try:
        token = key_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        print(f"cuda_exec: key file not found: {key_path}", file=sys.stderr)
        raise SystemExit(1)
    except OSError as exc:
        print(f"cuda_exec: cannot read key file {key_path}: {exc}", file=sys.stderr)
        raise SystemExit(1)
    if not token:
        print(f"cuda_exec: key file is empty: {key_path}", file=sys.stderr)
        raise SystemExit(1)
    return token


_BEARER_KEY: str = load_key()


async def verify_bearer_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> None:
    """FastAPI dependency — rejects requests whose bearer token does not match
    the key loaded at startup."""

    if not hmac.compare_digest(credentials.credentials, _BEARER_KEY):
        raise HTTPException(status_code=401, detail="invalid bearer token")
