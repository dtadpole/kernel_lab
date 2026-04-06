"""Layer 0: SDK + Infra verification.

Three checks that must pass before any agent work can begin:
  1. claude CLI is available and executable
  2. claude_agent_sdk Python package is importable
  3. End-to-end API call works (query → ResultMessage)
"""

import shutil
import subprocess


def check_cli() -> dict:
    """Verify the claude CLI is installed and reports a version."""
    path = shutil.which("claude")
    if not path:
        return {"ok": False, "error": "claude CLI not found in PATH"}

    result = subprocess.run(
        [path, "--version"],
        capture_output=True, text=True, timeout=10,
    )
    version = result.stdout.strip() or result.stderr.strip()
    if result.returncode != 0:
        return {"ok": False, "error": f"claude --version exited {result.returncode}: {version}"}

    return {"ok": True, "path": path, "version": version}


def check_sdk() -> dict:
    """Verify claude_agent_sdk is importable and has expected exports."""
    try:
        from claude_agent_sdk import (  # noqa: F401
            ClaudeAgentOptions,
            ClaudeSDKClient,
            ResultMessage,
            query,
        )
    except ImportError as e:
        return {"ok": False, "error": str(e)}

    return {"ok": True}


async def check_api() -> dict:
    """Minimal end-to-end test: send a trivial prompt, expect a ResultMessage back."""
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    try:
        result_text = None
        async for message in query(
            prompt='Respond with exactly: HELLO',
            options=ClaudeAgentOptions(
                allowed_tools=[],
                max_turns=1,
            ),
        ):
            if isinstance(message, ResultMessage):
                result_text = message.result

        if result_text is None:
            return {"ok": False, "error": "No ResultMessage received"}

        return {"ok": True, "response": result_text}

    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
