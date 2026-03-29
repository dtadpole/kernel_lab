"""Agent orchestration using claude-agent-sdk.

Runs a single long agent session that iteratively optimizes a CUDA kernel
via MCP tools backed by the cuda_exec service.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    HookContext,
    HookInput,
    HookJSONOutput,
    HookMatcher,
    ResultMessage,
    TextBlock,
    query,
)
from claude_agent_sdk.types import PreToolUseHookSpecificOutput, SyncHookJSONOutput

from cuda_agent.prompts import SYSTEM_PROMPT, format_initial_prompt
from cuda_agent.task import OptimizationTask

_MCP_SERVER_SCRIPT = str(Path(__file__).resolve().parent / "mcp_server.py")
_BLOCKED_TOOLS_FILE = Path(__file__).resolve().parent / "blocked_tools.json"


def _load_blocked_tools_re() -> re.Pattern[str]:
    """Load the CUDA toolkit blocklist from the external config file."""
    with _BLOCKED_TOOLS_FILE.open(encoding="utf-8") as f:
        data = json.load(f)
    tools = data["blocked"]
    pattern = r"\b(" + "|".join(re.escape(t) for t in tools) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


_CUDA_TOOLKIT_RE = _load_blocked_tools_re()

# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def _log_dir_for_task(task: OptimizationTask) -> Path:
    """Build the per-run log directory, mirroring cuda_exec's layout.

    Path: ~/.cuda_agent/<run_tag>/<version>/<direction_id>_<direction_slug>/
    """
    return (
        Path.home()
        / ".cuda_agent"
        / task.run_tag
        / task.version
        / f"{task.direction_id}_{task.direction_slug}"
    )


async def _deny_direct_cuda_toolkit(
    hook_input: HookInput,
    tool_use_id: str | None,
    ctx: HookContext,
) -> HookJSONOutput:
    """PreToolUse hook: block Bash commands that invoke CUDA Toolkit binaries.

    All CUDA Toolkit usage must go through the cuda MCP tools so that
    workflow rules, auth, and logging are enforced consistently.
    """

    command = hook_input.get("tool_input", {}).get("command", "")
    if _CUDA_TOOLKIT_RE.search(command):
        return SyncHookJSONOutput(
            hookSpecificOutput=PreToolUseHookSpecificOutput(
                hookEventName="PreToolUse",
                permissionDecision="deny",
                permissionDecisionReason=(
                    "Direct CUDA Toolkit calls are not allowed. "
                    "Use the cuda MCP tools (cuda_compile, cuda_evaluate, "
                    "cuda_profile, cuda_execute) instead."
                ),
            ),
        )
    return SyncHookJSONOutput()


def _make_log_tool_use(log_dir: Path):
    """Factory: return a PostToolUse hook that writes to the given log_dir."""

    async def _log_tool_use(
        hook_input: HookInput,
        tool_use_id: str | None,
        ctx: HookContext,
    ) -> HookJSONOutput:
        """PostToolUse hook: append a JSON-lines entry for every tool invocation."""

        tool_name = hook_input.get("tool_name", "unknown")
        tool_input = hook_input.get("tool_input", {})
        tool_response = hook_input.get("tool_response")

        # Truncate large values to keep the log readable.
        input_str = json.dumps(tool_input, ensure_ascii=False, default=str)
        if len(input_str) > 1000:
            input_str = input_str[:1000] + "..."
        output_str = json.dumps(tool_response, ensure_ascii=False, default=str) if tool_response else ""
        if len(output_str) > 1000:
            output_str = output_str[:1000] + "..."

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool_name,
            "input": input_str,
            "output": output_str,
        }

        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "tool_use.jsonl"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return SyncHookJSONOutput()

    return _log_tool_use


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


async def run_optimization(task: OptimizationTask) -> str:
    """Run an iterative CUDA optimization loop and return the final summary.

    The agent decides internally when to compile, evaluate, modify code,
    and declare convergence.  ``max_turns`` on the SDK side bounds the
    total number of agent loop iterations (each tool call + response
    counts toward this limit).
    """

    prompt = format_initial_prompt(task)

    # Each optimization iteration can consume ~4-6 agent turns
    # (compile + evaluate + maybe profile + reasoning).
    max_turns = task.max_iterations * 8

    # Build env for the MCP server subprocess.  We pass through HOME and
    # key-path so the server can locate the bearer token file, plus
    # CUDA_EXEC_URL for the target service.
    mcp_env: dict[str, str] = {
        "HOME": os.environ.get("HOME", str(Path.home())),
        "CUDA_EXEC_URL": task.cuda_exec_url,
    }
    key_path = os.environ.get("CUDA_EXEC_KEY_PATH")
    if key_path:
        mcp_env["CUDA_EXEC_KEY_PATH"] = key_path

    log_dir = _log_dir_for_task(task)
    log_tool_use = _make_log_tool_use(log_dir)

    options = ClaudeAgentOptions(
        model="claude-sonnet-4-6",
        system_prompt=SYSTEM_PROMPT,
        max_turns=max_turns,
        permission_mode="bypassPermissions",
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="Bash", hooks=[_deny_direct_cuda_toolkit]),
            ],
            "PostToolUse": [
                HookMatcher(matcher=None, hooks=[log_tool_use]),
            ],
        },
        mcp_servers={
            "cuda_toolkit": {
                "command": sys.executable,
                "args": [_MCP_SERVER_SCRIPT],
                "env": mcp_env,
            },
        },
    )

    result_text = ""
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, file=sys.stderr)
        elif isinstance(message, ResultMessage):
            result_text = message.result

    return result_text
