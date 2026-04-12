"""Layer 1: Agent Runner — subprocess lifecycle for one Wave.

Mental model:
  - One AgentRunner = one Wave = one subprocess (one PID)
  - A Wave may contain multiple Sessions (Steward CONTINUE)
  - API: start() → run_until_result() → send_message() → stop()

Lifecycle:
  runner = AgentRunner(config, wave=0)
  await runner.start(prompt)          # Start subprocess, write process_start.json
  result = await runner.run_until_result()  # Poll until ResultMessage (one Session)
  await runner.send_message(prompt)   # Send CONTINUE to same subprocess (new Session)
  result = await runner.run_until_result()  # Next Session
  await runner.stop()                 # Kill subprocess, write process_end.json
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    RateLimitEvent,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TaskNotificationMessage,
    TaskProgressMessage,
    TaskStartedMessage,
    TextBlock,
    ThinkingBlock,
    create_sdk_mcp_server,
    tool,
)

from agents.config import AgentConfig, MonitorConfig, StorageConfig
from agents.events import (
    AskEvent,
    DefaultHandler,
    EventHandler,
    PermissionEvent,
    StartEvent,
    StopEvent,
    SubagentEvent,
    TextOutputEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from agents.events import MonitorAlert
from agents.steward import Steward
from agents.session_log import SessionLog
from agents.storage import WaveStorage


@dataclass
class RunResult:
    """Result of a single agent run."""
    result_text: str = ""
    session_id: str = ""
    stop_reason: str = ""
    log: SessionLog = field(default_factory=SessionLog)
    usage: dict = field(default_factory=dict)


class AgentRunner:
    """One Wave = one subprocess. start() → run_until_result() → send_message() → stop()."""

    def __init__(
        self,
        agent_config: AgentConfig,
        storage_config: StorageConfig | None = None,
        handler: EventHandler | None = None,
        monitor_config: MonitorConfig | None = None,
        steward: Steward | None = None,
        cwd: str | None = None,
        agents: dict[str, AgentDefinition] | None = None,
        wave: int = 0,
        task_slug: str = "default",
    ):
        self.agent_config = agent_config
        self.storage_config = storage_config or StorageConfig()
        self.handler = handler or DefaultHandler()
        self.monitor_config = monitor_config or MonitorConfig()
        self.steward = steward
        self.cwd = cwd or str(Path.cwd())
        self.agents = agents
        self.wave = wave

        self._client: ClaudeSDKClient | None = None
        self._client_ctx = None
        self._session_id: str | None = None
        self._pid: int | None = None
        self._is_running = False
        self._solver_source = "llm"
        self._last_stream_event_time = datetime.now()
        self._strategy_terminate = False

        # Storage — one WaveStorage per Runner
        self._storage = WaveStorage(
            self.storage_config, agent_config.name, task_slug, wave
        )
        self.log = SessionLog(self._storage)

    # ── Lifecycle API ──

    async def start(self, prompt: str) -> None:
        """Start a new subprocess. Write process_start.json. Send initial prompt."""
        self._is_running = True

        # Init transcript
        self._storage.init_transcript(self.agent_config.name, prompt)
        self._storage.append_transcript(
            f"### System Prompt\n```\n{self.agent_config.system_prompt[:2000]}\n```\n"
        )
        self._storage.append_transcript(f"### User Prompt\n```\n{prompt[:2000]}\n```\n")
        self._storage.append_event({
            "ts": datetime.now().isoformat(),
            "type": "PromptEvent",
            "wave": self.wave,
            "system_prompt_length": len(self.agent_config.system_prompt),
            "user_prompt": prompt[:1000],
        })

        # Build options + start subprocess
        options = self._build_options()
        self._client_ctx = ClaudeSDKClient(options=options)
        self._client = await self._client_ctx.__aenter__()

        # Get PID
        try:
            self._pid = self._client._transport._process.pid
        except Exception:
            self._pid = None

        # Wrap transport.write() to log ALL stdin writes — catches query(),
        # inject, monitor inject, and any future code path.
        self._wrap_transport_logging()

        # Write process_start.json
        try:
            self._storage.write_process_start(
                pid=self._pid or 0,
                agent=self.agent_config.name,
                model=self.agent_config.model,
            )
        except Exception as e:
            print(f"[Runner] Failed to write process_start.json: {e}")

        # Send initial prompt
        await self._client.query(prompt)

        print(f"[Runner] {self.agent_config.name} wave {self.wave} started (PID {self._pid})")

    async def run_until_result(self) -> RunResult:
        """Poll messages until ResultMessage (one Session). Does NOT close subprocess.

        Contains liveness detection (10s poll) and strategy checks (60s).
        Raises on liveness timeout or pipe errors.
        """
        import time as _time

        if not self._client:
            raise RuntimeError("Not started. Call start() first.")

        result = RunResult(log=self.log or SessionLog())
        total_usage: dict = {}

        _POLL_INTERVAL = 10
        _STRATEGY_INTERVAL = self.monitor_config.check_interval if self.monitor_config else 60.0
        _last_strategy_check = _time.monotonic()
        self._strategy_terminate = False

        response_iter = self._client.receive_response().__aiter__()
        while True:
            try:
                message = await asyncio.wait_for(
                    response_iter.__anext__(), timeout=_POLL_INTERVAL
                )
            except asyncio.TimeoutError:
                # ── Strategy terminate flag ──
                if self._strategy_terminate:
                    result.stop_reason = "terminate"
                    result.result_text = "Terminated by strategy check"
                    stop_event = StopEvent(reason="terminate", result_text="strategy")
                    result.log.append(stop_event)
                    await self.handler.on_stop(stop_event)
                    break

                # ── Liveness check (every 10s) ──
                age = (datetime.now() - self._last_stream_event_time).total_seconds()
                if self._solver_source.startswith("tool:") or self._solver_source == "rate_limit":
                    limit = self.monitor_config.tool_timeout if self.monitor_config else 1200.0
                else:
                    limit = self.monitor_config.heartbeat_timeout if self.monitor_config else 300.0
                if age > limit:
                    reason = f"liveness_timeout ({self._solver_source}, {age:.0f}s > {limit:.0f}s)"
                    print(f"[Runner] {reason}")
                    result.stop_reason = "liveness_timeout"
                    result.result_text = reason
                    stop_event = StopEvent(reason="liveness_timeout", result_text=reason)
                    result.log.append(stop_event)
                    await self.handler.on_stop(stop_event)
                    raise RuntimeError(reason)

                # ── Strategy check (every 60s, non-blocking) ──
                now_mono = _time.monotonic()
                if now_mono - _last_strategy_check >= _STRATEGY_INTERVAL:
                    _last_strategy_check = now_mono
                    asyncio.create_task(self._run_strategy_check(result.log))

                continue
            except StopAsyncIteration:
                break

            # ── Process message ──
            source = self._process_message(message, result, total_usage)

            # Update state + heartbeat
            self._last_stream_event_time = datetime.now()
            self._solver_source = source
            if self._storage:
                try:
                    self._storage.write_heartbeat(self._last_stream_event_time, source=source)
                except Exception:
                    pass

            # Log raw stdout
            try:
                raw = json.dumps(message, default=str) if isinstance(message, dict) else str(message)
                self._storage.log_stdout(raw[:10000])
            except Exception:
                pass

        result.usage = total_usage
        return result

    async def send_message(self, prompt: str) -> None:
        """Send a new message to the same subprocess (new Session). Does NOT close pipes."""
        if not self._client:
            raise RuntimeError("Not started. Call start() first.")

        # Log as event (stdin is logged automatically by transport wrapper)
        self._storage.append_event({
            "ts": datetime.now().isoformat(),
            "type": "ContinuePrompt",
            "prompt": prompt[:1000],
        })
        self._storage.append_transcript(f"\n### Continue Prompt\n```\n{prompt[:2000]}\n```\n")

        await self._client.query(prompt)

    async def _inject_guidance(self, guidance: str, source: str) -> None:
        """Inject Steward guidance into Solver via client.query(). Logs all outcomes."""
        if not self._client:
            print(f"[Runner] Inject skipped — no client ({source})")
            return
        try:
            await self._client.query(guidance)
            print(f"[Runner] Injected steward guidance ({source}): {guidance[:80]}")
        except Exception as e:
            print(f"[Runner] Inject failed ({source}): {e}")

    async def stop(self) -> None:
        """Kill subprocess. Write process_end.json. Close all logs.

        Every step in independent try/except — must clean up even if parts fail.
        """
        print(f"[Runner] {self.agent_config.name} wave {self.wave} stopping (PID {self._pid})")

        # 1. Disconnect client (with timeout — SDK's process.wait() can hang)
        exit_code = -1
        try:
            if self._client_ctx:
                await asyncio.wait_for(
                    self._client_ctx.__aexit__(None, None, None),
                    timeout=60,
                )
        except asyncio.TimeoutError:
            print(f"[Runner] disconnect timed out after 60s, force-killing PID {self._pid}")
            self._force_kill_pid(self._pid)
        except (asyncio.CancelledError, Exception) as e:
            print(f"[Runner] disconnect: {e}")

        # 2. Get exit code
        try:
            if self._client and self._client._transport and self._client._transport._process:
                exit_code = self._client._transport._process.returncode or -1
        except Exception as e:
            print(f"[Runner] exit code: {e}")

        # 3. Write process_end.json
        try:
            self._storage.write_process_end(self._pid or 0, exit_code)
        except Exception as e:
            print(f"[Runner] process_end.json: {e}")

        # 4. Write meta.json
        try:
            self._storage.write_meta({
                "agent": self.agent_config.name,
                "wave": self.wave,
                "model": self.agent_config.model,
                "pid": self._pid,
                "exit_code": exit_code,
            })
        except Exception as e:
            print(f"[Runner] meta.json: {e}")

        # 5. Close log files
        try:
            self._storage.close_logs()
        except Exception as e:
            print(f"[Runner] close logs: {e}")

        # 6. Reset state
        self._client = None
        self._client_ctx = None
        self._is_running = False

        print(f"[Runner] {self.agent_config.name} wave {self.wave} stopped (exit_code={exit_code})")

    def _force_kill_pid(self, pid: int | None) -> None:
        """Force-kill a subprocess by PID. Bypasses SDK when it hangs."""
        if not pid:
            return
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass

    def _wrap_transport_logging(self) -> None:
        """Intercept transport.write() to log all stdin to the wave's stdin.log.

        This is the single chokepoint for ALL writes to the subprocess stdin —
        client.query(), inject, monitor inject, and any future code path all
        go through transport.write(). By wrapping here, we guarantee complete
        stdin logging without requiring each call site to remember to log.
        """
        try:
            transport = self._client._transport
            original_write = transport.write

            async def logged_write(data: str) -> None:
                try:
                    self._storage.log_stdin(data[:5000])
                except Exception:
                    pass
                return await original_write(data)

            transport.write = logged_write
        except Exception as e:
            print(f"[Runner] Failed to wrap transport logging: {e}")

    # ── Legacy API (backward compat for Steward) ──

    async def run(self, prompt: str, task_slug: str = "default") -> RunResult:
        """Run a complete wave: start → run_until_result → stop."""
        # Re-create storage with correct task_slug if needed
        if task_slug != "default":
            self._storage = WaveStorage(
                self.storage_config, self.agent_config.name, task_slug, self.wave
            )
            self.log = SessionLog(self._storage)

        await self.start(prompt)
        try:
            result = await self.run_until_result()
        except Exception:
            result = RunResult(log=self.log or SessionLog())
            result.stop_reason = "error"
        finally:
            await self.stop()
        return result

    @property
    def is_running(self) -> bool:
        return self._is_running

    def _build_options(self) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions from AgentConfig + hooks + MCP tools."""
        ac = self.agent_config

        # Build allowed_tools: builtin tools + MCP-prefixed custom tools
        allowed = list(ac.builtin_tools)
        for custom_tool in ac.custom_tools:
            # MCP tools get prefixed as mcp__<server>__<tool> by the SDK
            allowed.append(custom_tool)
            allowed.append(f"mcp__supervisor-tools__{custom_tool}")

        opts = ClaudeAgentOptions(
            cwd=self.cwd,
            allowed_tools=allowed,
            disallowed_tools=["Skill", "Agent", "TodoWrite", "TodoRead"] + list(ac.disallowed_tools),
            system_prompt=ac.system_prompt,
            permission_mode=ac.permission_mode,
            max_turns=ac.max_turns,
            model=ac.model,
            hooks=self._build_hooks(),
            thinking={"type": "enabled", "budget_tokens": 10000},
            include_partial_messages=True,
            stderr=self._on_stderr,
            env={
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000",
                "CUDA_EXEC_RUN_TAG": self.storage_config.resolved_run_tag,
            },
            # Only load our MCP servers, skip all default plugins
            # (data/datamate/meta/calendar plugins cause stream closed crashes)
            extra_args={"strict-mcp-config": True},
        )

        if ac.max_budget_usd > 0:
            opts.max_budget_usd = ac.max_budget_usd

        # Custom MCP tools
        mcp_tools = self._build_mcp_tools()
        if mcp_tools:
            opts.mcp_servers = mcp_tools

        # Subagent definitions
        if self.agents:
            opts.agents = self.agents

        return opts

    def _on_stderr(self, line: str) -> None:
        """Callback for CLI stderr — log to wave folder + update heartbeat."""
        try:
            self._storage.log_stderr(line)
        except Exception:
            pass
        self._last_stream_event_time = datetime.now()
        try:
            self._storage.write_heartbeat(self._last_stream_event_time, source="stderr")
        except Exception:
            pass

    def _build_hooks(self) -> dict:
        """Bridge SDK hook callbacks to our EventHandler."""
        handler = self.handler
        log = self.log

        async def on_pre_tool_use(input_data, tool_use_id, context):
            import sys
            import traceback as _tb
            try:
                return await _pre_tool_use_inner(input_data, tool_use_id, context)
            except Exception as exc:
                # Capture the ACTUAL Python exception so we can diagnose hook failures
                err_msg = f"[PreToolUse CRASH] {type(exc).__name__}: {exc}\n{''.join(_tb.format_exception(exc))}"
                print(err_msg, file=sys.stderr, flush=True)
                try:
                    with open("/tmp/pre_tool_hook_crash.log", "a") as f:
                        f.write(f"[{datetime.now().isoformat()}] {err_msg}\n")
                except Exception:
                    pass
                return {}  # allow tool to proceed on hook failure

        async def _pre_tool_use_inner(input_data, tool_use_id, context):
            # Universal hook trace — always fires, before any logic
            import json as _json
            _trace = {
                "ts": datetime.now().isoformat(),
                "hook": "PreToolUse",
                "tool": input_data.get("tool_name", ""),
                "input_keys": list(input_data.get("tool_input", {}).keys()),
            }
            try:
                with open("/tmp/hook_trace.jsonl", "a") as _f:
                    _f.write(_json.dumps(_trace) + "\n")
                    _f.flush()
            except Exception:
                pass

            tool_name = input_data.get("tool_name", "")
            tool_input = input_data.get("tool_input", {})

            # Track phase: entering tool execution
            self._solver_source = f"tool:{tool_name}"
            if self._storage:
                self._storage.write_heartbeat(datetime.now(), source=self._solver_source)

            event = ToolCallEvent(
                tool_name=tool_name,
                tool_input=tool_input,
                tool_use_id=tool_use_id or "",
            )
            if log:
                log.append(event)
            await handler.on_tool_call(event)

            result = self._check_tool_rules(tool_name, tool_input)

            # Direction gate: block writes to watched dirs without approved direction
            # Run unless tool_rules explicitly denied (permissionDecision=deny).
            # A constraint-only result (additionalContext) is NOT a deny — gate must still check.
            tool_denied = (
                isinstance(result, dict)
                and result.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
            )
            needs_direction = False
            if not tool_denied:
                gate_dirs = getattr(handler, '_direction_gate_dirs', None) or []

                gate_tools = getattr(handler, '_direction_gate_tools', [])
                if tool_name in gate_tools:
                    if tool_name in ("Write", "Edit"):
                        path = tool_input.get("file_path", "")
                        if any(path.startswith(d) for d in gate_dirs):
                            needs_direction = True
                    elif tool_name == "Bash":
                        cmd = tool_input.get("command", "")
                        if any(w in cmd for w in [" > ", " >> ", "tee ", "cp ", "mv ", "sed -i"]):
                            if any(d in cmd for d in gate_dirs):
                                needs_direction = True

                if needs_direction and hasattr(handler, 'state') and not getattr(handler.state, 'current_direction', None):
                    gate_msg = getattr(handler, '_direction_gate_message', "You must set_direction first.")
                    result = {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": gate_msg,
                        }
                    }

                # Log direction gate decision
                if needs_direction and self._storage:
                    direction = getattr(handler.state, 'current_direction', None) if hasattr(handler, 'state') else None
                    gate_denied = isinstance(result, dict) and result.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
                    decision = "deny" if gate_denied else "allow"
                    target = tool_input.get("file_path", "") or tool_input.get("command", "")[:200]
                    self._storage.append_event({
                        "ts": datetime.now().isoformat(),
                        "type": "PreToolHook",
                        "subtype": "direction_gate",
                        "tool": tool_name,
                        "target": target,
                        "direction": direction,
                        "decision": decision,
                    })

            # Trace final decision
            try:
                _decision = "deny" if isinstance(result, dict) and result.get("hookSpecificOutput", {}).get("permissionDecision") == "deny" else "allow"
                _trace_end = {
                    "ts": datetime.now().isoformat(),
                    "hook": "PreToolUse_result",
                    "tool": tool_name,
                    "target": tool_input.get("file_path", "") or tool_input.get("command", "")[:100],
                    "tool_rules": "deny" if tool_denied else "pass",
                    "gate": "deny" if (needs_direction and not getattr(handler.state, 'current_direction', None)) else "pass" if needs_direction else "n/a",
                    "final": _decision,
                }
                with open("/tmp/hook_trace.jsonl", "a") as _f:
                    _f.write(_json.dumps(_trace_end) + "\n")
                    _f.flush()
            except Exception:
                pass

            return result if result is not None else {}

        async def on_post_tool_use(input_data, tool_use_id, context):
            # Universal hook trace
            import json as _json
            try:
                with open("/tmp/hook_trace.jsonl", "a") as _f:
                    _f.write(_json.dumps({
                        "ts": datetime.now().isoformat(),
                        "hook": "PostToolUse",
                        "tool": input_data.get("tool_name", ""),
                    }) + "\n")
                    _f.flush()
            except Exception:
                pass

            tool_name = input_data.get("tool_name", "")
            tool_response = input_data.get("tool_response", "")

            # Track phase: tool execution complete, back to LLM
            self._solver_source = "tool_end"
            if self._storage:
                self._storage.write_heartbeat(datetime.now(), source=self._solver_source)

            summary = str(tool_response)
            if len(summary) > 500:
                summary = summary[:1000] + "..."

            event = ToolResultEvent(
                tool_name=tool_name,
                tool_use_id=tool_use_id or "",
                result_summary=summary,
            )
            if log:
                log.append(event)
            await handler.on_tool_result(event)
            return {}

        async def on_post_tool_failure(input_data, tool_use_id, context):
            tool_name = input_data.get("tool_name", "")
            error = input_data.get("error", "unknown error")

            event = ToolResultEvent(
                tool_name=tool_name,
                tool_use_id=tool_use_id or "",
                result_summary=str(error)[:1000],
                is_error=True,
            )
            if log:
                log.append(event)
            await handler.on_tool_result(event)
            return {}

        async def on_subagent_start(input_data, tool_use_id, context):
            event = SubagentEvent(
                agent_id=input_data.get("agent_id", ""),
                agent_type=input_data.get("agent_type", ""),
                action="start",
            )
            if log:
                log.append(event)
            return {}

        async def on_subagent_stop(input_data, tool_use_id, context):
            event = SubagentEvent(
                agent_id=input_data.get("agent_id", ""),
                agent_type=input_data.get("agent_type", ""),
                action="stop",
            )
            if log:
                log.append(event)
            return {}

        async def on_permission_request(input_data, tool_use_id, context):
            tool_name = input_data.get("tool_name", "")
            tool_input = input_data.get("tool_input", {})

            event = PermissionEvent(tool_name=tool_name, tool_input=tool_input)
            if log:
                log.append(event)
            allowed = await handler.on_permission(event)

            if allowed:
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PermissionRequest",
                        "decision": {"allow": True},
                    }
                }
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PermissionRequest",
                    "decision": {"deny": True, "reason": "Denied by Supervisor"},
                }
            }

        return {
            "PreToolUse": [HookMatcher(matcher=None, hooks=[on_pre_tool_use])],
            "PostToolUse": [HookMatcher(matcher=None, hooks=[on_post_tool_use])],
            "PostToolUseFailure": [HookMatcher(matcher=None, hooks=[on_post_tool_failure])],
            "SubagentStart": [HookMatcher(matcher=None, hooks=[on_subagent_start])],
            "SubagentStop": [HookMatcher(matcher=None, hooks=[on_subagent_stop])],
            "PermissionRequest": [HookMatcher(matcher=None, hooks=[on_permission_request])],
        }

    def _build_mcp_tools(self) -> dict:
        """Create custom MCP tools (ask_supervisor, request_formal_bench, etc.)."""
        custom = self.agent_config.custom_tools
        if not custom:
            return {}

        runner_ref = self  # closure reference
        tools_list = []

        if "ask_supervisor" in custom:
            @tool(
                "ask_supervisor",
                "Ask the Supervisor a question and get guidance or a decision.",
                {"question": str, "context": str},
            )
            async def ask_supervisor(args):
                question = args.get("question", "")
                ctx = args.get("context", "")

                event = AskEvent(question=question, context=ctx)
                if runner_ref.log:
                    runner_ref.log.append(event)

                # Try Steward first, fall back to handler
                if runner_ref.steward and runner_ref.log:
                    summary = runner_ref.log.to_summary()
                    answer = await runner_ref.steward.answer(question, summary)
                else:
                    answer = await runner_ref.handler.on_ask(event)

                return {"content": [{"type": "text", "text": answer}]}

            tools_list.append(ask_supervisor)

        if "request_formal_bench" in custom:
            @tool(
                "request_formal_bench",
                "Request the Supervisor to run a formal benchmark. "
                "You cannot run ik:bench yourself — only the Supervisor can schedule it. "
                "The Supervisor controls GPU and impl selection — just specify kernel and reason.",
                {
                    "kernel": str,
                    "reason": str,
                },
            )
            async def request_formal_bench(args):
                kernel = args.get("kernel", "")
                reason = args.get("reason", "")

                query_parts = [f"kernel={kernel}"]

                event = AskEvent(
                    question=f"REQUEST_FORMAL_BENCH: {' '.join(query_parts)}",
                    context=f"Reason: {reason}",
                )
                if runner_ref.log:
                    runner_ref.log.append(event)

                answer = await runner_ref.handler.on_ask(event)
                return {"content": [{"type": "text", "text": answer}]}

            tools_list.append(request_formal_bench)

        if "submit_bench_reflection" in custom:
            @tool(
                "submit_bench_reflection",
                "Submit reflection after a formal benchmark. Call this every time "
                "after request_formal_bench returns results. "
                "gem_id and gem_notes_md are only needed when a new gem was produced. "
                "reflection_md is always required.",
                {
                    "gem_id": str,          # optional: e.g. "gen-cuda/v003", only if gem produced
                    "gem_notes_md": str,    # optional: Markdown implementation notes for gem
                    "reflection_md": str,   # required: Markdown reflection (at most 3 points)
                },
            )
            async def submit_bench_reflection(args):
                gem_id = args.get("gem_id", "")
                gem_notes_md = args.get("gem_notes_md", "")
                reflection_md = args.get("reflection_md", "")

                # Pack both into the event
                context = json.dumps({
                    "gem_id": gem_id,
                    "gem_notes_md": gem_notes_md,
                    "reflection_md": reflection_md,
                })

                event = AskEvent(
                    question="SUBMIT_BENCH_REFLECTION",
                    context=context,
                )
                if runner_ref.log:
                    runner_ref.log.append(event)

                answer = await runner_ref.handler.on_ask(event)
                return {"content": [{"type": "text", "text": answer}]}

            tools_list.append(submit_bench_reflection)

        if "set_direction" in custom:
            @tool(
                "set_direction",
                "Set your optimization direction after brainstorming. "
                "Every call is reviewed by the Steward. Pass a JSON string with: "
                "name, description, opportunity, evidence, ideas.",
                {"direction_json": str},
            )
            async def set_direction(args):
                direction_json = args.get("direction_json", "{}")

                event = AskEvent(
                    question="SET_DIRECTION",
                    context=direction_json,
                )
                if runner_ref.log:
                    runner_ref.log.append(event)

                answer = await runner_ref.handler.on_ask(event)
                return {"content": [{"type": "text", "text": answer}]}

            tools_list.append(set_direction)

        if "start_exploring" in custom:
            @tool(
                "start_exploring",
                "Request to enter brainstorming mode for a new direction. "
                "Call this when you believe the current direction is exhausted. "
                "Explain what you tried and why you think no further progress is possible.",
                {"reason": str},
            )
            async def start_exploring(args):
                reason = args.get("reason", "")

                event = AskEvent(
                    question="START_EXPLORING",
                    context=reason,
                )
                if runner_ref.log:
                    runner_ref.log.append(event)

                answer = await runner_ref.handler.on_ask(event)
                return {"content": [{"type": "text", "text": answer}]}

            tools_list.append(start_exploring)

        # ── Library system tools ──

        if "consult_taxonomist" in custom:
            @tool(
                "consult_taxonomist",
                "Ask the Taxonomist for advice on structure, classification, "
                "page boundaries, naming, and merge/split decisions. "
                "Use when you're unsure where a piece of knowledge belongs.",
                {"question": str, "proposal_context": str},
            )
            async def consult_taxonomist(args):
                question = args.get("question", "")
                ctx = args.get("proposal_context", "")

                event = AskEvent(
                    question=f"CONSULT_TAXONOMIST: {question}",
                    context=ctx,
                )
                if runner_ref.log:
                    runner_ref.log.append(event)

                answer = await runner_ref.handler.on_ask(event)
                return {"content": [{"type": "text", "text": answer}]}

            tools_list.append(consult_taxonomist)

        if "consult_auditor" in custom:
            @tool(
                "consult_auditor",
                "Ask the Auditor to validate evidence, check for conflicts "
                "with existing wiki content, and detect over-generalization. "
                "Use when evidence is weak or the claim is high-impact.",
                {"question": str, "proposal_context": str},
            )
            async def consult_auditor(args):
                question = args.get("question", "")
                ctx = args.get("proposal_context", "")

                event = AskEvent(
                    question=f"CONSULT_AUDITOR: {question}",
                    context=ctx,
                )
                if runner_ref.log:
                    runner_ref.log.append(event)

                answer = await runner_ref.handler.on_ask(event)
                return {"content": [{"type": "text", "text": answer}]}

            tools_list.append(consult_auditor)

        if not tools_list:
            return {}

        server = create_sdk_mcp_server("supervisor-tools", tools=tools_list)
        return {"supervisor-tools": server}

    def _resolve_path_pattern(self, pattern: str) -> str:
        """Resolve <run_tag> and ~ in path patterns."""
        import os
        run_tag = self.storage_config.resolved_run_tag
        resolved = pattern.replace("<run_tag>", run_tag)
        resolved = os.path.expanduser(resolved)
        # Resolve relative paths against cwd
        if not os.path.isabs(resolved):
            resolved = os.path.join(self.cwd, resolved)
        return resolved

    def _is_path_blocked(self, path: str, rule) -> bool:
        """Check if a resolved absolute path is blocked by a rule.

        Uses "most specific prefix wins" semantics:
        - Find the longest matching blocked_path prefix
        - Find the longest matching allowed_path prefix
        - The longer (more specific) match determines the result
        - This allows re-blocking subdirectories within an allowed area
          (e.g., block peak/ within an allowed run directory)
        """
        import os
        path_dir = path.rstrip("/") + "/"

        best_blocked_len = 0
        for blocked in rule.blocked_paths:
            blocked_resolved = self._resolve_path_pattern(blocked).rstrip("/") + "/"
            if path_dir.startswith(blocked_resolved) or path == blocked_resolved.rstrip("/"):
                best_blocked_len = max(best_blocked_len, len(blocked_resolved))

        if best_blocked_len == 0:
            return False  # Not in any blocked path

        best_allowed_len = 0
        for allowed in rule.allowed_paths:
            allowed_resolved = self._resolve_path_pattern(allowed).rstrip("/") + "/"
            if path_dir.startswith(allowed_resolved) or path == allowed_resolved.rstrip("/"):
                best_allowed_len = max(best_allowed_len, len(allowed_resolved))

        # Most specific match wins
        return best_blocked_len > best_allowed_len

    @staticmethod
    def _deny(reason: str) -> dict:
        """Return a PreToolUse hook output that blocks the tool call."""
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    # Navigation commands — allowed on any path (don't read file content)
    _NAV_COMMANDS = re.compile(
        r'^\s*(ls|mkdir|touch|dirname|basename|pwd)\b'
    )
    # Read-only commands that should bypass blocked_paths even when
    # the command string contains a blocked directory (e.g., cd ~/kernel_lab && ...)
    _READONLY_COMMANDS = re.compile(
        r'doc_retrieval\s+(find|read|browse)\b'
        r'|nvidia-smi\b'
    )
    # Recursive ls — blocked (would expose subdirectory contents)
    _RECURSIVE_LS = re.compile(r'\bls\b.*\s+-\S*R')
    # Commands that read content or compile/execute — check blocked paths
    _DANGEROUS_COMMANDS = re.compile(
        r'\b(cat|head|tail|less|more|strings|xxd|od|hexdump|tac|nl|'
        r'wc|grep|awk|sed|sort|uniq|cut|tr|diff|comm|paste|'
        r'nvcc|g\+\+|gcc|clang|cp|mv|ln|source|\.)\b'
    )
    # Commands that are always forbidden regardless of path or tool_rules.
    # Checked for ALL agents — even those with no tool_rules.
    _FORBIDDEN_COMMANDS = re.compile(
        r'\b(git|gh|kill|pkill|killall|sudo|reboot|shutdown)\b'
    )

    def _extract_paths_from_command(self, command: str) -> list[str]:
        """Extract file paths from dangerous commands in a Bash command string.

        Skips navigation commands (ls, find, tree, etc.) — those can access
        any directory. Only checks commands that read content, compile, or
        execute files from blocked paths.
        """
        import os
        # Split on pipes/semicolons and check each subcommand
        # But for simplicity, check the whole command
        if self._NAV_COMMANDS.match(command) and not self._DANGEROUS_COMMANDS.search(command):
            return []
        if self._READONLY_COMMANDS.search(command):
            return []
        paths = []
        for match in re.finditer(r'(?:~/|/|[a-zA-Z0-9_.]+/)[a-zA-Z0-9_./<>\-]*', command):
            raw = match.group(0)
            resolved = os.path.expanduser(raw)
            if not os.path.isabs(resolved):
                resolved = os.path.join(self.cwd, resolved)
            paths.append(resolved)
        return paths

    def _check_tool_rules(self, tool_name: str, tool_input: dict) -> dict:
        """Enforce tool_rules from config. Returns hook output dict."""
        import os

        # ── Global forbidden commands — checked for ALL agents regardless of tool_rules ──
        if tool_name == "Bash":
            command = tool_input.get("command", "")
            if command and self._FORBIDDEN_COMMANDS.search(command):
                return self._deny(f"Forbidden command detected in: {command[:100]}")
            if command and self._RECURSIVE_LS.search(command):
                return self._deny("Recursive ls (-R) is not allowed. Use ls <dir> instead.")

        # ── Per-agent tool_rules ──
        for rule in self.agent_config.tool_rules:
            if rule.tool == tool_name:
                if not rule.allow:
                    return self._deny(f"Tool '{tool_name}' is not allowed for this agent role.")

                if not rule.blocked_paths:
                    # No path restrictions — just apply constraint
                    if rule.constraint:
                        return {
                            "hookSpecificOutput": {
                                "hookEventName": "PreToolUse",
                                "additionalContext": f"Constraint: {rule.constraint}",
                            }
                        }
                    return {}

                # ── Path-based tools (Read, Glob, Grep): check file_path/path ──
                if tool_name in ("Read", "Glob", "Grep"):
                    file_path = (
                        tool_input.get("file_path", "")
                        or tool_input.get("path", "")
                    )
                    if file_path and not file_path.startswith("{"):
                        resolved = os.path.expanduser(file_path)
                        if not os.path.isabs(resolved):
                            resolved = os.path.join(self.cwd, resolved)
                        if self._is_path_blocked(resolved, rule):
                            return self._deny(f"Access denied: '{file_path}' is blocked.")

                # ── Bash: scan commands for blocked paths ──
                elif tool_name == "Bash":
                    command = tool_input.get("command", "")
                    if command:
                        for path in self._extract_paths_from_command(command):
                            if self._is_path_blocked(path, rule):
                                return self._deny(f"Access denied: command references blocked path '{path}'.")

                if rule.constraint:
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "additionalContext": f"Constraint: {rule.constraint}",
                        }
                    }
        return {}

    def _process_message(self, message, result: RunResult, total_usage: dict) -> str:
        """Process a single SDK message. Returns source string for heartbeat."""
        source = "unknown"

        if isinstance(message, SystemMessage):
            source = "system_init"
            if message.subtype == "init":
                sid = message.data.get("session_id", "")
                self._session_id = sid
                result.session_id = sid
                start_event = StartEvent(session_id=sid)
                result.log.append(start_event)

        elif isinstance(message, AssistantMessage):
            has_thinking = False
            has_text = False
            for block in message.content:
                if isinstance(block, TextBlock):
                    has_text = True
                    event = TextOutputEvent(text=block.text)
                    result.log.append(event)
                    # on_text is sync in Supervisor (just pass), safe to call
                    asyncio.get_event_loop().create_task(self.handler.on_text(event))
                elif isinstance(block, ThinkingBlock):
                    has_thinking = True
            if has_text:
                source = "text"
            elif has_thinking:
                source = "thinking"
            else:
                source = "assistant"
            if message.usage:
                for k, v in message.usage.items():
                    if isinstance(v, (int, float)):
                        total_usage[k] = total_usage.get(k, 0) + v

        elif isinstance(message, ResultMessage):
            source = "result"
            result.result_text = message.result or ""
            result.stop_reason = getattr(message, "stop_reason", "end_turn")
            stop_event = StopEvent(
                reason=result.stop_reason,
                result_text=result.result_text[:5000],
            )
            result.log.append(stop_event)
            asyncio.get_event_loop().create_task(self.handler.on_stop(stop_event))

        elif isinstance(message, RateLimitEvent):
            source = "rate_limit"
            info = message.rate_limit_info
            status = getattr(info, "status", None)
            resets_at = getattr(info, "resets_at", None)
            utilization = getattr(info, "utilization", None)
            event = TextOutputEvent(
                text=f"[rate_limit] status={status} utilization={utilization} resets_at={resets_at}"
            )
            result.log.append(event)
            if status and hasattr(status, "value") and "rejected" in str(status.value):
                self._solver_source = "rate_limit"

        elif isinstance(message, TaskStartedMessage):
            source = "subagent_start"
            event = SubagentEvent(
                agent_id=getattr(message, "task_id", ""),
                agent_type="subagent", action="start",
            )
            result.log.append(event)

        elif isinstance(message, TaskProgressMessage):
            source = "subagent_progress"
            tool_name = getattr(message, "last_tool_name", "")
            if tool_name:
                event = TextOutputEvent(text=f"[subagent_progress] tool={tool_name}")
                result.log.append(event)

        elif isinstance(message, TaskNotificationMessage):
            source = "subagent_done"
            event = SubagentEvent(
                agent_id=getattr(message, "task_id", ""),
                agent_type="subagent", action="stop",
            )
            result.log.append(event)

        elif isinstance(message, StreamEvent):
            source = "stream"

        return source

    def _write_monitor_log(self, entry: dict) -> None:
        """Append one JSON line to monitor.jsonl in the wave directory."""
        try:
            monitor_path = self._storage._wave_dir / "monitor.jsonl"
            with open(monitor_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass

    async def _run_strategy_check(self, log: SessionLog) -> None:
        """Strategy check — runs as create_task, never blocks main loop.

        Checks hard_limit, stuck detection, loop detection, progress.
        Calls Steward if needed. Results delivered via client.query().
        """
        try:
            if not self.monitor_config:
                return

            elapsed = log.elapsed().total_seconds()
            idle = log.last_event_age().total_seconds()
            age = (datetime.now() - self._last_stream_event_time).total_seconds()

            # Log this check
            monitor_entry = {
                "ts": datetime.now().isoformat(),
                "elapsed_s": round(elapsed, 1),
                "idle_s": round(idle, 1),
                "stream_age_s": round(age, 1),
                "source": self._solver_source,
                "events": len(log._events) if hasattr(log, '_events') else 0,
                "tool_calls": sum(1 for e in (log._events if hasattr(log, '_events') else []) if getattr(e, 'tool_name', None)),
                "pid": self._pid,
                "pid_alive": self._pid is not None and os.path.exists(f"/proc/{self._pid}"),
                "alerts": [],
                "actions": [],
            }

            # Hard limit — non-negotiable
            if elapsed > self.monitor_config.hard_limit:
                alert = MonitorAlert(
                    alert_type="hard_limit",
                    details=f"Session hit hard limit: {elapsed:.0f}s",
                )
                log.append(alert)
                monitor_entry["alerts"].append("hard_limit")
                action = await self.handler.on_monitor_alert(alert)
                monitor_entry["actions"].append(f"hard_limit→{action}")
                if action == "terminate":
                    self._strategy_terminate = True
                self._write_monitor_log(monitor_entry)
                return

            # Total timeout
            if elapsed > self.monitor_config.total_timeout:
                alert = MonitorAlert(
                    alert_type="total_timeout",
                    details=f"Session running for {elapsed:.0f}s",
                )
                log.append(alert)
                monitor_entry["alerts"].append("total_timeout")
                action = await self.handler.on_monitor_alert(alert)
                monitor_entry["actions"].append(f"total_timeout→{action}")
                if action == "terminate":
                    self._strategy_terminate = True
                elif action and action.startswith("inject:"):
                    await self._inject_guidance(action[len("inject:"):], "total_timeout")
                self._write_monitor_log(monitor_entry)
                return

            # Idle timeout
            if idle > self.monitor_config.idle_timeout:
                alert = MonitorAlert(
                    alert_type="idle_timeout",
                    details=f"No activity for {idle:.0f}s",
                )
                log.append(alert)
                monitor_entry["alerts"].append("idle_timeout")
                action = await self.handler.on_monitor_alert(alert)
                monitor_entry["actions"].append(f"idle_timeout→{action}")
                if action == "terminate":
                    self._strategy_terminate = True
                elif action == "interrupt" and self._client:
                    try:
                        await asyncio.wait_for(self._client.interrupt(), timeout=5)
                    except (asyncio.TimeoutError, Exception):
                        pass
                elif action and action.startswith("inject:"):
                    await self._inject_guidance(action[len("inject:"):], "idle_timeout")
                self._write_monitor_log(monitor_entry)
                return

            # Loop detection
            seq = log.recent_tool_sequence(self.monitor_config.loop_threshold)
            if len(seq) >= self.monitor_config.loop_threshold and len(set(seq)) == 1:
                alert = MonitorAlert(
                    alert_type="loop_detected",
                    details=f"Tool '{seq[0]}' called {len(seq)} times consecutively",
                )
                log.append(alert)
                monitor_entry["alerts"].append(f"loop_detected:{seq[0]}")
                action = await self.handler.on_monitor_alert(alert)
                monitor_entry["actions"].append(f"loop→{action}")
                if action and action.startswith("inject:"):
                    await self._inject_guidance(action[len("inject:"):], "loop_detected")
                self._write_monitor_log(monitor_entry)
                return

            # Progress check
            if self.monitor_config.progress_check_interval > 0:
                if not hasattr(self, "_last_progress_elapsed"):
                    self._last_progress_elapsed = 0.0
                if elapsed - self._last_progress_elapsed >= self.monitor_config.progress_check_interval:
                    self._last_progress_elapsed = elapsed
                    alert = MonitorAlert(
                        alert_type="progress_check",
                        details=f"Periodic progress check at {elapsed:.0f}s",
                    )
                    log.append(alert)
                    monitor_entry["alerts"].append("progress_check")
                    action = await self.handler.on_monitor_alert(alert)
                    monitor_entry["actions"].append(f"progress_check→{action}")
                    if action == "terminate":
                        self._strategy_terminate = True
                    elif action and action.startswith("inject:"):
                        await self._inject_guidance(action[len("inject:"):], "progress_check")

            # No alerts triggered — healthy
            self._write_monitor_log(monitor_entry)

        except Exception as e:
            print(f"[Runner] Strategy check error: {e}")
