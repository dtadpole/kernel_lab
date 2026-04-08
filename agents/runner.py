"""Layer 1: Agent Runner — ClaudeSDKClient wrapper with hooks, monitoring, and logging.

Bridges the Claude Agent SDK to the event system. Provides run/resume/interrupt
operations and integrates SessionLog, Monitor, Steward, and custom MCP tools.
"""

from __future__ import annotations

import asyncio
import json
import re
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
from agents.monitor import AgentMonitor
from agents.steward import Steward
from agents.session_log import SessionLog
from agents.storage import SessionStorage


@dataclass
class RunResult:
    """Result of a single agent run."""
    result_text: str = ""
    session_id: str = ""
    stop_reason: str = ""
    log: SessionLog = field(default_factory=SessionLog)
    usage: dict = field(default_factory=dict)


class AgentRunner:
    """Layer 1: Thin wrapper over ClaudeSDKClient with event-driven hooks.

    Responsibilities:
      - Launch/resume/interrupt agent sessions
      - Bridge SDK hooks → AgentEvent → EventHandler
      - Manage SessionLog + SessionStorage
      - Run AgentMonitor as parallel task
      - Provide ask_supervisor MCP tool for Solver → Supervisor communication
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        storage_config: StorageConfig | None = None,
        handler: EventHandler | None = None,
        monitor_config: MonitorConfig | None = None,
        steward: Steward | None = None,
        cwd: str | None = None,
        agents: dict[str, AgentDefinition] | None = None,
    ):
        self.agent_config = agent_config
        self.storage_config = storage_config or StorageConfig()
        self.handler = handler or DefaultHandler()
        self.monitor_config = monitor_config or MonitorConfig()
        self.steward = steward
        self.cwd = cwd or str(Path.cwd())
        self.agents = agents

        self._client: ClaudeSDKClient | None = None
        self._session_id: str | None = None
        self._monitor: AgentMonitor | None = None
        self._is_running = False

        # Will be initialized per run
        self.log: SessionLog | None = None
        self._storage: SessionStorage | None = None

    async def run(self, prompt: str, task_slug: str = "default") -> RunResult:
        """Execute a full agent session."""
        # 1. Set up storage and log
        self._storage = SessionStorage(
            self.storage_config, self.agent_config.name, task_slug
        )
        self.log = SessionLog(self._storage)

        # Init transcript with full prompt and system prompt
        self._storage.init_transcript(self.agent_config.name, prompt)
        self._storage.append_transcript(f"### System Prompt\n```\n{self.agent_config.system_prompt}\n```\n")
        self._storage.append_transcript(f"### User Prompt\n```\n{prompt}\n```\n")

        # Log prompt as first event
        self._storage.append_event({
            "ts": datetime.now().isoformat(),
            "type": "PromptEvent",
            "system_prompt_length": len(self.agent_config.system_prompt),
            "user_prompt": prompt[:1000],
        })

        # 2. Build options
        options = self._build_options()

        # 3. Run
        return await self._execute(prompt, options)

    async def resume(self, prompt: str) -> RunResult:
        """Resume an existing session with a new prompt."""
        if not self._session_id:
            raise RuntimeError("No session to resume. Call run() first.")
        if self.log is None:
            self.log = SessionLog()

        # Ensure previous monitor is fully stopped before creating new client
        if self._monitor:
            await self._monitor.stop()
            self._monitor = None

        options = self._build_options()
        options.resume = self._session_id
        return await self._execute(prompt, options)

    async def interrupt(self) -> None:
        """Interrupt the currently running agent (graceful)."""
        if self._client:
            await self._client.interrupt()

    async def terminate(self) -> None:
        """Terminate the agent process (hard kill). Used for hard_limit."""
        if self._client:
            await self._client.disconnect()
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def last_stream_age_seconds(self) -> float:
        """Seconds since the last stream event (heartbeat). 0 if no events yet."""
        if not hasattr(self, "_last_stream_event_time") or self._last_stream_event_time is None:
            return 0.0
        return (datetime.now() - self._last_stream_event_time).total_seconds()

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

    def _init_stderr_log(self) -> None:
        """Open the stderr log file under ~/.cuda_exec/<run_tag>/."""
        run_tag = self.storage_config.resolved_run_tag
        log_dir = Path.home() / ".cuda_exec" / run_tag
        log_dir.mkdir(parents=True, exist_ok=True)
        self._stderr_log_path = log_dir / f"stderr_{self.agent_config.name}.log"
        self._stderr_file = open(self._stderr_log_path, "a")

    def _on_stderr(self, line: str) -> None:
        """Callback for Solver CLI stderr — write to log file and update heartbeat."""
        if not hasattr(self, "_stderr_file") or self._stderr_file is None:
            self._init_stderr_log()
        ts = datetime.now().isoformat(timespec="seconds")
        self._stderr_file.write(f"[{ts}] {line}\n")
        self._stderr_file.flush()
        # stderr activity = alive, update heartbeat
        self._last_stream_event_time = datetime.now()
        if self._storage:
            self._storage.write_heartbeat(self._last_stream_event_time)

    def _build_hooks(self) -> dict:
        """Bridge SDK hook callbacks to our EventHandler."""
        handler = self.handler
        log = self.log

        async def on_pre_tool_use(input_data, tool_use_id, context):
            tool_name = input_data.get("tool_name", "")
            tool_input = input_data.get("tool_input", {})

            event = ToolCallEvent(
                tool_name=tool_name,
                tool_input=tool_input,
                tool_use_id=tool_use_id or "",
            )
            if log:
                log.append(event)
            await handler.on_tool_call(event)

            # Check for pending inject from monitor
            result = self._check_tool_rules(tool_name, tool_input)
            if self._monitor and self._monitor._pending_inject:
                guidance = self._monitor.consume_pending_inject()
                if guidance:
                    inject_ctx = f"[Supervisor guidance]: {guidance}"
                    if result and "hookSpecificOutput" in result:
                        existing = result["hookSpecificOutput"].get("additionalContext", "")
                        result["hookSpecificOutput"]["additionalContext"] = f"{existing}\n{inject_ctx}"
                    else:
                        result = {
                            "hookSpecificOutput": {
                                "hookEventName": "PreToolUse",
                                "additionalContext": inject_ctx,
                            }
                        }
            return result

        async def on_post_tool_use(input_data, tool_use_id, context):
            tool_name = input_data.get("tool_name", "")
            tool_response = input_data.get("tool_response", "")

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
                "The Supervisor controls which GPU to use — do not specify gpu.",
                {
                    "kernel": str,
                    "reason": str,
                    "arch": str,       # optional: GPU arch (default: auto-detect)
                    "impls": str,      # optional: impl slugs comma-separated, or "all"
                    "timeout": int,    # optional: per-config timeout in seconds
                },
            )
            async def request_formal_bench(args):
                kernel = args.get("kernel", "")
                reason = args.get("reason", "")
                arch = args.get("arch", "")
                impls = args.get("impls", "")
                timeout = args.get("timeout", 0)

                # Build query string with optional params
                query_parts = [f"kernel={kernel}"]
                if arch:
                    query_parts.append(f"arch={arch}")
                if impls:
                    query_parts.append(f"impls={impls}")
                if timeout:
                    query_parts.append(f"timeout={timeout}")

                event = AskEvent(
                    question=f"REQUEST_FORMAL_BENCH: {' '.join(query_parts)}",
                    context=f"Reason: {reason}",
                )
                if runner_ref.log:
                    runner_ref.log.append(event)

                answer = await runner_ref.handler.on_ask(event)
                return {"content": [{"type": "text", "text": answer}]}

            tools_list.append(request_formal_bench)

        if "save_gem_md" in custom:
            @tool(
                "save_gem_md",
                "Save implementation notes alongside a gem. "
                "Call this after request_formal_bench returns '★ NEW GEM PRODUCED'. "
                "Use the gem_id from the gem notification (e.g. 'gen-cuda/v003'). "
                "Write in Markdown. Include ONLY implementation details — "
                "what you changed, what you generated, core technical points. "
                "Do NOT include reflections or learnings.",
                {
                    "gem_id": str,   # e.g. "gen-cuda/v003" from gem notification
                    "notes": str,    # Markdown: what changed, what generated, core points
                },
            )
            async def save_gem_md(args):
                gem_id = args.get("gem_id", "")
                notes = args.get("notes", "")

                event = AskEvent(
                    question=f"SAVE_GEM_NOTES: gem_id={gem_id}",
                    context=notes,
                )
                if runner_ref.log:
                    runner_ref.log.append(event)

                answer = await runner_ref.handler.on_ask(event)
                return {"content": [{"type": "text", "text": answer}]}

            tools_list.append(save_gem_md)

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

        Returns True if path matches a blocked_path and does NOT match
        any allowed_path exception.
        """
        import os
        path_dir = path.rstrip("/") + "/"
        for blocked in rule.blocked_paths:
            blocked_resolved = self._resolve_path_pattern(blocked).rstrip("/") + "/"
            if path_dir.startswith(blocked_resolved) or path == blocked_resolved.rstrip("/"):
                # Check allowed_paths exceptions
                for allowed in rule.allowed_paths:
                    allowed_resolved = self._resolve_path_pattern(allowed).rstrip("/") + "/"
                    if path_dir.startswith(allowed_resolved) or path == allowed_resolved.rstrip("/"):
                        return False
                return True
        return False

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
    # Recursive ls — blocked (would expose subdirectory contents)
    _RECURSIVE_LS = re.compile(r'\bls\b.*\s+-\S*R')
    # Commands that read content or compile/execute — check blocked paths
    _DANGEROUS_COMMANDS = re.compile(
        r'\b(cat|head|tail|less|more|strings|xxd|od|hexdump|tac|nl|'
        r'wc|grep|awk|sed|sort|uniq|cut|tr|diff|comm|paste|'
        r'nvcc|g\+\+|gcc|clang|cp|mv|ln|source|\.)\b'
    )
    # Commands that are always forbidden regardless of path
    _FORBIDDEN_COMMANDS = re.compile(
        r'\b(git|gh)\b'
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
        for rule in self.agent_config.tool_rules:
            if rule.tool == tool_name:
                if not rule.allow:
                    return self._deny(f"Tool '{tool_name}' is not allowed for this agent role.")
                # ── Bash: check forbidden commands first (git, etc.) ──
                if tool_name == "Bash":
                    command = tool_input.get("command", "")
                    if command and self._FORBIDDEN_COMMANDS.search(command):
                        return self._deny(f"Forbidden command detected in: {command[:100]}")
                    if command and self._RECURSIVE_LS.search(command):
                        return self._deny("Recursive ls (-R) is not allowed. Use ls <dir> instead.")

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

    async def _execute(self, prompt: str, options: ClaudeAgentOptions) -> RunResult:
        """Core execution loop: start client, stream messages, run monitor."""
        self._is_running = True
        self._last_stream_event_time: datetime | None = None
        result = RunResult(log=self.log or SessionLog())
        total_usage: dict = {}

        try:
            try:
                client_ctx = ClaudeSDKClient(options=options)
                client = await client_ctx.__aenter__()
            except Exception as e:
                raise RuntimeError(f"Failed to start Claude CLI: {e}") from e

            try:
                self._client = client
                await client.query(prompt)

                # Start monitor (with runner reference for interrupt/inject)
                monitor = AgentMonitor(result.log, self.handler, runner=self, config=self.monitor_config)
                self._monitor = monitor
                await monitor.start()

                try:
                    async for message in client.receive_response():
                        # Init message — capture session_id
                        if isinstance(message, SystemMessage):
                            if message.subtype == "init":
                                sid = message.data.get("session_id", "")
                                self._session_id = sid
                                result.session_id = sid
                                start_event = StartEvent(session_id=sid)
                                result.log.append(start_event)

                        # Assistant message — complete, fully assembled
                        elif isinstance(message, AssistantMessage):
                            for block in message.content:
                                if isinstance(block, TextBlock):
                                    event = TextOutputEvent(text=block.text)
                                    result.log.append(event)
                                    await self.handler.on_text(event)
                                # ThinkingBlock: heartbeat only (updated below), not logged
                            # Accumulate usage
                            if message.usage:
                                for k, v in message.usage.items():
                                    if isinstance(v, (int, float)):
                                        total_usage[k] = total_usage.get(k, 0) + v

                        # Result message — session complete
                        elif isinstance(message, ResultMessage):
                            result.result_text = message.result or ""
                            result.stop_reason = getattr(message, "stop_reason", "end_turn")
                            stop_event = StopEvent(
                                reason=result.stop_reason,
                                result_text=result.result_text[:5000],
                            )
                            result.log.append(stop_event)
                            await self.handler.on_stop(stop_event)

                        # Rate limit — log warning and wait
                        elif isinstance(message, RateLimitEvent):
                            info = message.rate_limit_info
                            status = getattr(info, "status", None)
                            resets_at = getattr(info, "resets_at", None)
                            utilization = getattr(info, "utilization", None)
                            event = TextOutputEvent(
                                text=f"[rate_limit] status={status} utilization={utilization} resets_at={resets_at}"
                            )
                            result.log.append(event)
                            if status and hasattr(status, "value") and "rejected" in str(status.value):
                                # Rate limited — wait for reset
                                if resets_at:
                                    import time
                                    wait = max(0, resets_at - time.time())
                                    if wait > 0 and wait < 300:
                                        await asyncio.sleep(wait + 1)

                        # Subagent task events
                        elif isinstance(message, TaskStartedMessage):
                            event = SubagentEvent(
                                agent_id=getattr(message, "task_id", ""),
                                agent_type="subagent",
                                action="start",
                            )
                            result.log.append(event)

                        elif isinstance(message, TaskProgressMessage):
                            tool_name = getattr(message, "last_tool_name", "")
                            if tool_name:
                                event = TextOutputEvent(
                                    text=f"[subagent_progress] tool={tool_name}"
                                )
                                result.log.append(event)

                        elif isinstance(message, TaskNotificationMessage):
                            event = SubagentEvent(
                                agent_id=getattr(message, "task_id", ""),
                                agent_type="subagent",
                                action="stop",
                            )
                            result.log.append(event)

                        # StreamEvent — heartbeat (don't log to journal, just update timestamp)
                        elif isinstance(message, StreamEvent):
                            pass  # handled below

                        # Update heartbeat on ANY message
                        self._last_stream_event_time = datetime.now()
                        if self._storage:
                            self._storage.write_heartbeat(self._last_stream_event_time)

                finally:
                    await monitor.stop()
                    self._monitor = None

            finally:
                # Close client — catch CancelledError from transport cleanup
                try:
                    await client_ctx.__aexit__(None, None, None)
                except (asyncio.CancelledError, Exception):
                    pass  # Client cleanup failed — already handled

        finally:
            self._client = None
            self._is_running = False
            if hasattr(self, "_stderr_file") and self._stderr_file:
                self._stderr_file.close()
                self._stderr_file = None

        result.usage = total_usage

        # Write session meta
        if self._storage:
            self._storage.write_meta({
                "session_id": result.session_id,
                "agent": self.agent_config.name,
                "task": prompt[:1000],
                "system_prompt": self.agent_config.system_prompt[:1000],
                "model": self.agent_config.model,
                "permission_mode": self.agent_config.permission_mode,
                "tools": self.agent_config.all_tools,
                "stop_reason": result.stop_reason,
                "usage": total_usage,
            })

        return result
