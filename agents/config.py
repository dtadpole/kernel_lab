"""Configuration system for the agent framework.

Loads agent profiles, tool permissions, monitor settings from YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ToolRule:
    """Fine-grained permission rule for a tool."""
    tool: str
    allow: bool = True
    constraint: str = ""
    blocked_paths: list[str] = field(default_factory=list)  # path prefixes to block
    allowed_paths: list[str] = field(default_factory=list)  # exceptions to blocked_paths


@dataclass
class MonitorConfig:
    """Watchdog thresholds."""
    idle_timeout: float = 900.0        # 15 min no tool calls
    total_timeout: float = 14400.0     # 4 hours default
    hard_limit: float = 21600.0        # 6 hours absolute max, no extension
    check_interval: float = 60.0       # strategy check every 1 min (inside main poll loop)
    loop_threshold: int = 5            # same tool N times in a row
    progress_check_interval: float = 900.0  # 15 min periodic progress check
    heartbeat_timeout: float = 300.0   # 5 min no SDK message in LLM phase = dead
    tool_timeout: float = 1200.0       # 20 min tool/rate_limit phase = hung

    @classmethod
    def for_solver(cls) -> MonitorConfig:
        """Solver: long-running optimization, 30 min idle, 2h soft, 12h hard."""
        return cls(
            idle_timeout=1800, total_timeout=7200, hard_limit=43200,
            check_interval=60, loop_threshold=5, progress_check_interval=300,
            heartbeat_timeout=1200.0,  # 20 min — Solver may have long thinking phases
            tool_timeout=1200.0,      # 20 min — same as default
        )

    @classmethod
    def for_benchmarker(cls) -> MonitorConfig:
        """Benchmarker: single command, 5 min timeout."""
        return cls(
            idle_timeout=120, total_timeout=300, hard_limit=600,
            check_interval=30, loop_threshold=3,
        )

    @classmethod
    def for_steward(cls) -> MonitorConfig:
        """Steward: quick response, 10 min timeout."""
        return cls(
            idle_timeout=120, total_timeout=600, hard_limit=900,
            check_interval=30, loop_threshold=3,
        )

    @classmethod
    def for_rigger(cls) -> MonitorConfig:
        """Rigger: harness engineering, 1 hour total."""
        return cls(
            idle_timeout=300, total_timeout=3600, hard_limit=7200,
            check_interval=60, loop_threshold=5,
        )


@dataclass
class PulseTrigger:
    """One pulse trigger configuration."""
    match: str = ""        # Bash command pattern (empty = file_write, uses watched_dirs)
    cooldown: int = 60     # seconds between triggers of this type


@dataclass
class DirectionConfig:
    """Direction gate + pulse configuration."""
    gate_tools: list[str] = field(default_factory=list)           # e.g. [Write, Edit, Bash]
    gate_watched_dirs: list[str] = field(default_factory=list)
    pulse_file_write_tools: list[str] = field(default_factory=list)  # e.g. [Write, Edit]
    pulse_command_match_tool: str = "Bash"                           # tool for command pattern matching
    pulse_watched_dirs: list[str] = field(default_factory=list)
    pulse_triggers: dict[str, PulseTrigger] = field(default_factory=dict)

    def gate_dirs_resolved(self) -> list[str]:
        import os
        return [os.path.expanduser(d) for d in self.gate_watched_dirs]

    def pulse_dirs_resolved(self) -> list[str]:
        import os
        return [os.path.expanduser(d) for d in self.pulse_watched_dirs]


@dataclass
class StewardConfig:
    """Steward (guidance agent) settings."""
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 2000
    builtin_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    tool_rules: list[ToolRule] = field(default_factory=list)


@dataclass
class StorageConfig:
    """Runtime storage paths."""
    kb_root: str = "~/kernel_lab_kb"
    run_tag: str = ""            # auto-detected from host if empty
    log_dir: str = "~/.kernel_lab/logs"
    max_sessions: int = 200

    @property
    def journal_path(self) -> Path:
        """Journal lives under runs/<run_tag>/journal/."""
        tag = self.run_tag or self._auto_run_tag()
        return Path(self.kb_root).expanduser() / "runs" / tag / "journal"

    @property
    def resolved_run_tag(self) -> str:
        """Get the run_tag, auto-detecting from host if not set."""
        return self.run_tag or self._auto_run_tag()

    @staticmethod
    def _auto_run_tag() -> str:
        """Auto-detect run_tag from cuda_exec, matching run_<host_slug>."""
        try:
            from cuda_exec.impls import _resolve_run_tag
            return _resolve_run_tag()
        except ImportError:
            import socket
            return f"run_{socket.gethostname().split('.')[0]}"

    @property
    def log_path(self) -> Path:
        return Path(self.log_dir).expanduser()


@dataclass
class AgentConfig:
    """Configuration for a single agent role."""
    name: str = ""
    description: str = ""
    model: str = "claude-sonnet-4-6"
    permission_mode: str = "acceptEdits"
    max_turns: int = 50
    max_budget_usd: float = 5.0
    builtin_tools: list[str] = field(default_factory=list)
    custom_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    tool_rules: list[ToolRule] = field(default_factory=list)
    system_prompt: str = ""
    system_prompt_file: str = ""

    @property
    def all_tools(self) -> list[str]:
        return self.builtin_tools + self.custom_tools


@dataclass
class SystemConfig:
    """Top-level system configuration."""
    defaults: dict = field(default_factory=dict)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    steward: StewardConfig = field(default_factory=StewardConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    direction: DirectionConfig = field(default_factory=DirectionConfig)
    agents: dict[str, AgentConfig] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SystemConfig:
        """Load config from YAML, merge defaults, load prompt files."""
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        defaults = raw.get("defaults", {})
        config_dir = path.parent

        # Monitor
        mon_raw = raw.get("monitor", {})
        monitor = MonitorConfig(**{k: v for k, v in mon_raw.items()
                                   if k in MonitorConfig.__dataclass_fields__})

        # Steward
        ra_raw = raw.get("steward", {})
        steward_tools = ra_raw.get("tools", {})
        steward_rules_raw = ra_raw.get("tool_rules", [])
        steward_rules = []
        for r in steward_rules_raw:
            steward_rules.append(ToolRule(
                tool=r.get("tool", ""),
                allow=r.get("allow", True),
                constraint=r.get("constraint", ""),
                blocked_paths=r.get("blocked_paths", []),
                allowed_paths=r.get("allowed_paths", []),
            ))
        steward = StewardConfig(
            model=ra_raw.get("model", "claude-sonnet-4-6"),
            max_tokens=ra_raw.get("max_tokens", 2000),
            builtin_tools=steward_tools.get("builtin", []),
            disallowed_tools=steward_tools.get("disallowed", []),
            tool_rules=steward_rules,
        )

        # Storage
        st_raw = raw.get("storage", {})
        storage = StorageConfig(**{k: v for k, v in st_raw.items()
                                   if k in StorageConfig.__dataclass_fields__})

        # Direction
        dir_raw = raw.get("direction", {})
        gate_raw = dir_raw.get("gate", {})
        pulse_raw = dir_raw.get("pulse", {})
        pulse_triggers = {}
        for name, trigger_raw in pulse_raw.get("triggers", {}).items():
            pulse_triggers[name] = PulseTrigger(
                match=trigger_raw.get("match", ""),
                cooldown=trigger_raw.get("cooldown", 60),
            )
        direction = DirectionConfig(
            gate_tools=gate_raw.get("tools", []),
            gate_watched_dirs=gate_raw.get("watched_dirs", []),
            pulse_file_write_tools=pulse_raw.get("file_write_tools", []),
            pulse_command_match_tool=pulse_raw.get("command_match_tool", "Bash"),
            pulse_watched_dirs=pulse_raw.get("watched_dirs", []),
            pulse_triggers=pulse_triggers,
        )

        # Agents
        agents = {}
        for name, agent_raw in raw.get("agents", {}).items():
            # Merge defaults
            merged = {**defaults, **agent_raw}

            # Extract tools
            tools_raw = merged.pop("tools", {})
            builtin = tools_raw.get("builtin", [])
            custom = tools_raw.get("custom", [])

            # Extract tool rules
            rules_raw = merged.pop("tool_rules", [])
            rules = [ToolRule(**r) for r in rules_raw]

            # Load system prompt from file if specified
            prompt_file = merged.pop("system_prompt_file", "")
            system_prompt = merged.pop("system_prompt", "")
            if prompt_file and not system_prompt:
                prompt_path = config_dir / prompt_file
                if prompt_path.exists():
                    system_prompt = prompt_path.read_text()

            # Filter to known fields
            known = AgentConfig.__dataclass_fields__.keys()
            filtered = {k: v for k, v in merged.items() if k in known}

            agents[name] = AgentConfig(
                name=name,
                builtin_tools=builtin,
                custom_tools=custom,
                tool_rules=rules,
                system_prompt=system_prompt,
                system_prompt_file=prompt_file,
                **filtered,
            )

        return cls(
            defaults=defaults,
            monitor=monitor,
            steward=steward,
            storage=storage,
            direction=direction,
            agents=agents,
        )

    def get_agent(self, name: str) -> AgentConfig:
        """Get agent config by name, reload prompt file from disk.

        Prompt files are re-read on every call so that edits to
        solver.md / steward.md take effect without restarting.
        """
        if name not in self.agents:
            raise KeyError(f"Agent '{name}' not found. Available: {list(self.agents.keys())}")
        agent = self.agents[name]
        # Reload system prompt from file if one was specified
        if agent.system_prompt_file:
            prompt_path = Path(__file__).resolve().parent.parent / "conf" / "agent" / agent.system_prompt_file
            if prompt_path.exists():
                agent.system_prompt = prompt_path.read_text()
        return agent
