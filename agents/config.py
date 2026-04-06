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


@dataclass
class MonitorConfig:
    """Watchdog thresholds."""
    idle_timeout: float = 900.0        # 15 min no tool calls
    total_timeout: float = 14400.0     # 4 hours default
    hard_limit: float = 21600.0        # 6 hours absolute max, no extension
    check_interval: float = 60.0       # check every 1 min
    loop_threshold: int = 5            # same tool N times in a row
    progress_check_interval: float = 900.0  # 15 min periodic progress check

    @classmethod
    def for_solver(cls) -> MonitorConfig:
        """Solver: long-running optimization, 30 min idle, 2h soft, 12h hard."""
        return cls(
            idle_timeout=1800, total_timeout=7200, hard_limit=43200,
            check_interval=60, loop_threshold=5, progress_check_interval=300,
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
class StewardConfig:
    """Steward (guidance agent) settings."""
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 2000


@dataclass
class StorageConfig:
    """Runtime storage paths."""
    kb_root: str = "~/kernel_lab_kb"
    journal_dir: str = "agent_journal"
    log_dir: str = "~/.kernel_lab/logs"
    session_id_format: str = "{date}_{time}_{agent}_{hash}"
    max_sessions: int = 200

    @property
    def journal_path(self) -> Path:
        return Path(self.kb_root).expanduser() / self.journal_dir

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

        # Response agent
        ra_raw = raw.get("steward", {})
        steward = StewardConfig(**{k: v for k, v in ra_raw.items()
                                                if k in StewardConfig.__dataclass_fields__})

        # Storage
        st_raw = raw.get("storage", {})
        storage = StorageConfig(**{k: v for k, v in st_raw.items()
                                   if k in StorageConfig.__dataclass_fields__})

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
            agents=agents,
        )

    def get_agent(self, name: str) -> AgentConfig:
        """Get agent config by name, raise if not found."""
        if name not in self.agents:
            raise KeyError(f"Agent '{name}' not found. Available: {list(self.agents.keys())}")
        return self.agents[name]
