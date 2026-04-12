"""ResponseRouter: scenario-specific Steward dispatch.

Each hook scenario (ask_question, permission, session_end, progress_check, etc.)
has its own Jinja2 user-message template (conf/agent/response_prompts/*.md).
The router renders the template with wave_context + scenario variables, then
dispatches through AgentRunner for uniform journal logging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import jinja2

from agents.config import AgentConfig, MonitorConfig, StorageConfig, ToolRule


# Per-scenario max_turns (how deep each scenario can go with tool calls)
SCENARIO_MAX_TURNS: dict[str, int] = {
    "permission": 5,
    "progress_check": 10,
    "ask_question": 20,
    "session_end": 20,
    "set_direction": 20,
    "direction_pulse": 10,
    "start_exploring": 20,
}


@dataclass
class ResponseScenario:
    """Complete configuration for one response scenario."""
    name: str
    system_prompt: str
    user_template: jinja2.Template
    tools: list[str] = field(default_factory=list)
    max_turns: int = 10
    model: str = "claude-sonnet-4-6"


@dataclass
class ResponseVerdict:
    """Parsed response from the Steward."""
    action: str        # SUCCESS, CONTINUE, ABORT, ALLOW, DENY, INJECT, INTERRUPT, EXTEND, WRAP_UP, KILL
    detail: str        # text after the colon (if any)
    reasoning: str     # full text after the first line

    @classmethod
    def parse(cls, raw: str) -> ResponseVerdict:
        """Parse a structured response (first line = action, rest = reasoning)."""
        if not raw or not raw.strip():
            return cls(action="", detail="", reasoning="Empty response from Steward")

        lines = raw.strip().split("\n", 1)
        first_line = lines[0].strip()
        reasoning = lines[1].strip() if len(lines) > 1 else ""

        # Parse ACTION:detail format
        match = re.match(r"^([A-Z_]+)(?::(.*))?$", first_line)
        if match:
            action = match.group(1)
            detail = (match.group(2) or "").strip()
        else:
            # Fallback: treat the whole first line as the action
            action = first_line.upper().replace(" ", "_")
            detail = ""

        return cls(action=action, detail=detail, reasoning=reasoning)


class ResponseRouter:
    """Routes hook events to scenario-specific Steward calls.

    System prompt: conf/agent/prompts/steward.md (Steward identity, static).
    User message:  conf/agent/response_prompts/<scenario>.md (Jinja2 template).

    Each scenario MD file is a complete user-message template with {{ variables }}
    that get rendered with wave_context (global) + scenario-specific variables.
    """

    def __init__(
        self,
        prompts_dir: str | Path,
        model: str = "claude-sonnet-4-6",
        storage_config: StorageConfig | None = None,
        base_prompt_path: str | Path = "conf/agent/prompts/steward.md",
    ):
        self.model = model
        self.storage_config = storage_config or StorageConfig()
        self.scenarios: dict[str, ResponseScenario] = {}
        self._base_prompt = self._load_base_prompt(Path(base_prompt_path))
        self._jinja_env = jinja2.Environment(
            undefined=jinja2.StrictUndefined,
            keep_trailing_newline=True,
        )
        self._load_prompts(Path(prompts_dir))

    @staticmethod
    def _load_base_prompt(path: Path) -> str:
        """Load the shared Steward base system prompt."""
        if path.exists():
            return path.read_text().strip()
        return ""

    def _load_prompts(self, prompts_dir: Path) -> None:
        """Load all .md files from prompts_dir as Jinja2 user-message templates."""
        if not prompts_dir.exists():
            return

        for md_file in sorted(prompts_dir.glob("*.md")):
            name = md_file.stem  # e.g., "ask_question"
            template_src = md_file.read_text()
            user_template = self._jinja_env.from_string(template_src)

            self.scenarios[name] = ResponseScenario(
                name=name,
                system_prompt=self._base_prompt,
                user_template=user_template,
                max_turns=SCENARIO_MAX_TURNS.get(name, 10),
                model=self.model,
            )

    def has_scenario(self, name: str) -> bool:
        return name in self.scenarios

    def render_user_message(self, scenario: str, variables: dict) -> str:
        """Render the scenario's Jinja2 template with the given variables.

        Args:
            scenario: Scenario name (ask_question, permission, etc.).
            variables: Dict with 'wave' (wave_context) + scenario-specific keys.

        Raises:
            jinja2.UndefinedError: if a required template variable is missing.
            KeyError: if scenario name is unknown.
        """
        if scenario not in self.scenarios:
            raise KeyError(f"Unknown scenario: {scenario}. Available: {list(self.scenarios.keys())}")

        template = self.scenarios[scenario].user_template
        return template.render(**variables)

    async def respond(
        self,
        scenario: str,
        variables: dict,
    ) -> ResponseVerdict:
        """Call the Steward and return a parsed verdict."""
        config = self.scenarios[scenario]
        user_prompt = self.render_user_message(scenario, variables)

        raw_response = await self._call_agent(config, user_prompt)
        return ResponseVerdict.parse(raw_response)

    async def respond_raw(
        self,
        scenario: str,
        variables: dict,
    ) -> str:
        """Call the Steward and return raw text (for ask_question scenario)."""
        config = self.scenarios[scenario]
        user_prompt = self.render_user_message(scenario, variables)
        return await self._call_agent(config, user_prompt)

    async def _call_agent(
        self, config: ResponseScenario, user_prompt: str, retries: int = 2,
    ) -> str:
        """Execute through AgentRunner for journal logging, with retry."""
        import asyncio
        from agents.runner import AgentRunner  # deferred to avoid circular import

        # Steward gets read-only tools + web research + doc_retrieval via Bash
        steward_tools = config.tools if config.tools else [
            "Read", "Grep", "Glob", "Bash", "WebSearch", "WebFetch",
        ]
        # Steward is read-only: no Write/Edit, Bash for research only
        steward_rules = [
            ToolRule(tool="Bash", allow=True,
                     constraint="Read-only commands for research. No file modification, no process management."),
            ToolRule(tool="Write", allow=False),
            ToolRule(tool="Edit", allow=False),
        ]
        agent_config = AgentConfig(
            name=f"steward_{config.name}",
            model=config.model,
            permission_mode="acceptEdits",
            max_turns=config.max_turns,
            max_budget_usd=5.0,
            builtin_tools=steward_tools,
            tool_rules=steward_rules,
            system_prompt=config.system_prompt,
        )

        for attempt in range(retries + 1):
            try:
                runner = AgentRunner(
                    agent_config=agent_config,
                    storage_config=self.storage_config,
                    monitor_config=MonitorConfig.for_steward(),
                )
                result = await runner.run(
                    prompt=user_prompt,
                    task_slug=config.name,
                )
                return result.result_text or ""

            except Exception as e:
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return f"CONTINUE\nSteward error after {retries + 1} attempts: {e}"
