"""ResponseRouter: scenario-specific Steward dispatch.

Each hook scenario (ask_question, permission, session_end, progress_check, etc.)
has its own system prompt and context-building template. The router dispatches
through AgentRunner for uniform journal logging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from agents.config import AgentConfig, MonitorConfig, StorageConfig


# ── Context templates per scenario ──
# Each template defines what variables are injected into the user prompt.

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

# Common context block prepended to every scenario
_CONTEXT_HEADER = """## Solver Mode: {mode}

## Current Direction
{direction_json}
Direction file: {direction_path}

## Recent Events
{recent_events}

## For Details
- Transcript: {transcript_path}
- Events: {events_path}"""

CONTEXT_TEMPLATES: dict[str, str] = {
    "ask_question": _CONTEXT_HEADER + """

## Question
{question}""",

    "permission": _CONTEXT_HEADER + """

## Permission Request
Tool: {tool_name}
Input: {tool_input}""",

    "session_end": _CONTEXT_HEADER + """

## Session Ended
Stop reason: {stop_reason}
Elapsed: {elapsed_time}
Tool calls: {total_tool_calls}
Errors: {error_count}

## Solver's Final Output
{result_text}""",

    "progress_check": _CONTEXT_HEADER + """

## Progress Check
Elapsed: {elapsed_time}""",

    "set_direction": """## Direction Proposal
{direction_json}

""" + _CONTEXT_HEADER,

    "direction_pulse": """## Current Direction
{direction_json}

## Trigger
Solver just completed: {trigger_type}

""" + _CONTEXT_HEADER,

    "start_exploring": """## Current Direction
{direction_json}

## Solver's Reason for Wanting to Change
{reason}

""" + _CONTEXT_HEADER,
}


@dataclass
class ResponseScenario:
    """Complete configuration for one response scenario."""
    name: str
    system_prompt: str
    context_template: str
    tools: list[str] = field(default_factory=list)
    max_turns: int = 10  # enough for reading trajectory + tool calls + verdict
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

    Loads system prompts from conf/agent/response_prompts/*.md.
    Combines with context templates to build the full prompt.
    Dispatches through AgentRunner for journal logging.
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
        self._load_prompts(Path(prompts_dir))

    @staticmethod
    def _load_base_prompt(path: Path) -> str:
        """Load the shared Steward base system prompt."""
        if path.exists():
            return path.read_text().strip()
        return ""

    def _load_prompts(self, prompts_dir: Path) -> None:
        """Load all .md files from prompts_dir as scenarios.

        Each scenario's system prompt = base Steward prompt + scenario-specific prompt.
        """
        if not prompts_dir.exists():
            return

        for md_file in sorted(prompts_dir.glob("*.md")):
            name = md_file.stem  # e.g., "ask_question"
            scenario_prompt = md_file.read_text().strip()
            context_template = CONTEXT_TEMPLATES.get(name, "{context}")

            # Combine: base identity + scenario-specific instructions
            if self._base_prompt:
                system_prompt = f"{self._base_prompt}\n\n---\n\n## Current Scenario: {name}\n\n{scenario_prompt}"
            else:
                system_prompt = scenario_prompt

            self.scenarios[name] = ResponseScenario(
                name=name,
                system_prompt=system_prompt,
                context_template=context_template,
                max_turns=SCENARIO_MAX_TURNS.get(name, 10),
                model=self.model,
            )

    def has_scenario(self, name: str) -> bool:
        return name in self.scenarios

    def build_context(self, scenario: str, variables: dict[str, str]) -> str:
        """Build the user prompt from template + variables.

        Missing variables are replaced with '(not available)'.
        """
        if scenario not in self.scenarios:
            raise KeyError(f"Unknown scenario: {scenario}. Available: {list(self.scenarios.keys())}")

        template = self.scenarios[scenario].context_template

        # Safe format: replace {key} with value, leave unknown keys
        def replacer(match):
            key = match.group(1)
            return variables.get(key, "(not available)")

        return re.sub(r"\{(\w+)\}", replacer, template)

    async def respond(
        self,
        scenario: str,
        variables: dict[str, str],
    ) -> ResponseVerdict:
        """Call the Steward and return a parsed verdict.

        Args:
            scenario: Scenario name (ask_question, permission, session_end, progress_check, etc.).
            variables: Template variables for context building.

        Returns:
            Parsed ResponseVerdict with action, detail, and reasoning.
        """
        config = self.scenarios[scenario]
        user_prompt = self.build_context(scenario, variables)

        raw_response = await self._call_agent(config, user_prompt)
        return ResponseVerdict.parse(raw_response)

    async def respond_raw(
        self,
        scenario: str,
        variables: dict[str, str],
    ) -> str:
        """Call the Steward and return raw text (for ask_question scenario)."""
        config = self.scenarios[scenario]
        user_prompt = self.build_context(scenario, variables)
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
        agent_config = AgentConfig(
            name=f"steward_{config.name}",
            model=config.model,
            permission_mode="acceptEdits",
            max_turns=config.max_turns,
            max_budget_usd=5.0,
            builtin_tools=steward_tools,
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
