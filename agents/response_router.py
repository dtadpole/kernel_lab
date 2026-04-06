"""ResponseRouter: scenario-specific Steward dispatch.

Each hook scenario (ask_question, permission, stuck, session_end, time_limit)
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

CONTEXT_TEMPLATES: dict[str, str] = {
    "ask_question": """## Current Task
{task_description}

## Session Summary
{session_summary}

## Solver's Question
{question}

## Solver's Context
{solver_context}""",

    "permission": """## Requested Operation
Tool: {tool_name}
Input: {tool_input}

## Current Task
{task_description}

## Recent Operations
{recent_tool_calls}""",

    "stuck": """## Alert
{alert_type}: {alert_details}

## Current Task
{task_description}

## Recent 10 Events
{recent_events}

## Tool Call Statistics
{tool_call_counts}

## Elapsed Time
{elapsed_time}""",

    "session_end": """## Original Task
{task_description}

## Solver's Final Output
{result_text}

## Stop Reason
{stop_reason}

## Session Statistics
- Elapsed: {elapsed_time}
- Tool calls: {total_tool_calls}
- Errors: {error_count}

## Operation Summary
{session_summary}""",

    "time_limit": """## Elapsed Time
{elapsed_time} (limit: {time_limit})

## Current Task
{task_description}

## Recent Progress
{recent_progress}

## Tool Call Trend
{tool_call_trend}""",
}


@dataclass
class ResponseScenario:
    """Complete configuration for one response scenario."""
    name: str
    system_prompt: str
    context_template: str
    tools: list[str] = field(default_factory=list)
    max_turns: int = 1
    model: str = "claude-sonnet-4-6"


@dataclass
class ResponseVerdict:
    """Parsed response from the Steward."""
    action: str        # ACCEPT, REJECT, RETRY, ALLOW, DENY, CONTINUE, INJECT, INTERRUPT, EXTEND, WRAP_UP, KILL
    detail: str        # text after the colon (if any)
    reasoning: str     # full text after the first line

    @classmethod
    def parse(cls, raw: str) -> ResponseVerdict:
        """Parse a structured response (first line = action, rest = reasoning)."""
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
    ):
        self.model = model
        self.storage_config = storage_config or StorageConfig()
        self.scenarios: dict[str, ResponseScenario] = {}
        self._load_prompts(Path(prompts_dir))

    def _load_prompts(self, prompts_dir: Path) -> None:
        """Load all .md files from prompts_dir as scenarios."""
        if not prompts_dir.exists():
            return

        for md_file in sorted(prompts_dir.glob("*.md")):
            name = md_file.stem  # e.g., "ask_question"
            system_prompt = md_file.read_text().strip()
            context_template = CONTEXT_TEMPLATES.get(name, "{context}")

            self.scenarios[name] = ResponseScenario(
                name=name,
                system_prompt=system_prompt,
                context_template=context_template,
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
            scenario: Scenario name (ask_question, permission, stuck, session_end, time_limit).
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

        agent_config = AgentConfig(
            name=f"steward_{config.name}",
            model=config.model,
            permission_mode="default",
            max_turns=config.max_turns,
            builtin_tools=config.tools,
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
