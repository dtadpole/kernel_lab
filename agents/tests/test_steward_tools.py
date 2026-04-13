"""Tests for steward tool restrictions and verdict parsing."""
import pytest
from agents.config import SystemConfig, StewardConfig, ToolRule
from agents.response_router import ResponseVerdict

pytestmark = pytest.mark.quick


class TestStewardConfig:
    """Steward tool config loading from agents.yaml."""

    def test_loads_disallowed_tools(self):
        cfg = SystemConfig.from_yaml("conf/agent/agents.yaml")
        assert "SendMessage" in cfg.steward.disallowed_tools
        assert "ToolSearch" in cfg.steward.disallowed_tools
        assert "AskUserQuestion" in cfg.steward.disallowed_tools
        assert "EnterPlanMode" in cfg.steward.disallowed_tools
        assert "NotebookEdit" in cfg.steward.disallowed_tools

    def test_loads_builtin_tools(self):
        cfg = SystemConfig.from_yaml("conf/agent/agents.yaml")
        assert "Read" in cfg.steward.builtin_tools
        assert "Grep" in cfg.steward.builtin_tools
        assert "Glob" in cfg.steward.builtin_tools
        assert "Bash" in cfg.steward.builtin_tools
        # Should NOT have write tools
        assert "Write" not in cfg.steward.builtin_tools
        assert "Edit" not in cfg.steward.builtin_tools
        assert "SendMessage" not in cfg.steward.builtin_tools

    def test_loads_tool_rules(self):
        cfg = SystemConfig.from_yaml("conf/agent/agents.yaml")
        rules = {r.tool: r for r in cfg.steward.tool_rules}
        assert rules["Bash"].allow is True
        assert rules["Write"].allow is False
        assert rules["Edit"].allow is False

    def test_empty_config_has_defaults(self):
        sc = StewardConfig()
        assert sc.model == "claude-sonnet-4-6"
        assert sc.builtin_tools == []
        assert sc.disallowed_tools == []
        assert sc.tool_rules == []


class TestStewardConfigNegative:
    """Negative cases — verify tools are actually blocked."""

    def test_sendmessage_not_in_builtin(self):
        cfg = SystemConfig.from_yaml("conf/agent/agents.yaml")
        assert "SendMessage" not in cfg.steward.builtin_tools

    def test_write_not_in_builtin(self):
        cfg = SystemConfig.from_yaml("conf/agent/agents.yaml")
        assert "Write" not in cfg.steward.builtin_tools

    def test_edit_not_in_builtin(self):
        cfg = SystemConfig.from_yaml("conf/agent/agents.yaml")
        assert "Edit" not in cfg.steward.builtin_tools

    def test_write_denied_by_tool_rules(self):
        cfg = SystemConfig.from_yaml("conf/agent/agents.yaml")
        rules = {r.tool: r for r in cfg.steward.tool_rules}
        assert "Write" in rules
        assert rules["Write"].allow is False

    def test_edit_denied_by_tool_rules(self):
        cfg = SystemConfig.from_yaml("conf/agent/agents.yaml")
        rules = {r.tool: r for r in cfg.steward.tool_rules}
        assert "Edit" in rules
        assert rules["Edit"].allow is False

    def test_disallowed_tools_merged_in_agent_config(self):
        """AgentConfig.disallowed_tools should be passed through."""
        from agents.config import AgentConfig
        ac = AgentConfig(disallowed_tools=["SendMessage", "ToolSearch"])
        assert "SendMessage" in ac.disallowed_tools
        assert "ToolSearch" in ac.disallowed_tools

    def test_disallowed_count(self):
        cfg = SystemConfig.from_yaml("conf/agent/agents.yaml")
        assert len(cfg.steward.disallowed_tools) >= 5


class TestResponseVerdictParseNegative:
    """Negative cases for verdict parsing."""

    def test_rejected_not_parsed_as_approved(self):
        v = ResponseVerdict.parse("REDIRECT:evidence is vague")
        assert v.action != "APPROVED"
        assert v.action == "REDIRECT"

    def test_random_text_not_parsed_as_approved(self):
        v = ResponseVerdict.parse("I think this looks good")
        assert v.action != "APPROVED"

    def test_lowercase_not_matched(self):
        """Lowercase 'approved' should still match via upper()."""
        v = ResponseVerdict.parse("approved: looks fine")
        assert v.action == "APPROVED"

    def test_partial_match_does_not_false_positive(self):
        """'APPROVAL' should not match 'APPROVED'."""
        v = ResponseVerdict.parse("APPROVAL granted")
        # APPROVAL doesn't start with any known action exactly
        # (APPROVED is longer, won't match APPROVAL)
        assert v.action != "APPROVED"


class TestResponseVerdictParse:
    """ResponseVerdict.parse() robustness."""

    def test_strict_format(self):
        v = ResponseVerdict.parse("APPROVED:all good")
        assert v.action == "APPROVED"
        assert v.detail == "all good"

    def test_strict_format_no_detail(self):
        v = ResponseVerdict.parse("ON_TRACK")
        assert v.action == "ON_TRACK"
        assert v.detail == ""

    def test_startswith_fallback(self):
        """The bug: steward outputs 'APPROVED and sent to the Solver.'"""
        v = ResponseVerdict.parse("APPROVED and sent to the Solver.")
        assert v.action == "APPROVED"
        assert "sent to the Solver" in v.detail

    def test_startswith_redirect(self):
        v = ResponseVerdict.parse("REDIRECT — evidence is vague")
        assert v.action == "REDIRECT"
        assert "evidence is vague" in v.detail

    def test_startswith_with_reasoning(self):
        v = ResponseVerdict.parse("APPROVED: looks good\n\nDetailed reasoning here.")
        assert v.action == "APPROVED"
        assert v.detail == "looks good"
        assert "Detailed reasoning" in v.reasoning

    def test_empty_string(self):
        v = ResponseVerdict.parse("")
        assert v.action == ""
        assert "Empty response" in v.reasoning

    def test_none_input(self):
        v = ResponseVerdict.parse(None)
        assert v.action == ""

    def test_continue(self):
        v = ResponseVerdict.parse("CONTINUE\nKeep going.")
        assert v.action == "CONTINUE"
        assert v.reasoning == "Keep going."

    def test_unknown_action_fallback(self):
        v = ResponseVerdict.parse("SOMETHING_WEIRD")
        assert v.action == "SOMETHING_WEIRD"

    def test_on_track_with_explanation(self):
        v = ResponseVerdict.parse("ON_TRACK — solver is doing fine")
        assert v.action == "ON_TRACK"
        assert "solver is doing fine" in v.detail
