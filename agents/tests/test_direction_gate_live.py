"""Live test: call the ACTUAL hook closure from AgentRunner._build_hooks().

This doesn't simulate the gate logic — it creates a real AgentRunner,
extracts the real PreToolUse hook function, and calls it with test inputs.
If the hook doesn't fire or doesn't deny, we know exactly where the break is.
"""

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agents.config import AgentConfig, MonitorConfig, StorageConfig
from agents.events import DefaultHandler, ToolCallEvent
from agents.runner import AgentRunner


# ── Minimal handler that mimics Workshop's direction gate attributes ──

@dataclass
class MockState:
    current_direction: dict | None = None
    mode: str = "exploring"


class MockHandler(DefaultHandler):
    """Handler with the same direction gate attributes as Workshop."""

    def __init__(self, gate_dirs: list[str], gate_tools: list[str], direction: dict | None = None):
        super().__init__()
        self._direction_gate_dirs = gate_dirs
        self._direction_gate_tools = gate_tools
        self._direction_gate_message = "You must set_direction first."
        self.state = MockState(current_direction=direction)


# ── Test runner ──

async def run_tests():
    agent_config = AgentConfig(
        name="test_gate",
        model="claude-sonnet-4-6",
        permission_mode="acceptEdits",
        max_turns=1,
    )

    gate_dirs = ["/home/zhenc/kernel_lab", "/home/zhenc/kernel_lab_kb"]
    gate_tools = ["Write", "Edit", "Bash"]

    results = []

    # ── Test 1: Write without direction → should deny ──
    handler = MockHandler(gate_dirs=gate_dirs, gate_tools=gate_tools, direction=None)
    runner = AgentRunner(
        agent_config=agent_config,
        handler=handler,
        monitor_config=MonitorConfig(),
        storage_config=StorageConfig(kb_root="/tmp/test_gate_live", run_tag="test"),
    )

    hooks = runner._build_hooks()
    pre_tool_hooks = hooks["PreToolUse"]
    hook_fn = pre_tool_hooks[0].hooks[0]  # The actual on_pre_tool_use closure

    result = await hook_fn(
        {"tool_name": "Write", "tool_input": {"file_path": "/home/zhenc/kernel_lab_kb/runs/test/gen/sm90/matmul/cuda/cuda.cu"}},
        "test-id-1",
        {},
    )

    denied = (
        isinstance(result, dict)
        and result.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
    )
    results.append(("Write without direction → deny", denied))

    # ── Test 2: Write with direction → should allow ──
    handler2 = MockHandler(
        gate_dirs=gate_dirs, gate_tools=gate_tools,
        direction={"name": "warp-specialization"},
    )
    runner2 = AgentRunner(
        agent_config=agent_config,
        handler=handler2,
        monitor_config=MonitorConfig(),
        storage_config=StorageConfig(kb_root="/tmp/test_gate_live", run_tag="test"),
    )
    hooks2 = runner2._build_hooks()
    hook_fn2 = hooks2["PreToolUse"][0].hooks[0]

    result2 = await hook_fn2(
        {"tool_name": "Write", "tool_input": {"file_path": "/home/zhenc/kernel_lab_kb/runs/test/gen/sm90/matmul/cuda/cuda.cu"}},
        "test-id-2",
        {},
    )
    allowed = result2 is None or (
        isinstance(result2, dict)
        and result2.get("hookSpecificOutput", {}).get("permissionDecision") != "deny"
    )
    results.append(("Write with direction → allow", allowed))

    # ── Test 3: Write outside watched dirs → should allow ──
    handler3 = MockHandler(gate_dirs=gate_dirs, gate_tools=gate_tools, direction=None)
    runner3 = AgentRunner(
        agent_config=agent_config,
        handler=handler3,
        monitor_config=MonitorConfig(),
        storage_config=StorageConfig(kb_root="/tmp/test_gate_live", run_tag="test"),
    )
    hooks3 = runner3._build_hooks()
    hook_fn3 = hooks3["PreToolUse"][0].hooks[0]

    result3 = await hook_fn3(
        {"tool_name": "Write", "tool_input": {"file_path": "/tmp/scratch.cu"}},
        "test-id-3",
        {},
    )
    allowed3 = result3 is None or (
        isinstance(result3, dict)
        and result3.get("hookSpecificOutput", {}).get("permissionDecision") != "deny"
    )
    results.append(("Write outside watched dir → allow", allowed3))

    # ── Test 4: Read tool (not in gate_tools) → should allow ──
    result4 = await hook_fn(
        {"tool_name": "Read", "tool_input": {"file_path": "/home/zhenc/kernel_lab_kb/runs/test/gen/sm90/matmul/cuda/cuda.cu"}},
        "test-id-4",
        {},
    )
    allowed4 = result4 is None or (
        isinstance(result4, dict)
        and result4.get("hookSpecificOutput", {}).get("permissionDecision") != "deny"
    )
    results.append(("Read (not gated) → allow", allowed4))

    # ── Test 5: Bash redirect without direction → should deny ──
    result5 = await hook_fn(
        {"tool_name": "Bash", "tool_input": {"command": "cat file.cu > /home/zhenc/kernel_lab_kb/runs/test/gen/cuda.cu"}},
        "test-id-5",
        {},
    )
    denied5 = (
        isinstance(result5, dict)
        and result5.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
    )
    results.append(("Bash redirect without direction → deny", denied5))

    # ── Test 6: Empty gate_dirs → should allow everything ──
    handler6 = MockHandler(gate_dirs=[], gate_tools=gate_tools, direction=None)
    runner6 = AgentRunner(
        agent_config=agent_config,
        handler=handler6,
        monitor_config=MonitorConfig(),
        storage_config=StorageConfig(kb_root="/tmp/test_gate_live", run_tag="test"),
    )
    hooks6 = runner6._build_hooks()
    hook_fn6 = hooks6["PreToolUse"][0].hooks[0]

    result6 = await hook_fn6(
        {"tool_name": "Write", "tool_input": {"file_path": "/home/zhenc/kernel_lab_kb/runs/test/gen/sm90/matmul/cuda/cuda.cu"}},
        "test-id-6",
        {},
    )
    allowed6 = result6 is None or (
        isinstance(result6, dict)
        and result6.get("hookSpecificOutput", {}).get("permissionDecision") != "deny"
    )
    results.append(("Empty gate_dirs → allow", allowed6))

    # ── Print results ──
    print()
    all_pass = True
    for name, passed in results:
        icon = "✓" if passed else "✗"
        print(f"  {icon} {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print(f"All {len(results)} tests passed.")
    else:
        print(f"FAILURES detected!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_tests())
