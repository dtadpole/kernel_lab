# Supervisor — 编排与调度设计

## Overview

Supervisor 是整个系统的编排中枢。它是一个 Python asyncio 进程，通过 `AgentRunner` 管理 Solver 的生命周期，通过实现 `EventHandler` protocol 被动接收所有事件，通过 `Steward` 在关键节点做决策。

Supervisor 自身不主动轮询——它是**事件驱动**的，由 AgentRunner 的 hooks、SDK message stream、和 AgentMonitor 来驱动。

## 进程模型

```
┌──────────────────────────────────────────────────────────┐
│  Python 进程                                              │
│                                                          │
│  Supervisor (implements EventHandler)                    │
│    │                                                     │
│    ├── AgentRunner                                       │
│    │     ├── ClaudeSDKClient ──→ claude CLI 子进程 (Solver) │
│    │     ├── SDK hooks ──→ Supervisor.on_*()             │
│    │     └── MCP tools ──→ Supervisor.on_ask()           │
│    │                                                     │
│    ├── AgentMonitor (parallel asyncio task)              │
│    │     └── _check_health() ──→ Supervisor.on_monitor_alert() │
│    │                                                     │
│    └── Steward (via ResponseRouter)                      │
│          └── 被 Supervisor.on_*() 调用做决策              │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## 事件调度：谁调用了 Supervisor 的什么方法

Supervisor 实现 `EventHandler` protocol，以下是完整的调用链：

### 来自 AgentRunner 的 SDK Hooks

```
Claude CLI 子进程 (Solver)
  │
  ├── Solver 要调工具
  │     → SDK PreToolUse hook
  │       → AgentRunner.on_pre_tool_use()       [runner.py _build_hooks()]
  │         → Supervisor.on_tool_call()
  │
  ├── 工具执行完毕
  │     → SDK PostToolUse hook
  │       → AgentRunner.on_post_tool_use()       [runner.py _build_hooks()]
  │         → Supervisor.on_tool_result()
  │
  ├── 工具执行失败
  │     → SDK PostToolUseFailure hook
  │       → AgentRunner.on_post_tool_failure()   [runner.py _build_hooks()]
  │         → Supervisor.on_tool_result(is_error=True)
  │
  └── Solver 需要权限
        → SDK PermissionRequest hook
          → AgentRunner.on_permission_request()  [runner.py _build_hooks()]
            → Supervisor.on_permission()
              → Steward.check_permission()
```

### 来自 AgentRunner 的 MCP Tool

```
Solver 调用 ask_supervisor MCP tool
  → AgentRunner 的 ask_supervisor handler      [runner.py _build_mcp_tools()]
    → Supervisor.on_ask()
      → Steward.answer_question()
    ← 答案文本返回给 Solver（通过 MCP tool response）

Solver 调用 request_formal_bench MCP tool
  → AgentRunner 的 request_formal_bench handler [runner.py _build_mcp_tools()]
    → Supervisor.on_ask()  (question 以 REQUEST_FORMAL_BENCH 开头)
    ← 答案文本返回给 Solver
```

### 来自 AgentRunner 的 Message Stream

```
AgentRunner._execute() 中的 async for message in client.receive_response()
  │
  ├── SystemMessage (subtype="init")
  │     → 捕获 session_id
  │     → 记录 StartEvent
  │
  ├── AssistantMessage (TextBlock)
  │     → 记录 TextOutputEvent
  │     → Supervisor.on_text()                   [runner.py _execute()]
  │
  └── ResultMessage
        → 记录 StopEvent
        → Supervisor.on_stop()                   [runner.py _execute()]
          → Steward.review_session_end()
```

### 来自 AgentMonitor

```
AgentMonitor._run_loop()  (并行 asyncio task，每 check_interval 秒执行一次)
  │
  └── _check_health() 检测到异常
        → Supervisor.on_monitor_alert()          [monitor.py _run_loop()]
          │
          ├── alert_type == "idle_timeout"
          │     → Steward.handle_stuck()
          │
          ├── alert_type == "loop_detected"
          │     → Steward.handle_stuck()
          │
          ├── alert_type == "total_timeout"
          │     → Steward.handle_time_limit()
          │
          └── alert_type == "hard_limit"
                → 无条件返回 "interrupt"（不经过 Steward）
```

## 事件 → Steward → 动作 映射

| Supervisor 方法 | 触发源 | Steward 方法 | 可能的动作 |
|----------------|--------|-------------|-----------|
| `on_tool_call` | PreToolUse hook | — | 记录状态、检查 tool_rules |
| `on_tool_result` | PostToolUse hook | — | 记录状态、累计 error_count |
| `on_text` | Message stream | — | 记录（无动作） |
| `on_ask` | ask_supervisor MCP | `answer_question()` | 返回答案（inline） |
| `on_permission` | PermissionRequest hook | `check_permission()` | ALLOW / DENY |
| `on_stop` | ResultMessage | `review_session_end()` | ACCEPT / REJECT / RETRY |
| `on_monitor_alert` | AgentMonitor | `handle_stuck()` / `handle_time_limit()` | CONTINUE / INJECT / INTERRUPT / EXTEND / WRAP_UP / KILL |

## 决策循环

Supervisor 的 `run_task()` 管理完整的任务生命周期：

```
run_task(task, run_tag)
  │
  for iteration in range(max_iterations):
    │
    ├── 生成 run_tag: supervisor_run_YYYYMMDD_HHMMSS
    │
    ├── phase = "solving"
    │   └── AgentRunner.run(prompt, task_slug)
    │         │
    │         │  期间所有事件通过 hooks → Supervisor.on_*()
    │         │  Steward 在 on_ask / on_permission / on_stop / on_monitor_alert 中被调用
    │         │  Monitor 并行运行
    │         │
    │         └── Solver 停止 → on_stop() → Steward.review_session_end()
    │              → verdict 存入 self._pending_verdict
    │
    ├── phase = "deciding"
    │   └── 读取 _pending_verdict
    │       │
    │       ├── ACCEPT (level 1)
    │       │     → phase = "done", break
    │       │
    │       ├── REJECT (level 3)
    │       │     → 构建新 prompt（含拒绝原因），继续循环
    │       │
    │       ├── RETRY (level 3)
    │       │     → 构建新 prompt（含具体指导），继续循环
    │       │
    │       └── other
    │             → phase = "done", break
    │
  └── return TaskResult
```

## Intervention Level 处理

Supervisor 根据 Steward 返回的 intervention_level 采取不同动作：

```python
async def _handle_steward_response(self, response: StewardResponse):
    match response.intervention_level:
        case 1:  # Inline
            pass  # 记录，无动作（或答案已通过 MCP 返回）

        case 2:  # Inject
            await self._solver_runner.interrupt()
            await self._solver_runner.resume(response.detail)

        case 3:  # Restart
            # 当前 iteration 结束，run_task 循环会启动新 iteration
            pass

        case 4:  # Kill
            await self._solver_runner.interrupt()
            # 不再 resume
```

## run_tag 管理

每次 `run_task()` 生成一个 run_tag，注入到 Solver 的 prompt 中：

```
run_tag = supervisor_run_YYYYMMDD_HHMMSS
```

Solver 使用这个 run_tag 调用所有 `ik:exec` 命令。Scratch 目录在 `~/.cuda_exec/<run_tag>/`。

run_tag 在 `SupervisorState` 中记录，可通过 `get_status()` 查询。
