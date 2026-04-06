# Steward — Runtime Guidance Agent Design

## Overview

Steward 负责照看一个 long-running 的 Solver。它持续监控 Solver 的状态，在关键节点提供 guidance、审核完成质量、处理异常情况。

Steward 通过 ResponseRouter 被调用，每次调用针对一个具体场景，使用专属的 system prompt 和 context 模板。

## 当前角色 vs 未来演进

| 阶段 | 角色 | 能力 |
|------|------|------|
| 现在 | **Steward** | 基于规则 + session context 提供操作性 guidance |
| 将来 | Mentor | 基于 KB 积累的经验传授 |
| 将来 | Coach | 方法论和优化思路指导 |
| 将来 | Critic | 批判性思维，发现方案漏洞 |
| 将来 | Strategist | 多方案策略选择 |
| 将来 | Consultant | Supervisor 的战略级顾问 |

---

## Supervisor ↔ Steward 接口

### 请求（Supervisor → Steward）

```python
@dataclass
class StewardRequest:
    scenario: str                     # 场景名
    task_description: str             # 原始任务
    session_summary: str              # SessionLog.to_summary()
    run_tag: str                      # 当前 run_tag
    elapsed_seconds: float            # 已运行时间
    event_specific: dict              # 场景特定数据
```

### 响应（Steward → Supervisor）

```python
@dataclass
class StewardResponse:
    action: str                       # 具体动作
    detail: str                       # 附带信息（guidance 文本、拒绝原因等）
    reasoning: str                    # 分析文本
    intervention_level: int           # 1=inline, 2=inject, 3=restart, 4=kill
```

### Intervention Level

```
Level 1: Inline      无中断，信息流回 Solver 或 Supervisor 记录
Level 2: Inject      Supervisor interrupt Solver → resume with new context
Level 3: Restart     当前 Solver session 结束，启动新 iteration（新 prompt）
Level 4: Kill        立即终止 Solver，不再 resume
```

### Supervisor 处理逻辑

```python
async def _handle_steward_response(self, response: StewardResponse):
    if response.intervention_level == 1:
        # Inline — 记录，无动作（或直接返回给 MCP tool）
        pass
    elif response.intervention_level == 2:
        # Inject — interrupt Solver，注入 guidance 后 resume
        await self._solver_runner.interrupt()
        await self._solver_runner.resume(response.detail)
    elif response.intervention_level == 3:
        # Restart — 当前 session 结束，由 run_task for 循环启动新 iteration
        pass
    elif response.intervention_level == 4:
        # Kill — 立即终止 Solver，不再 resume
        await self._solver_runner.interrupt()
```

---

## 六个触发场景

### 场景 1: ask_question

Solver 通过 `ask_supervisor` MCP tool 主动提问。

**触发方式：** Solver 调用 `ask_supervisor` MCP tool
**Intervention Level：** 1 (inline)

**Supervisor → Steward：**
- task_description: 原始任务
- session_summary: 当前 session 摘要
- question: Solver 的问题
- solver_context: Solver 附带的上下文

**System Prompt：**
```
你是 Steward，Solver 正在优化 CUDA kernel 并向你请教。

回答原则：
- 给出明确的技术判断，不要模棱两可
- 如果涉及多个选项，推荐一个并说明理由
- 如果你不确定，说明不确定的原因
- 控制在 300 字以内
```

**Steward → Supervisor：** 自由文本回答（直接通过 MCP tool 返回 Solver）

**Supervisor 动作：** 无需额外动作，答案已通过 MCP tool 注入 Solver 的对话

---

### 场景 2: permission

Solver 请求执行受限操作。

**触发方式：** PermissionRequest hook
**Intervention Level：** 1 (inline)

**Supervisor → Steward：**
- tool_name: 请求的工具名
- tool_input: 工具参数
- task_description: 原始任务
- recent_tool_calls: 最近的工具调用记录

**System Prompt：**
```
你是 Steward，审核 Solver 的工具使用请求。

判断原则：
- 操作目标是否在允许范围内？（data/gen/ 和 ~/.cuda_exec/<run_tag>/）
- 命令是否安全？（rm -rf, 修改系统文件 → DENY）
- 操作是否跟当前任务相关？

回答格式：
第一行必须是 ALLOW 或 DENY
第二行起是理由
```

**Steward → Supervisor：** ALLOW / DENY + reasoning
**Supervisor 动作：** 传回 hook 的 permissionDecision（allow 或 block）

---

### 场景 3: stuck

Monitor 检测到 Solver 停滞（idle 或循环）。

**触发方式：** AgentMonitor 检测到 idle_timeout 或 loop_detected
**Intervention Level：** 1 (CONTINUE) / 2 (INJECT) / 4 (INTERRUPT)

**Supervisor → Steward：**
- alert_type: idle_timeout / loop_detected
- alert_details: 具体描述
- task_description: 原始任务
- recent_events: 最近 10 个事件
- tool_call_counts: 各工具调用次数
- elapsed_time: 已运行时间

**System Prompt：**
```
你是 Steward，Solver 在优化 kernel 时停滞了。诊断情况并建议下一步。

诊断要点：
- Solver 是在重复相同操作？（loop）
- Solver 是否空闲太久？（可能在长时间思考，也可能卡死）
- 是否有明显的错误模式无法恢复？
- 是否有它没尝试过的替代方案？

回答格式：
第一行必须是以下之一：
- CONTINUE — Solver 可能还在有效思考
- INJECT:<具体指导> — 注入新方向让 Solver 尝试
- INTERRUPT — 停止 Solver，情况不可恢复

第二行起是分析和理由。
```

**Steward → Supervisor：**
- CONTINUE → intervention_level=1, Supervisor 不做任何事
- INJECT:\<guidance\> → intervention_level=2, Supervisor interrupt Solver → resume with guidance
- INTERRUPT → intervention_level=4, Supervisor kill Solver

---

### 场景 4: session_end

Solver 声称完成任务（最关键的场景）。

**触发方式：** StopEvent (Solver 停止运行)
**Intervention Level：** 1 (ACCEPT) / 3 (REJECT/RETRY)

**Supervisor → Steward：**
- task_description: 原始任务
- result_text: Solver 的最终输出
- stop_reason: 停止原因 (end_turn / max_turns / error)
- elapsed_time: 已运行时间
- total_tool_calls: 工具调用总次数
- error_count: 错误总次数
- session_summary: 完整 session 摘要

**System Prompt：**
```
你是 Steward，审核 Solver 是否真正完成了任务。

审核要点：
1. Solver 的输出是否满足原始任务要求？
2. 所有必要步骤是否完成？（编译？正确性测试？性能试运行？）
3. 是否有未完成工作的迹象：
   - 错误发生后未修复
   - 遇到困难后突然声称完成
   - 输出与任务描述不匹配
   - 运行很久但产出很少
4. 工作质量是否可接受？

回答格式：
第一行必须是以下之一：
- ACCEPT — 确认完成，工作符合要求
- REJECT:<原因> — 不接受，说明缺失或错误
- RETRY:<具体指导> — 让 Solver 带着具体指导重试

第二行起是详细评估。
```

**Steward → Supervisor：**
- ACCEPT → intervention_level=1, Supervisor 标记 task 完成
- REJECT:\<reason\> → intervention_level=3, Supervisor 构建新 prompt（含拒绝原因），启动新 Solver iteration
- RETRY:\<guidance\> → intervention_level=3, Supervisor 构建新 prompt（含具体指导），启动新 Solver iteration

---

### 场景 5: time_limit

Solver 运行时间超过 total_timeout。

**触发方式：** AgentMonitor 检测到 total_timeout
**Intervention Level：** 1 (EXTEND) / 2 (WRAP_UP) / 4 (KILL)

**Supervisor → Steward：**
- elapsed_time: 已运行时间
- time_limit: 设定的时间限制
- task_description: 原始任务
- recent_progress: 最近的事件（判断是否在进步）
- tool_call_trend: 最近 20 个工具调用序列

**System Prompt：**
```
你是 Steward，Solver 运行超时。决定是否应该继续。

判断原则：
- Solver 最近是否在取得进展？（新的编译成功、性能提升？）
- 还是在原地打转？
- 已经取得的成果是否值得保存？
- 还需要多少额外时间？

回答格式：
第一行必须是以下之一：
- EXTEND:<分钟数> — 延长时间（如 EXTEND:30）
- WRAP_UP — 让 Solver 保存当前进度并优雅结束
- KILL — 立即终止，不再有价值

第二行起是分析。
```

**Steward → Supervisor：**
- EXTEND:\<minutes\> → intervention_level=1, Supervisor 调大 monitor 的 total_timeout
- WRAP_UP → intervention_level=2, Supervisor inject "保存进度并结束"
- KILL → intervention_level=4, Supervisor 立即终止

---

### 场景 6: progress_check

定期进度评估，即使 Solver 活跃时也触发。

**触发方式：** Monitor 每 N 分钟主动检查（独立于 idle/loop 检测）
**Intervention Level：** 1 (ON_TRACK) / 2 (DRIFTING/REDIRECT)

**Supervisor → Steward：**
- task_description: 原始任务
- session_summary: 当前 session 摘要
- elapsed_time: 已运行时间
- recent_events: 最近事件
- tool_call_counts: 工具使用统计

**System Prompt：**
```
你是 Steward，定期检查 Solver 的进展方向。

判断要点：
- Solver 是否在朝正确方向前进？
- 是否在做无意义的工作？（反复修改同一行、没有测试就声称完成等）
- 是否跑偏了？（任务是优化 matmul 却在改 fa4 的代码）
- 进度是否合理？（运行 30 分钟但还在读文件、没有开始编码）

回答格式：
第一行必须是以下之一：
- ON_TRACK — Solver 进展正常
- DRIFTING:<concern> — Solver 有偏离迹象，说明担忧
- REDIRECT:<new direction> — Solver 需要修正方向

第二行起是分析。
```

**Steward → Supervisor：**
- ON_TRACK → intervention_level=1, Supervisor 记录但不干预
- DRIFTING:\<concern\> → intervention_level=2, Supervisor inject 方向修正
- REDIRECT:\<new direction\> → intervention_level=2, Supervisor inject 新方向

---

## 各场景的 Intervention Level 汇总

```
Level 1: Inline      无中断
         ├── ask_question → 答案
         ├── permission → ALLOW/DENY
         ├── stuck → CONTINUE
         ├── session_end → ACCEPT
         ├── time_limit → EXTEND
         └── progress_check → ON_TRACK

Level 2: Inject      interrupt Solver → resume with new context
         ├── stuck → INJECT:<guidance>
         ├── time_limit → WRAP_UP
         └── progress_check → DRIFTING/REDIRECT

Level 3: Restart     当前 session 结束，启动新 iteration
         └── session_end → REJECT/RETRY

Level 4: Kill        立即终止
         ├── stuck → INTERRUPT
         ├── time_limit → KILL
         └── hard_limit → (无条件，不经过 Steward)
```

---

## 设计原则

**所有 agent（包括 Steward）都必须通过 AgentRunner 调用。** 这确保：
- 所有 agent 的调用都有 journal（events.jsonl + transcript.md）
- 所有 agent 都受 Monitor 监控
- 统一的 hook 和 event 体系

当前 Steward 通过 ResponseRouter 直接调用 SDK query() — 这是临时实现，
应改为通过 AgentRunner 调用，使 Steward 的 trajectory 也记录在 agent_journal/ 中。

## 实现文件

```
agents/
  steward.py             # Steward class（简单 query 封装）
  response_router.py     # ResponseRouter（场景路由 + context 构建 + verdict 解析）

conf/agent/
  response_prompts/      # 每个场景的 system prompt
    ask_question.md
    permission.md
    stuck.md
    session_end.md
    time_limit.md
    progress_check.md    # 新增

agents/
  supervisor.py          # Supervisor 实现 EventHandler，调用 ResponseRouter
  monitor.py             # AgentMonitor 触发 stuck/time_limit/progress_check
```
