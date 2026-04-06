# Kernel Lab — System Design

Source: Google Drive / Obsidian Notes / Kernel Lab System Design.md

## 系统架构

```
Supervisor（只读，在线编排）
│
├── Solver          生成/修改 kernel 代码
├── Harness         提供可执行 Skill
│   ├── Skill docstring → 第一层，够用直接调用
│   └── Skill 动态引用 KB → 需要深入时才读
├── Benchmarker     独立，客观
│   ├── 编译检查
│   ├── 正确性验证
│   └── 完整 benchmark，结果直接给 Supervisor
│
└── 触发 Reflector
    ├── 定时触发        每轮 session 结束后
    └── Supervisor 触发
        ├── Solver 反复犯同样错误
        ├── 出现意外好结果
        ├── 探索方向明显跑偏
        └── 遇到 KB 没有的新情况

Reflector（离线 + 按需）
└── 读 Episodic → 分析 → 更新 KB

Knowledge Base（Markdown Wiki）
├── Declarative Memory    概念 + 关系（wikilink）
├── Procedural Memory     原理 + 边界条件
├── Episodic Memory       Session log
└── 搜索：BM25
```

## 关键决策

| 决策 | 结论 | 理由 |
|------|------|------|
| Benchmarker 独立还是 Solver 工具 | 必须独立 | Solver 不能既是运动员又是裁判 |
| Validator | 删掉 | Benchmarker 兼做，没有独立存在的必要 |
| Supervisor 角色 | 只读 | 不写代码，不修改文件，只编排决策 |
| Skill 放哪里 | Harness | KB 只存知识，不存 Skill |
| Skill 和 KB 的关系 | Skill 动态引用 KB | KB 更新，Skill 行为自动跟着变 |
| Reflector 触发 | 定时 + Supervisor 主动 | 两种方式并存 |
| KB 形式 | Markdown Wiki | LLM 原生可读写 |
| KB 搜索 | BM25 | 现阶段够用，有语义 gap 再加 Dense |

## Git Repo 拆分

两个 Repo：

### `kernel-lab`（Code Repo）

```
kernel-lab/
├── agents/              # Supervisor, Solver, Reflector
├── harness/             # Skills (compile, evaluate, profile)
├── evaluator/           # 独立评测
├── services/            # cuda_exec, kb_embed
├── data/
│   ├── fixtures/        # 测试 configs, reference impls
│   ├── generated/       # kernel 代码 (by Solver)
│   └── nvidia-docs/     # CUDA 文档原始文件
├── results/             # benchmark 结果
└── docs/                # 项目文档
```

### `kernel-lab-kb`（KB Repo）

```
kernel_lab_kb/
│
│  ── 执行记录 ──
├── ik_bench/                # benchmark 执行记录
│   ├── runs/                  每次 bench run
│   └── gems/                  只存突破性结果
├── agent_journal/           # agent session 完整轨迹
│   └── <agent>/<task_slug>/<session_id>/
│       ├── meta.json            元数据
│       ├── events.jsonl         结构化事件流
│       ├── transcript.md        可读对话记录
│       ├── config_snapshot.yaml  配置快照
│       └── artifacts/           产出物
│
│  ── 知识 ──
├── declarative/             # 是什么：概念 + 关系 (wikilink)
│   ├── sm90-wgmma.md
│   ├── tma-descriptors.md
│   └── ...
├── procedural/              # 怎么做：技术 + 约束 + 偏好 + 决策模式
│   ├── warp-specialization.md
│   ├── barrier-reinit.md
│   └── ...
├── episodic/                # 提炼后的 session 摘要 + insights
│   ├── 2026-04-03-matmul-92-to-102.md
│   └── ...                    Reflector 从 agent_journal 提炼
│
└── index/                   # BM25 索引
```

#### 三层知识体系

| 层 | 目录 | 内容 | 回答什么问题 | 来源 |
|----|------|------|-------------|------|
| Declarative | `declarative/` | 事实、概念、属性、关系 | "X 是什么？" | 文档、规格书 |
| Procedural | `procedural/` | 技术、策略、偏好、决策模式 | "遇到 X 该怎么做？" | 经验、用户、Reflector |
| Episodic | `episodic/` | Session 摘要、insights、反思 | "上次做 X 发生了什么？" | Reflector 从 agent_journal 提炼 |

#### 执行记录 vs 知识的关系

```
agent_journal/  ──→  Reflector  ──→  episodic/     (摘要)
(原始轨迹)            (提炼)    ──→  declarative/  (新概念)
                               ──→  procedural/   (新策略)
```

- `agent_journal/` = 原始数据，每次 session 无条件记录
- `episodic/` = 消化后的摘要和 insights，由 Reflector 或 knowledge agent 产出
- 两者是上下游关系，不是重复

#### 知识格式规范：Markdown Wiki + Backlink

所有内容统一使用 Markdown 文件。KB 按 Wiki 模式组织：知识点之间通过 wikilink 互相引用，形成可遍历的知识图谱。

##### 1. 统一文件格式

所有文件（agent_journal, declarative, procedural, episodic）均为 `.md` 格式，带 YAML frontmatter：

```yaml
---
slug: wgmma-interleaved-layout        # 唯一标识符，用于 wikilink
title: WGMMA 要求 Interleaved SMEM Layout
type: declarative                      # declarative | procedural | episodic | journal
created: 2026-04-05
sources:                               # backlink 到原始证据
  - agent_journal/solver/matmul_wgmma/s001_20260405_143000/transcript.md
  - ik_bench/gems/matmul/sm90/gen-cuda/v002_20260405_021157/report.md
tags: [wgmma, smem, layout, sm90]
---
```

##### 2. Wikilink 语法

使用 `[[slug]]` 引用其他知识点，`[[slug|显示文本]]` 自定义显示：

```markdown
# WGMMA 要求 Interleaved SMEM Layout

WGMMA 指令（见 [[sm90-wgmma]]）要求操作数 B 在 SMEM 中按 interleaved 方式排列，
不能使用 row-major layout。

## 实践要点

- 使用 [[tma-descriptors]] 加载数据时，swizzle 模式必须匹配（见 [[tma-swizzle-constraints]]）
- 如果从 global memory 直接拷贝，需要显式做 SMEM rearrange（见 [[smem-rearrange-pattern]]）
- 相关优化策略见 [[wgmma-optimization-strategy]]
```

##### 3. Backlink 规则

**核心原则：所有高阶知识必须可溯源到原始证据。**

| 知识层 | 必须 backlink 到 | 示例 |
|--------|-----------------|------|
| `episodic/` 摘要 | `agent_journal/` 原始轨迹 | `sources: [agent_journal/solver/matmul_wgmma/s001_.../transcript.md]` |
| `declarative/` 概念 | 文档来源 或 发现该概念的 session | `sources: [data/nvidia-docs/ptx-isa.html#wgmma]` |
| `procedural/` 策略 | 验证该策略的 benchmark 或 session | `sources: [ik_bench/gems/.../report.md, agent_journal/.../transcript.md]` |
| `procedural/` 偏好 | 用户原话所在的 session | `sources: [agent_journal/.../transcript.md#L42]` |

Backlink 使用相对于 KB repo 根目录的路径。可以精确到文件内的行号（`#L42`）或章节（`#section-name`）。

##### 4. 知识图谱结构

```
declarative/sm90-wgmma.md
  ├── [[wgmma-interleaved-layout]]     → declarative/
  ├── [[tma-descriptors]]              → declarative/
  └── [[wgmma-optimization-strategy]]  → procedural/
        ├── [[tma-swizzle-constraints]] → procedural/
        └── sources: ik_bench/gems/...  → 执行记录（backlink）

episodic/2026-04-05-wgmma-session.md
  ├── sources: agent_journal/solver/... → 执行记录（backlink）
  ├── [[sm90-wgmma]]                   → declarative/（forward link）
  └── [[wgmma-optimization-strategy]]  → procedural/（forward link）
```

知识点之间通过 `[[wikilink]]` 形成 forward link，通过 `sources:` frontmatter 形成 backlink。两者共同构成可遍历的知识图谱。

##### 5. Agent Journal 的格式化记录

Agent journal 的 `transcript.md` 必须格式化记录所有 input/output：

```markdown
---
slug: s001-20260405-143000-solver-matmul-wgmma
type: journal
agent: solver
task: matmul_wgmma
session_id: s001_20260405_143000
---

# Session: solver @ 2026-04-05 14:30

## Task
优化 matmul kernel 的 wgmma 调度

## Transcript

### Turn 1
**Input (prompt):**
> 优化 matmul kernel，目标 SM90，使用 WGMMA

**Output (assistant):**
> 我来分析当前的 kernel 实现...

**Tool: Read** `data/generated/matmul_v3.cu`
> (2400 bytes)

### Turn 2
**Tool: Edit** `data/generated/matmul_v3.cu`
> 修改 SMEM layout 为 interleaved

**Tool: Bash** `cd cuda_exec && python compile.py ...`
> 编译成功，ptxas: 48 registers, 32768 bytes smem

...
```

### 为什么拆两个

| 维度 | Code Repo | KB Repo |
|------|-----------|---------|
| 写入者 | 用户 + Solver | Reflector |
| 更新频率 | 每次优化 session | 每轮 session 结束 |
| 内容性质 | 可执行代码 | Markdown 文档 |
| 读取者 | 开发者 | 所有 Agent |
| 跨机器共享 | 需要 GPU | 纯文本，任何机器 |

### CUDA 文档归属

CUDA 文档（PTX ISA, Programming Guide 等）放在 **Code Repo** 的 `data/nvidia-docs/`：
- 它是基础设施数据，跟搜索服务 (`doc_retrieval/`) 紧耦合
- 跟随 CUDA 版本管理
- 不是我们产出的知识

### 连接方式

Code Repo 通过 `KERNEL_LAB_KB` 环境变量指向 KB Repo 路径：
```bash
export KERNEL_LAB_KB=~/kernel-lab-kb
```

Harness 提供统一搜索接口：
- `/kb:docs` → 搜索 CUDA 文档（Code Repo `data/nvidia-docs/`）
- `/kb:search` → 搜索我们的 KB（KB Repo）

---

## Agent SDK 实现架构

基于 Claude Agent SDK（Python）的进程模型实现上述系统设计。

### 进程模型

```
┌─────────────────────────────────┐
│  Python 进程 (Supervisor)        │  asyncio event loop
│                                 │
│  ClaudeSDKClient                │
│    ├── client.query(prompt)     │  发送指令
│    ├── client.receive_response()│  接收消息流
│    └── client.interrupt()       │  中断执行
│         │                       │
│         │ stdio (JSON)          │  IPC 通道
│         ▼                       │
│  ┌──────────────────────┐       │
│  │ claude CLI 子进程     │       │  SDK 自动 spawn
│  │ (Solver agent)       │       │
│  │  ├── 调 Claude API   │       │
│  │  ├── 执行工具        │       │  Read/Edit/Bash 等
│  │  └── spawn 子 agent  │       │  如果用 Agent tool
│  └──────────────────────┘       │
└─────────────────────────────────┘
```

- **Supervisor** = 你的 Python 进程，拥有完整控制权
- **Solver** = claude CLI 子进程，被 Supervisor 通过 hooks 监控
- **Hooks** 在 Supervisor 进程中执行，可观察、注入 context、拒绝操作
- Supervisor 可随时 `interrupt()` Solver、注入新 prompt、启动新 session

### 分层 + 侧车架构

```
         ┌────────────────────────────┐     ┌───────────────────┐
Layer 4  │  User Interaction          │─────│                   │
         │  终端 TUI / 进度查询        │     │                   │
         ├────────────────────────────┤     │                   │
Layer 3  │  Multi-Role                │─────│  Knowledge System │
         │  Rigger / Benchmarker │     │  (side-car)       │
         ├────────────────────────────┤     │                   │
Layer 2  │  Supervisor                │─────│  • KnowledgeStore │
         │  编排 Solver、hooks、状态    │     │  • Preference     │
         ├────────────────────────────┤     │  • Injection API  │
Layer 1  │  Agent Runner              │─────│                   │
         │  ClaudeSDKClient 封装       │     │                   │
         ├────────────────────────────┤     │                   │
Layer 0  │  SDK + Infra               │     │                   │
         └────────────────────────────┘     └───────────────────┘
```

#### 各层职责

| 层 | 职责 | 对外接口 |
|----|------|---------|
| Layer 0 | SDK 安装、CLI 可用性验证、API key 配置 | 环境就绪的布尔断言 |
| Layer 1 | ClaudeSDKClient 封装：launch / stream / interrupt / resume | `AgentRunner` class |
| Layer 2 | 管理 Solver 生命周期、安装 hooks、追踪状态、做编排决策 | `Supervisor` class |
| Layer 3 | Rigger / Benchmarker 等角色，Supervisor 按需召唤 | `AgentDefinition` 字典 |
| Layer 4 | 长时间运行的用户交互：进度查询、preference 输入、动态调整 | 终端 TUI / stdin 循环 |
| Knowledge System | 侧车：存储/检索知识、接受 preference 写入、为任意层提供 context 片段 | `KnowledgeSystem` class |

#### Knowledge System 侧车特征

- 独立模块，不在主调用链上
- 提供统一 API：`query(topic)` → 返回相关知识片段
- 任何层都可以横向调用：Agent Runner 的 hook 可以查、Supervisor 决策时可以查、用户交互时可以查
- 接受写入：用户 preference、Reflector 产出、session episode 都写入
- 可独立测试，不依赖其他层

#### 构建顺序

| 阶段 | 构建内容 | 可并行 |
|------|---------|--------|
| Phase 0 | SDK + Infra 验证 | — |
| Phase 1 | Agent Runner | Knowledge System |
| Phase 2 | Supervisor | （继续完善 KS） |
| Phase 3 | Multi-Role agents | — |
| Phase 4 | User Interaction | — |

每层有独立的 `test_*.py`，用 mock 或真实 SDK 调用验证。上层不存在时，下层照样能独立运行。

### 详细设计文档

- **Supervisor 调度设计** — [docs/design/supervisor.md](design/supervisor.md)：事件调用链、决策循环、intervention level 处理、run_tag 管理
- **Steward 详细设计** — [docs/design/steward.md](design/steward.md)：六个触发场景的 prompt、接口定义、intervention level 体系

### ResponseRouter — Hook 场景路由

Supervisor 在每个关键 hook 点调用 Steward 做决策。每种场景有专属的 system prompt 和 context 模板。

```
Hook 事件 ──→ ResponseRouter ──→ Steward (SDK query) ──→ ResponseVerdict
                  │                                                    │
                  ├── 选择 scenario prompt                              ├── action: ACCEPT/REJECT/RETRY/...
                  ├── 构建 context (模板 + 变量)                         ├── detail: 附带信息
                  └── 调用 Agent SDK query()                            └── reasoning: 分析文本
```

#### 五种 Response 场景

| 场景 | 触发时机 | 决策选项 | Prompt 文件 |
|------|---------|---------|------------|
| `ask_question` | Solver 通过 ask_supervisor 提问 | 自由文本回答 | `conf/agent/response_prompts/ask_question.md` |
| `permission` | Solver 请求受限操作 | ALLOW / DENY | `conf/agent/response_prompts/permission.md` |
| `stuck` | Monitor 检测到 idle 或循环 | CONTINUE / INJECT / INTERRUPT | `conf/agent/response_prompts/stuck.md` |
| `session_end` | Solver 声称完成 | ACCEPT / REJECT / RETRY | `conf/agent/response_prompts/session_end.md` |
| `time_limit` | 运行时间超过 total_timeout | EXTEND / WRAP_UP / KILL | `conf/agent/response_prompts/time_limit.md` |

#### 时间限制

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `total_timeout` | 3 小时 | 触发 time_limit Steward，可被 EXTEND |
| `hard_limit` | 6 小时 | 强制 interrupt，不可延长 |

### Supervisor 决策循环

```
Supervisor.run_task(task)
  │
  for iteration in range(max_iterations):
    │
    ├── phase=solving
    │   └── AgentRunner.run(solver_config, handler=self)
    │         ├── hooks → Supervisor.on_tool_call/on_tool_result
    │         ├── ask_supervisor MCP → Supervisor.on_ask → ResponseRouter("ask_question")
    │         ├── PermissionRequest → Supervisor.on_permission → ResponseRouter("permission")
    │         ├── MonitorAlert → Supervisor.on_monitor_alert → ResponseRouter("stuck"/"time_limit")
    │         └── Stop → Supervisor.on_stop → ResponseRouter("session_end")
    │
    ├── phase=deciding
    │   └── verdict = session_end 的 ResponseVerdict
    │       ├── ACCEPT → break (done)
    │       ├── REJECT/RETRY → 构建新 prompt，继续循环
    │       └── other → break
    │
  └── return TaskResult
```

### Future Expansion — 角色演进

当前实现的角色：Solver, Benchmarker, Rigger, Steward。
以下角色为未来扩展方向，随系统成熟逐步引入：

| 角色 | 职责 | 引入条件 |
|------|------|---------|
| **Mentor** | 基于 KB 经验传授，指导 agent 从历史 session 中学习 | Knowledge System 成熟 |
| **Coach** | 方法论指导，教 Solver 如何分析问题和选择优化策略 | 成熟的 procedural memory |
| **Critic** | 批判性思维，在方案实施前发现逻辑漏洞和潜在问题 | declarative + procedural |
| **Strategist** | 从多个优化方案中选择最优策略，需要 Rigor 审查配合 | 多方案并行探索能力 |
| **Consultant** | Supervisor 的高级顾问，分析复杂局面给出战略建议 | Supervisor 决策复杂度提升 |

```
Steward（现在）
  │
  ├──→ Mentor       基于 KB 经验传授
  ├──→ Coach        方法论和思路指导
  ├──→ Critic       批判性审查
  ├──→ Strategist   多方案策略选择
  └──→ Consultant   战略级顾问
```

这些角色可以共存，由 Supervisor 在不同场景下调用不同的 guidance 角色。
