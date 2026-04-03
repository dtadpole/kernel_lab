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
├── Evaluator       独立，客观
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
| Evaluator 独立还是 Solver 工具 | 必须独立 | Solver 不能既是运动员又是裁判 |
| Validator | 删掉 | Evaluator 兼做，没有独立存在的必要 |
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
kernel-lab-kb/
├── declarative/         # 概念 + 关系 (wikilink)
│   ├── sm90-wgmma.md
│   ├── tma-descriptors.md
│   └── ...
├── procedural/          # 原理 + 边界条件
│   ├── warp-specialization.md
│   ├── barrier-reinit.md
│   └── ...
├── episodic/            # Session logs
│   ├── 2026-04-03-matmul-92-to-102.md
│   └── ...
└── index/               # BM25 索引
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
