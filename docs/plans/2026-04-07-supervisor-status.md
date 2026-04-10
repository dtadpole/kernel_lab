# Supervisor System — Current Status & Next Steps

## What We Built (2026-04-06 ~ 04-07)

### Core Architecture
- **Supervisor** → orchestrates Solver + Benchmarker + Steward
- **Solver** → writes raw CUDA/PTX kernels (no CUTLASS/cuDNN/cuBLAS)
- **Benchmarker** → runs formal bench independently
- **Steward** → reviews session end, progress checks, stuck detection
- **Monitor** → async watchdog, triggers Steward at intervals

### Session Lifecycle (SUCCESS / CONTINUE / ABORT)
- Solver stops → Steward reviews → verdict:
  - **SUCCESS**: gem created, task done
  - **CONTINUE**: resume same session with guidance (preserves context)
  - **ABORT**: give up
- CONTINUE uses `AgentRunner.resume()` — same session, full context preserved

### Security — Path Blacklist/Whitelist
- **Blacklist**: ~/kernel_lab/, ~/kernel_lab_kb/, ~/.cuda_exec/, ~/.claude/worktrees/, ~/.claude/projects/
- **Whitelist**: cuda_exec/, plugins/, conf/, data/ref/, data/configs/, DESIGN.md, AGENTS.md, current run (<run_tag>)
- Enforced on: Read, Glob, Grep, Bash (content-read + compile commands)
- Navigation commands (ls) allowed on blocked paths
- `--strict-mcp-config` prevents plugin MCP crashes

### Solver Constraints
- Raw CUDA/PTX only (Python Triton/CuTe DSL if task requires)
- FORBIDDEN: CUTLASS, cuDNN, cuBLAS, Thrust, CUB
- Must plan first → code skeleton with TODOs → implement step by step
- Gen code goes to ~/kernel_lab_kb/runs/<run_tag>/gen/ (data/gen/ deprecated)

### Agent Budgets & Limits
| Agent | Budget | Max Turns | Model |
|-------|--------|-----------|-------|
| Solver | $50 | 2000 | claude-sonnet-4-6 |
| Rigger | $10 | 500 | claude-sonnet-4-6 |
| Benchmarker | $5 | 20 | claude-sonnet-4-6 |
| Steward | $5 | 5-20/scenario | claude-sonnet-4-6 |

### Monitor Timings
| Parameter | Value |
|-----------|-------|
| check_interval | 60s |
| progress_check_interval | 10 min |
| idle_timeout | 30 min |
| total_timeout | 2 hours |
| hard_limit | 6 hours |
| loop_threshold | 100 |

### Infrastructure
- Process group (`os.setpgrp`) — kill Supervisor kills all children
- `include_partial_messages=True` — streaming heartbeat during thinking
- stderr logging to ~/.cuda_exec/<run_tag>/stderr_solver.log
- Prompts externalized to conf/agent/prompts/ and conf/agent/tasks/

## Known Issues
1. CLI auto-loads ~/.claude/ memory — can't fully prevent without `--bare`
2. Solver can use `ssh localhost` to bypass some path restrictions
3. No auto-restart on stream closed (relies on manual restart)

## Next Steps
1. **Heartbeat timeout** — auto-restart Solver if heartbeat stops for >5 min
2. **Steward quality** — tune session_end/progress_check prompts based on real data
3. **Multi-kernel** — run matmul + fa4 Supervisors concurrently
4. **Results tracking** — dashboard for gem progression across runs
