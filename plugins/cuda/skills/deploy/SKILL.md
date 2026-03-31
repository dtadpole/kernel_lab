---
name: deploy
description: Start, stop, or health-check the cuda_exec service
user-invocable: true
argument-hint: <start|stop|status>
---

# TODO: Deploy Skill

Manage the cuda_exec FastAPI service lifecycle.

## Planned Workflow

- `start` — Launch uvicorn serving cuda_exec on the configured port
- `stop` — Gracefully shut down the running service
- `status` — Check /healthz endpoint and report service state

## MCP Tools Needed

- Likely shell commands rather than MCP tools:
  - `uvicorn cuda_exec.main:app --host 0.0.0.0 --port 8000`
  - `curl http://127.0.0.1:8000/healthz`
  - Process management (PID tracking, signals)

## Status: NOT IMPLEMENTED
