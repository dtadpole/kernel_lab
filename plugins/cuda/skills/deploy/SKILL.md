---
name: deploy
description: Deploy, manage, or check the cuda_exec service on remote GPU hosts
user-invocable: true
argument-hint: <start|stop|status|deploy> <host>
---

# CUDA Exec Service Deployment

Deploy and manage the cuda_exec service on remote GPU hosts via SSH.

## Actions

### deploy — Full deployment to a remote host
Syncs code, installs deps, sets up systemd service, and starts it.

```bash
bash plugins/cuda/deploy/deploy.sh <host> [port]
```

Example: `bash plugins/cuda/deploy/deploy.sh _one 8000`

### status — Check service status
```bash
ssh <host> 'systemctl --user status cuda-exec'
```

### logs — View service logs
```bash
ssh <host> 'journalctl --user -u cuda-exec -f'
```

### stop — Stop the service
```bash
ssh <host> 'systemctl --user stop cuda-exec'
```

### start — Start the service
```bash
ssh <host> 'systemctl --user start cuda-exec'
```

### restart — Restart the service
```bash
ssh <host> 'systemctl --user restart cuda-exec'
```

## Available Hosts

Defined in `plugins/cuda/deploy/hosts.json`:

| Host | SSH Alias | Port | GPU |
|------|-----------|------|-----|
| _one | `_one` | 8000 | RTX PRO 6000 Blackwell (98GB) |
| _two | `_two` | 8000 | RTX PRO 6000 Blackwell (98GB) |

## Remote Directory Convention

| Path | Purpose |
|------|---------|
| `~/.cuda_exec_service/` | Service code, venv, config |
| `~/.cuda_exec/` | Runtime data (turns, artifacts, logs) |
| `~/.keys/cuda_exec.key` | Bearer token for API auth |
| `~/.config/systemd/user/cuda-exec.service` | systemd unit |

## Deployment Flow

1. Install `uv` on remote host (if missing)
2. Rsync `cuda_exec/` source + `conf/` to `~/.cuda_exec_service/`
3. Create venv and install Python dependencies
4. Generate bearer token (if missing)
5. Install hardened systemd user service
6. Start service and verify via `/healthz`

## Health Check

```bash
ssh <host> 'curl -s http://127.0.0.1:8000/healthz'
# {"ok":true,"service":"cuda_exec"}
```

## Files

- `plugins/cuda/deploy/deploy.sh` — Deployment script
- `plugins/cuda/deploy/cuda-exec.service` — Hardened systemd unit template
- `plugins/cuda/deploy/hosts.json` — Host registry
