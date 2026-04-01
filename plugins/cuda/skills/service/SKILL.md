---
name: service
description: Deploy, start, stop, check status, or remove the cuda_exec service on remote GPU hosts
user-invocable: true
argument-hint: <deploy|start|stop|status|nuke> <host|--all>
---

# CUDA Exec Service Management

Manage the cuda_exec service on remote GPU hosts.

## Commands

| Command | Purpose |
|---------|---------|
| **deploy** | Sync code + install deps + configure systemd (does not start) |
| **start** | Start the service (with health check) |
| **stop** | Stop the service |
| **status** | Full status report (service, health, GPU, disk, logs) |
| **nuke** | Nuclear option: stop + remove service + optionally remove data |

## Usage

```bash
python plugins/cuda/deploy/cli.py <command> <host> [flags]
```

### Deploy and start
```bash
python plugins/cuda/deploy/cli.py deploy _one
python plugins/cuda/deploy/cli.py start _one
```

### Check status
```bash
python plugins/cuda/deploy/cli.py status _one
python plugins/cuda/deploy/cli.py status --all
```

### Stop
```bash
python plugins/cuda/deploy/cli.py stop _one
```

### Update code and restart
```bash
python plugins/cuda/deploy/cli.py deploy _one
python plugins/cuda/deploy/cli.py start _one
```

### Force rebuild (new deps)
```bash
python plugins/cuda/deploy/cli.py deploy _one --rebuild
python plugins/cuda/deploy/cli.py start _one
```

### Nuclear cleanup
```bash
python plugins/cuda/deploy/cli.py nuke _one          # remove service, keep data
python plugins/cuda/deploy/cli.py nuke _one --data    # remove everything
```

## Hosts

Configured in `conf/hosts/default.yaml`:

| Host | GPU | Port |
|------|-----|------|
| `_one` | RTX PRO 6000 Blackwell (98GB) | 8000 |
| `_two` | RTX PRO 6000 Blackwell (98GB) | 8000 |

## Remote Layout

| Path | Purpose |
|------|---------|
| `~/.cuda_exec_service/` | Code, venv, config |
| `~/.cuda_exec/` | Runtime data (turns, artifacts, logs) |
| `~/.keys/cuda_exec.key` | API bearer token |
