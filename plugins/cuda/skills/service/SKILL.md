---
name: service
description: Deploy, start, stop, check health/status, or remove the cuda_exec service on remote GPU hosts
user-invocable: true
argument-hint: <deploy|start|stop|health|status|nuke> <host|--all>
---

# CUDA Exec Service Management

Manage the cuda_exec service on remote GPU hosts.

## Commands

| Command | Purpose |
|---------|---------|
| **deploy** | Sync code + install deps + configure systemd (does not start) |
| **start** | Start the service |
| **stop** | Stop the service |
| **health** | Quick check: is the API responding? |
| **status** | Full report: service state, health, GPU, disk, logs |
| **nuke** | Nuclear option: stop + remove service + optionally remove data |

## Usage

```bash
python plugins/cuda/deploy/cli.py <command> <host> [flags]
```

### Deploy and start
```bash
python plugins/cuda/deploy/cli.py deploy _one
python plugins/cuda/deploy/cli.py start _one
python plugins/cuda/deploy/cli.py health _one
```

### Quick health check
```bash
python plugins/cuda/deploy/cli.py health _one
python plugins/cuda/deploy/cli.py health --all
```

### Full status
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

Configured in `conf/hosts/default.yaml` under each host's `services.cuda_exec`:

| Host | GPU | Port | GPU pin |
|------|-----|------|---------|
| `_one` | 1x RTX PRO 6000 Blackwell | 41980 | — |
| `_two` | 1x RTX PRO 6000 Blackwell | 42980 | — |
| `h8_3` | 8x NVIDIA H100 | 8980 | GPU 7 |
| `h8_4` | 8x NVIDIA H100 | 8980 | GPU 7 |

## Remote Layout

| Path | Purpose |
|------|---------|
| `~/.cuda_exec_service/` | Code, venv, config |
| `~/.cuda_exec/` | Runtime data (turns, artifacts, logs) |
| `~/.keys/cuda_exec.key` | API bearer token |
