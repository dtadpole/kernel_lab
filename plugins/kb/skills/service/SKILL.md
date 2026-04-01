---
name: service
description: Deploy, start, stop, check health/status, or remove the KB embedding service on remote GPU hosts
user-invocable: true
argument-hint: <deploy|start|stop|health|status|nuke> <host|--all>
---

# KB Embedding Service Management

Manage the embedding service on remote GPU hosts.
Runs Qwen/Qwen3-Embedding-4B via Python (transformers + FastAPI + uvicorn),
exposing an OpenAI-compatible `/v1/embeddings` endpoint consumed by
`doc_retrieval` for dense and hybrid search.

## Commands

| Command | Purpose |
|---------|---------|
| **deploy** | Sync code + install deps + configure systemd (does not start) |
| **start** | Start the service (waits for model load) |
| **stop** | Stop the service |
| **health** | Quick check: is the API responding? |
| **status** | Full report: service state, health, GPU, model cache, logs |
| **nuke** | Nuclear option: stop + remove service + optionally remove model cache |

## Usage

```bash
python3 plugins/kb/deploy/cli.py <command> <host> [flags]
```

### Deploy and start
```bash
python3 plugins/kb/deploy/cli.py deploy _one
python3 plugins/kb/deploy/cli.py start _one
python3 plugins/kb/deploy/cli.py health _one
```

### Quick health check
```bash
python3 plugins/kb/deploy/cli.py health _one
python3 plugins/kb/deploy/cli.py health --all
```

### Full status
```bash
python3 plugins/kb/deploy/cli.py status _one
```

### Stop
```bash
python3 plugins/kb/deploy/cli.py stop _one
```

### Update code and restart
```bash
python3 plugins/kb/deploy/cli.py deploy _one
python3 plugins/kb/deploy/cli.py start _one
```

### Force rebuild (new deps)
```bash
python3 plugins/kb/deploy/cli.py deploy _one --rebuild
python3 plugins/kb/deploy/cli.py start _one
```

### Nuclear cleanup
```bash
python3 plugins/kb/deploy/cli.py nuke _one          # remove service, keep model cache
python3 plugins/kb/deploy/cli.py nuke _one --data    # remove everything including model
```

## Hosts

Configured in `conf/kb/default.yaml`:

| Host | GPU | Port | Description |
|------|-----|------|-------------|
| `_one` | RTX PRO 6000 Blackwell (98GB) | 46982 | Embedding service |

## Access from Dev Machine

The service runs on the remote host. Access it locally via SSH tunnel:

```bash
ssh -NL 46982:localhost:46982 _one
```

This maps `localhost:46982` to the remote service, matching the `base_url` in
`conf/doc_retrieval/default.yaml`.

## Remote Layout

| Path | Purpose |
|------|---------|
| `~/.kb_embed_service/` | Code, venv |
| `~/.kb_embed_service/embed_server/` | Server source (synced from repo) |
| `~/.kb_embed_service/.venv/` | Python venv (torch, transformers, fastapi) |
| `~/.config/systemd/user/kb-embed.service` | systemd unit |
| `~/.cache/huggingface/` | Model weights cache (persists across restarts) |
