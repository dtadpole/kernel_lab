#!/usr/bin/env bash
# Deploy cuda_exec service to a remote host as a systemd user service.
#
# Usage: deploy.sh <ssh_host> [port]
#   ssh_host: SSH host alias (e.g. _one, _two)
#   port:     uvicorn listen port (default: 8000)
#
# Prerequisites:
#   - SSH key auth configured for the target host
#   - Target host has Python 3.12+, CUDA toolkit at /usr/local/cuda
#
# What this does:
#   1. Installs uv on the remote host (if missing)
#   2. Syncs cuda_exec source code to ~/.cuda_exec_service/
#   3. Creates/updates venv and installs dependencies
#   4. Generates bearer token key (if missing)
#   5. Installs and starts systemd user service

set -euo pipefail

SSH_HOST="${1:?Usage: deploy.sh <ssh_host> [port]}"
PORT="${2:-8000}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

REMOTE_SERVICE_DIR=".cuda_exec_service"
REMOTE_KEY_DIR=".keys"
REMOTE_KEY_PATH=".keys/cuda_exec.key"
SERVICE_NAME="cuda-exec"

echo "=== Deploying cuda_exec to ${SSH_HOST} (port ${PORT}) ==="
echo ""

# Step 1: Install uv if missing
echo "[1/6] Checking uv..."
ssh "$SSH_HOST" "
    if ! command -v uv >/dev/null 2>&1; then
        echo '  Installing uv...'
        curl -LsSf https://astral.sh/uv/install.sh | sh
        echo '  uv installed.'
    else
        echo '  uv already installed.'
    fi
"

# Step 2: Sync source code
echo ""
echo "[2/6] Syncing source code..."
# Create the remote directory structure
ssh "$SSH_HOST" "mkdir -p ~/${REMOTE_SERVICE_DIR}"

# Sync cuda_exec package and conf (needed for Hydra configs)
rsync -az --delete \
    --exclude='__pycache__' \
    --exclude='.venv' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='tests/' \
    "${REPO_ROOT}/cuda_exec/" \
    "${SSH_HOST}:~/${REMOTE_SERVICE_DIR}/cuda_exec/"

rsync -az --delete \
    --exclude='__pycache__' \
    "${REPO_ROOT}/conf/" \
    "${SSH_HOST}:~/${REMOTE_SERVICE_DIR}/conf/"

# Sync pyproject.toml for dependency installation
rsync -az \
    "${REPO_ROOT}/pyproject.toml" \
    "${SSH_HOST}:~/${REMOTE_SERVICE_DIR}/pyproject.toml"

echo "  Source synced."

# Step 3: Create venv and install dependencies
echo ""
echo "[3/6] Installing dependencies..."
ssh "$SSH_HOST" "
    cd ~/${REMOTE_SERVICE_DIR}
    export PATH=\$HOME/.local/bin:\$PATH

    if [ ! -d .venv ]; then
        echo '  Creating venv...'
        uv venv .venv --python 3.12
    fi

    echo '  Installing packages...'
    uv pip install --python .venv/bin/python \
        'fastapi>=0.116,<1.0' \
        'uvicorn[standard]>=0.35,<1.0' \
        torch \
        pydantic \
        psutil \
        ninja \
        httpx
    echo '  Dependencies installed.'
"

# Step 4: Generate bearer token key if missing
echo ""
echo "[4/6] Checking bearer token..."
ssh "$SSH_HOST" "
    mkdir -p ~/${REMOTE_KEY_DIR}
    chmod 700 ~/${REMOTE_KEY_DIR}
    if [ ! -f ~/${REMOTE_KEY_PATH} ]; then
        python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > ~/${REMOTE_KEY_PATH}
        chmod 600 ~/${REMOTE_KEY_PATH}
        echo '  Generated new bearer token.'
    else
        echo '  Bearer token already exists.'
    fi
"

# Step 5: Install systemd service
echo ""
echo "[5/6] Installing systemd service..."

# Generate service file with correct port
sed "s/--port 8000/--port ${PORT}/" \
    "${SCRIPT_DIR}/cuda-exec.service" | \
    ssh "$SSH_HOST" "
        mkdir -p \$HOME/.config/systemd/user
        cat > \$HOME/.config/systemd/user/${SERVICE_NAME}.service
    "

ssh "$SSH_HOST" "
    systemctl --user daemon-reload
    systemctl --user enable ${SERVICE_NAME}.service
    echo '  Service installed and enabled.'
"

# Step 6: Start/restart service
echo ""
echo "[6/6] Starting service..."
ssh "$SSH_HOST" "
    systemctl --user restart ${SERVICE_NAME}.service
    sleep 2
    if systemctl --user is-active --quiet ${SERVICE_NAME}.service; then
        echo '  Service is running.'
    else
        echo '  ERROR: Service failed to start!'
        systemctl --user status ${SERVICE_NAME}.service --no-pager
        exit 1
    fi
"

# Verify
echo ""
echo "=== Verifying deployment ==="
REMOTE_KEY=$(ssh "$SSH_HOST" "cat ~/${REMOTE_KEY_PATH}")
HEALTH=$(ssh "$SSH_HOST" "curl -sf http://127.0.0.1:${PORT}/healthz" 2>&1) || true

if echo "$HEALTH" | grep -q '"ok":true'; then
    echo "  Health check: OK"
else
    echo "  Health check: FAILED"
    echo "  Response: ${HEALTH}"
    echo ""
    echo "  Checking logs..."
    ssh "$SSH_HOST" "journalctl --user -u ${SERVICE_NAME} --no-pager -n 30"
    exit 1
fi

echo ""
echo "=== Deployment complete ==="
echo "  Host:    ${SSH_HOST}"
echo "  Port:    ${PORT}"
echo "  Service: ${SERVICE_NAME}.service"
echo "  Key:     ${REMOTE_KEY_PATH}"
echo ""
echo "  Remote bearer token (first 20 chars): ${REMOTE_KEY:0:20}..."
echo ""
echo "  To check status:  ssh ${SSH_HOST} 'systemctl --user status ${SERVICE_NAME}'"
echo "  To view logs:     ssh ${SSH_HOST} 'journalctl --user -u ${SERVICE_NAME} -f'"
echo "  To stop:          ssh ${SSH_HOST} 'systemctl --user stop ${SERVICE_NAME}'"
