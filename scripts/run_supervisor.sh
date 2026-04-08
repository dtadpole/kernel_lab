#!/bin/bash
# Wrapper that monitors supervisor exit code and auto-restarts.
# Usage: ./scripts/run_supervisor.sh --kernel matmul --gpu 6

LOG_DIR="$HOME/.kernel_lab/logs"
mkdir -p "$LOG_DIR"
EXIT_LOG="$LOG_DIR/supervisor_exits.log"

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting supervisor: $@" | tee -a "$EXIT_LOG"

    .venv/bin/python -m agents.main "$@" 2>&1 | tee /tmp/supervisor_${2:-kernel}.log
    EXIT_CODE=$?

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Supervisor exited with code $EXIT_CODE" | tee -a "$EXIT_LOG"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Clean exit. Not restarting." | tee -a "$EXIT_LOG"
        break
    fi

    # Check if killed by signal
    if [ $EXIT_CODE -gt 128 ]; then
        SIGNAL=$((EXIT_CODE - 128))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Killed by signal $SIGNAL ($(kill -l $SIGNAL 2>/dev/null || echo 'unknown'))" | tee -a "$EXIT_LOG"
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting in 10s..." | tee -a "$EXIT_LOG"
    sleep 10
done
