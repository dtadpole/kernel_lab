#!/bin/bash
# Test signal handling in formal.py
# Verifies: no infinite recursion, clean exit, no orphan processes
#
# Usage: bash tests/test_signal_handling.sh [GPU_ID]

GPU=${1:-2}
VENV=.venv/bin/python
LOG=/tmp/test_signal_$$

echo "=== Test 1: SIGTERM during bench ==="
$VENV -m cuda_exec.formal bench.kernel=matmul bench.gpu=$GPU > $LOG.stdout 2> $LOG.stderr &
PID=$!
echo "  Started PID $PID on GPU $GPU"
sleep 15  # let it start compiling
echo "  Sending SIGTERM..."
kill $PID 2>/dev/null
KILL_TIME=$(date +%s)
sleep 5
if ps -p $PID --no-headers 2>/dev/null; then
    echo "  FAIL: process $PID still alive after SIGTERM"
    kill -9 $PID 2>/dev/null
else
    wait $PID 2>/dev/null
    EXIT_CODE=$?
    echo "  OK: exited with code $EXIT_CODE (expect 143 = 128+15)"
fi
# Check orphans
ORPHANS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU 2>/dev/null | wc -l)
if [ "$ORPHANS" -gt 0 ]; then
    echo "  FAIL: $ORPHANS orphan process(es) on GPU $GPU"
    nvidia-smi --query-compute-apps=pid,name --format=csv,noheader -i $GPU
    nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU | xargs kill -9 2>/dev/null
else
    echo "  OK: no orphan processes on GPU $GPU"
fi
echo ""

echo "=== Test 2: SIGINT (Ctrl+C) during bench ==="
rm -f ~/.cuda_exec/.lock_cuda_$GPU 2>/dev/null
sleep 2
$VENV -m cuda_exec.formal bench.kernel=matmul bench.gpu=$GPU > $LOG.stdout2 2> $LOG.stderr2 &
PID=$!
echo "  Started PID $PID on GPU $GPU"
sleep 15
echo "  Sending SIGINT..."
kill -INT $PID 2>/dev/null
sleep 5
if ps -p $PID --no-headers 2>/dev/null; then
    echo "  FAIL: process $PID still alive after SIGINT"
    kill -9 $PID 2>/dev/null
else
    wait $PID 2>/dev/null
    EXIT_CODE=$?
    echo "  OK: exited with code $EXIT_CODE (expect 130 = 128+2)"
fi
ORPHANS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU 2>/dev/null | wc -l)
if [ "$ORPHANS" -gt 0 ]; then
    echo "  FAIL: $ORPHANS orphan process(es) on GPU $GPU"
    nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU | xargs kill -9 2>/dev/null
else
    echo "  OK: no orphan processes on GPU $GPU"
fi
echo ""

echo "=== Test 3: Double SIGTERM (rapid kill) ==="
rm -f ~/.cuda_exec/.lock_cuda_$GPU 2>/dev/null
sleep 2
$VENV -m cuda_exec.formal bench.kernel=matmul bench.gpu=$GPU > $LOG.stdout3 2> $LOG.stderr3 &
PID=$!
echo "  Started PID $PID on GPU $GPU"
sleep 15
echo "  Sending SIGTERM twice rapidly..."
kill $PID 2>/dev/null
kill $PID 2>/dev/null
sleep 5
if ps -p $PID --no-headers 2>/dev/null; then
    echo "  FAIL: process $PID still alive after double SIGTERM"
    kill -9 $PID 2>/dev/null
else
    wait $PID 2>/dev/null
    EXIT_CODE=$?
    echo "  OK: exited with code $EXIT_CODE"
fi
ORPHANS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU 2>/dev/null | wc -l)
if [ "$ORPHANS" -gt 0 ]; then
    echo "  FAIL: $ORPHANS orphan process(es) on GPU $GPU"
    nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU | xargs kill -9 2>/dev/null
else
    echo "  OK: no orphan processes on GPU $GPU"
fi

# Cleanup
rm -f $LOG.stdout $LOG.stderr $LOG.stdout2 $LOG.stderr2 $LOG.stdout3 $LOG.stderr3
rm -f ~/.cuda_exec/.lock_cuda_$GPU 2>/dev/null
echo ""
echo "=== Done ==="
