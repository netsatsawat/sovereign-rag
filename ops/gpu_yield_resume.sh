#!/bin/bash
# Waits until qwen-finance is no longer resident, then resumes the generation
# grid. Created 2026-08-10 when the user's own model took the shared Ollama.
cd "$(dirname "$0")/.."
while curl -s --max-time 10 http://localhost:11434/api/ps | grep -q "qwen-finance"; do
  sleep 300
done
echo "$(date '+%H:%M') qwen-finance gone — resuming grid" >> reports/stage3_full.log
nohup ./.venv/bin/python -u ops/stage3_pilot.py --n 1381 >> reports/stage3_full.log 2>&1 &
