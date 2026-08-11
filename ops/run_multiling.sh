#!/bin/bash
# Stage 9 driver. Waits for the stage-8 rec armset to release llama-server,
# then runs the multilingual generation arms in the only order the 24 GB
# machine allows: all glimmer arms while llama-server is resident, then stop
# it and run all qwen8b arms via ollama, then analyze per language.
cd "$(dirname "$0")/.."
LANGS="th ja zh es vi"
PY=./.venv/bin/python
LOG=reports/stage9_driver.log

echo "$(date '+%H:%M') driver up; waiting for stage8 rec completion" >> "$LOG"
while ! grep -q "stage8 glimmer run complete" reports/stage8_rec.log 2>/dev/null; do
  if ! pgrep -f "stage8_glimmer.py --armset rec" >/dev/null; then
    echo "$(date '+%H:%M') rec process gone without completing — proceeding anyway" >> "$LOG"
    break
  fi
  sleep 300
done

echo "$(date '+%H:%M') glimmer arms" >> "$LOG"
for L in $LANGS; do
  if [ -f "data/multiling/$L/queries.jsonl" ]; then
    $PY -u ops/stage9_lang.py --lang "$L" --model glimmer >> "$LOG" 2>&1
  else
    echo "  skip $L (no data)" >> "$LOG"
  fi
done

echo "$(date '+%H:%M') stopping llama-server, qwen8b arms" >> "$LOG"
pkill -f "llama-server" 2>/dev/null
sleep 5
for L in $LANGS; do
  if [ -f "data/multiling/$L/queries.jsonl" ]; then
    $PY -u ops/stage9_lang.py --lang "$L" --model qwen8b >> "$LOG" 2>&1
  fi
done

echo "$(date '+%H:%M') analyzing" >> "$LOG"
for L in $LANGS; do
  [ -f "reports/stage9_$L.jsonl" ] && $PY ops/stage9_lang.py --lang "$L" --analyze > /dev/null 2>>"$LOG"
done
echo "$(date '+%H:%M') stage9 driver complete" >> "$LOG"
echo "STAGE9 DRIVER COMPLETE"
