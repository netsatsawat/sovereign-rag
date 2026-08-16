#!/bin/bash
# Stage 9 driver. Waits for the stage-8 rec armset to release llama-server,
# then runs the multilingual generation arms in the only order the 24 GB
# machine allows: all glimmer arms while llama-server is resident, then stop
# it and run all qwen8b arms via ollama, then analyze per language.
#
# Prerequisites, nothing below starts a server for you:
#   llama-server resident on :8095 (the stage-8 launch line):
#     models/llama-b10353/llama-server -m models/muse-glimmer-30B-kquant-17gb.gguf \
#         --port 8095 -c 8192 --jinja --reasoning off --reasoning-format deepseek
#   ollama on :11434 with the models pulled:
#     ollama pull qwen3:8b && ollama pull qwen3-embedding:0.6b
#   retrieval is NOT run by this driver; run it per language (the dense arm
#   needs ollama):  ./.venv/bin/python ops/stage9_lang.py --lang $L --retrieval
#
# The driver refuses to print DRIVER COMPLETE if any step exited non-zero or
# any rows file contains error rows (a dead server writes error:URLError rows
# and every $PY call still exits 0; the row check is what catches it).
cd "$(dirname "$0")/.." || { echo "STAGE9 DRIVER FAILED (cd to repo root failed)"; exit 1; }
LANGS="th ja zh es vi"
PY=./.venv/bin/python
LOG=reports/stage9_driver.log
FAILED=0
PROCESSED=0   # languages that actually ran a generation step

fail() { echo "$(date '+%H:%M') FAIL: $*" >> "$LOG"; FAILED=1; }

check_error_rows() {  # $1 = lang
  # Fail only on UNRECOVERED errors. The checkpoint recovers from an error
  # by APPENDING a fresh row: old error rows stay in the jsonl forever, so
  # a raw grep for error rows would flag every run after any crash+resume
  # cycle and train the operator to ignore FAIL. Mirror the analysis
  # semantics instead (last row per (arm, qi) wins): count pairs whose
  # LATEST row is still an error.
  local f="reports/stage9_$1.jsonl" n
  [ -f "$f" ] || { fail "$1: rows file $f missing"; return; }
  n=$("$PY" - "$f" <<'PYEOF'
import json, sys
last = {}
for line in open(sys.argv[1]):
    try:
        r = json.loads(line)
    except Exception:
        continue
    last[(r.get("arm"), r.get("qi"))] = str(r.get("done_reason", ""))
print(sum(d.startswith("error") for d in last.values()))
PYEOF
)
  [ "$n" -eq 0 ] || fail "$1: $n unrecovered error rows in $f"
}

echo "$(date '+%H:%M') driver up; waiting for stage8 rec completion" >> "$LOG"
while ! grep -q "stage8 glimmer run complete" reports/stage8_rec.log 2>/dev/null; do
  if ! pgrep -f "stage8_glimmer.py --armset rec" >/dev/null; then
    echo "$(date '+%H:%M') rec process gone without completing, proceeding anyway" >> "$LOG"
    break
  fi
  sleep 300
done

if ! curl -sf --max-time 10 http://localhost:8095/health >/dev/null; then
  fail "llama-server not reachable on :8095, glimmer arms cannot run; aborting"
  echo "STAGE9 DRIVER FAILED (see $LOG)"
  exit 1
fi

echo "$(date '+%H:%M') glimmer arms" >> "$LOG"
for L in $LANGS; do
  if [ -f "data/multiling/$L/queries.jsonl" ]; then
    PROCESSED=$((PROCESSED+1))
    $PY -u ops/stage9_lang.py --lang "$L" --model glimmer >> "$LOG" 2>&1 \
      || fail "glimmer $L exited $?"
    check_error_rows "$L"
  else
    echo "  skip $L (no data)" >> "$LOG"
  fi
done

echo "$(date '+%H:%M') stopping llama-server, qwen8b arms" >> "$LOG"
pkill -f "llama-server" 2>/dev/null
sleep 5
if ! curl -sf --max-time 10 http://localhost:11434/api/tags >/dev/null; then
  fail "ollama not reachable on :11434, qwen8b arms cannot run; aborting"
  echo "STAGE9 DRIVER FAILED (see $LOG)"
  exit 1
fi
for L in $LANGS; do
  if [ -f "data/multiling/$L/queries.jsonl" ]; then
    PROCESSED=$((PROCESSED+1))
    $PY -u ops/stage9_lang.py --lang "$L" --model qwen8b >> "$LOG" 2>&1 \
      || fail "qwen8b $L exited $?"
    check_error_rows "$L"
  else
    echo "  skip $L (no data)" >> "$LOG"
  fi
done

# The skip branches above mean a driver with both servers reachable but no
# data (failed checkout, typo'd LANGS, wrong working directory) would run
# zero steps, never touch FAILED, and print COMPLETE; the header's
# row-check safety net only executes for languages that have data. A run
# that processed nothing is a failure, not a completion.
if [ "$PROCESSED" -eq 0 ]; then
  fail "no language under data/multiling had queries.jsonl, nothing ran"
fi

# Analyze only when every generation step and error-row check passed:
# --analyze OVERWRITES reports/stage9_{lang}.json, and on a failed run
# (dead server, missing rows) it would clobber the committed summaries
# with partial-data numbers before the driver even prints FAILED.
# Analysis of a complete rows file is deterministic, so nothing is lost
# by deferring it to the rerun that passes.
if [ "$FAILED" -eq 0 ]; then
  echo "$(date '+%H:%M') analyzing" >> "$LOG"
  for L in $LANGS; do
    if [ -f "reports/stage9_$L.jsonl" ]; then
      $PY ops/stage9_lang.py --lang "$L" --analyze > /dev/null 2>>"$LOG" \
        || fail "analyze $L exited $?"
    fi
  done
else
  echo "$(date '+%H:%M') skipping analyze: a generation step FAILED, refusing to overwrite committed summaries with partial-data numbers" >> "$LOG"
fi

if [ "$FAILED" -ne 0 ]; then
  echo "$(date '+%H:%M') stage9 driver FAILED, see FAIL lines above" >> "$LOG"
  echo "STAGE9 DRIVER FAILED (see $LOG)"
  exit 1
fi
echo "$(date '+%H:%M') stage9 driver complete" >> "$LOG"
echo "STAGE9 DRIVER COMPLETE"
