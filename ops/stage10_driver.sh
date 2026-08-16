#!/bin/bash
# Stage 10 driver: escape-hatch ablation, English single-hop control, 27B
# entity-stratum probe. Same single-model-at-a-time sequencing as stage 9:
# glimmer work while llama-server is resident, then swap to ollama.
cd "$(dirname "$0")/.." || { echo "STAGE10 DRIVER FAILED (cd)"; exit 1; }
PY=./.venv/bin/python
LOG=reports/stage10_driver.log
FAILED=0
fail() { echo "$(date '+%H:%M') FAIL: $*" >> "$LOG"; FAILED=1; }
step() { echo "$(date '+%H:%M') $*" >> "$LOG"; }

step "driver up"

# llama-server: reuse if healthy, start from the pinned launch line if not
if ! curl -sf --max-time 5 http://localhost:8095/health >/dev/null; then
  step "starting llama-server"
  nohup models/llama-b10353/llama-server -m models/muse-glimmer-30B-kquant-17gb.gguf \
    --port 8095 -c 8192 --jinja --reasoning off --reasoning-format deepseek \
    >> reports/llama_server.log 2>&1 &
  for _ in $(seq 1 60); do
    curl -sf --max-time 5 http://localhost:8095/health >/dev/null && break
    sleep 5
  done
fi
if ! curl -sf --max-time 5 http://localhost:8095/health >/dev/null; then
  fail "llama-server did not come up"; echo "STAGE10 DRIVER FAILED (see $LOG)"; exit 1
fi

step "hatch ablation (glimmer, 240 RAG calls)"
$PY -u ops/escape_hatch_ablation.py >> "$LOG" 2>&1 || fail "hatch run exited $?"

step "english control: prep + retrieval + glimmer arms"
$PY -u ops/prep_en_squad.py >> "$LOG" 2>&1 || fail "en prep exited $?"
if [ "$FAILED" -eq 0 ]; then
  $PY -u ops/stage9_lang.py --lang en --retrieval >> "$LOG" 2>&1 || fail "en retrieval exited $?"
  $PY -u ops/stage9_lang.py --lang en --model glimmer >> "$LOG" 2>&1 || fail "en glimmer exited $?"
fi

step "stopping llama-server; ollama arms"
pkill -f "llama-server" 2>/dev/null
sleep 5
if ! curl -sf --max-time 10 http://localhost:11434/api/tags >/dev/null; then
  fail "ollama not reachable"; echo "STAGE10 DRIVER FAILED (see $LOG)"; exit 1
fi
$PY -u ops/stage9_lang.py --lang en --model qwen8b >> "$LOG" 2>&1 || fail "en qwen8b exited $?"
$PY -u ops/recency_vs_scale_27b.py >> "$LOG" 2>&1 || fail "27b run exited $?"

# unrecovered-error check, stage-9-driver semantics (latest row per key wins)
for f in reports/stage10_hatch.jsonl reports/stage9_en.jsonl reports/stage10_27b_inf.jsonl; do
  [ -f "$f" ] || { fail "$f missing"; continue; }
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
  [ "$n" -eq 0 ] || fail "$f: $n unrecovered error rows"
done

if [ "$FAILED" -eq 0 ]; then
  step "analyzing"
  $PY ops/escape_hatch_ablation.py --analyze > /dev/null 2>>"$LOG" || fail "hatch analyze"
  $PY ops/stage9_lang.py --lang en --analyze > /dev/null 2>>"$LOG" || fail "en analyze"
  $PY ops/stage9_lang.py --lang en --calibration > /dev/null 2>>"$LOG" || fail "en calibration"
  $PY ops/recency_vs_scale_27b.py --analyze > /dev/null 2>>"$LOG" || fail "27b analyze"
fi

if [ "$FAILED" -ne 0 ]; then
  step "stage10 driver FAILED"; echo "STAGE10 DRIVER FAILED (see $LOG)"; exit 1
fi
step "stage10 driver complete"
echo "STAGE10 DRIVER COMPLETE"
