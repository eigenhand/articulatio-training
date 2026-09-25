#!/bin/bash
# Run 3: the data of run 2 (HUI + HiFiTTS-2) plus rarely heard words
# (thorsten_rare_de, cml_rare_de), continued from step2500 of run 2.
# Continued by pipeline_v3b.sh (adds Common Voice, then trains).
#
# The code store of run 2 was copied to training-v3 beforehand (codes/ and
# dataset_meta.jsonl from training-v2); process() skips existing entries, so it
# only encodes the new corpora.
#
#   scripts/run.sh tools/analyse_seltene_woerter.py
#   scripts/run.sh tools/grossschreibung.py
#   nohup scripts/run.sh tools/suche_seltene_woerter.py > "$LOG_DIR/rare-ingest.log" 2>&1 &
#   nohup recipes/pipeline_v3.sh > "$LOG_DIR/pipeline-v3.log" 2>&1 &
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export BREEZE_TRAINING_DIR="$PTBR_ARTIFACTS/training-v3"

echo "[v3] $(date -u +%H:%M) waiting for the search"
while pgrep -f "[s]uche_seltene_woerter.py" >/dev/null; do sleep 30; done
if ! grep -q "^\[DONE\]" "$LOG_DIR/rare-ingest.log"; then echo "[v3] search ERROR"; exit 1; fi
grep "^\[DONE\]" "$LOG_DIR/rare-ingest.log"

python3 - "$CORPORA_JSON" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
an = {"hui_de", "hifitts2_en", "thorsten_rare_de", "cml_rare_de"}
for c in d: c["enabled"] = c["name"] in an
p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print("[v3] enabled:", [c["name"] for c in d if c["enabled"]])
PY

echo "[v3] $(date -u +%H:%M) encoding (new entries only)"
"$RUN" de_lora/core/prepare_dataset.py process --device cuda || { echo "[v3] process ERROR"; exit 1; }
echo "[v3] $(date -u +%H:%M) finalize"
"$RUN" de_lora/core/prepare_dataset.py finalize --gold 8 --parity-device cuda || { echo "[v3] finalize ERROR"; exit 1; }
# NOTE: recipes/pipeline_v3b.sh waits for this line ("DONE - ready") or "ERROR".
echo "[v3] $(date -u +%H:%M) DONE - ready for training"
