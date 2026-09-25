#!/bin/bash
# Run 3, final part: add Common Voice and then train.
#   waits for: the preparation (Thorsten/CML encoded, pipeline_v3.sh) AND the CV search
#   then:      enable cv_rare_de, encode only the new entries, split again
#   then:      training from step2500 of run 2, only if parity shows 0 failures
#
#   nohup scripts/run.sh tools/cv_seltene_woerter.py > "$LOG_DIR/cv-ingest.log" 2>&1 &
#   nohup recipes/pipeline_v3b.sh > "$LOG_DIR/pipeline-v3b.log" 2>&1 &
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export BREEZE_TRAINING_DIR="$PTBR_ARTIFACTS/training-v3"
echo "[v3b] $(date -u +%H:%M) waiting for the preparation and the CV search"
# Wait for completion MESSAGES, not for process names. pgrep -f found the shell
# that had created this script here: its command line contained the whole script
# text including the name of the preparation chain - and the chain waited half
# an hour for itself.
until grep -qE "^\[v3\] .*DONE - ready" "$LOG_DIR/pipeline-v3.log" 2>/dev/null \
      && grep -q "^\[DONE\] Common Voice" "$LOG_DIR/cv-ingest.log" 2>/dev/null; do
    grep -qE "^\[v3\] .*ERROR" "$LOG_DIR/pipeline-v3.log" 2>/dev/null && { echo "[v3b] preparation ERROR"; exit 1; }
    sleep 30
done
grep "^\[DONE\]" "$LOG_DIR/cv-ingest.log"

python3 - "$CORPORA_JSON" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
an = {"hui_de", "hifitts2_en", "thorsten_rare_de", "cml_rare_de", "cv_rare_de"}
for c in d: c["enabled"] = c["name"] in an
p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print("[v3b] enabled:", [c["name"] for c in d if c["enabled"]])
PY

echo "[v3b] $(date -u +%H:%M) encoding Common Voice"
"$RUN" de_lora/core/prepare_dataset.py process --device cuda || { echo "[v3b] process ERROR"; exit 1; }
echo "[v3b] $(date -u +%H:%M) finalize"
"$RUN" de_lora/core/prepare_dataset.py finalize --gold 8 --parity-device cuda || { echo "[v3b] finalize ERROR"; exit 1; }
if ! grep -q "parity FAILED for 0/" "$LOG_DIR/pipeline-v3b.log"; then
    echo "[v3b] parity check not clean, training does NOT start"; grep parity "$LOG_DIR/pipeline-v3b.log"; exit 1
fi
echo "[v3b] $(date -u +%H:%M) parity: 0 failures - training starts"

# Word re-weighting per clip instead of corpus weights (see train_lora.wort_ausgleich):
# words from Zipf 4.0, target 50 examples, cap 4.
exec "$RUN" de_lora/core/train_lora.py --run de-en-v3 --epochs 1 \
    --batch 8 --grad-acc 4 --lr 3e-5 --ref-edit-frac 0.9 --val-items 96 \
    --sample-every-steps 500 \
    --wort-ausgleich "$WORDS_DIR/seltene_woerter.json" \
    --wa-ziel 50 --wa-max 4 --wa-min-zipf 4.0 \
    --resume-adapter "$PTBR_ARTIFACTS/training-v2/runs/de-en-v2/checkpoints/step2500"
