#!/bin/bash
# Run 4: Common Voice in breadth (cv_breit_de, at most 100 clips per speaker)
# on top of the data of run 3, continued from step1000 of run 3.
#   waits for: [DONE] in cv-breit.log
#   then:      back up the splits of run 3, enable cv_breit_de, encode only the
#              new entries (same code store as run 3), split again
#   then:      training, only if parity shows 0 failures
#
#   nohup scripts/run.sh tools/cv_breit.py > "$LOG_DIR/cv-breit.log" 2>&1 &
#   nohup recipes/pipeline_v4.sh > "$LOG_DIR/pipeline-v4.log" 2>&1 &
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export BREEZE_TRAINING_DIR="$PTBR_ARTIFACTS/training-v3"
echo "[v4] $(date -u +%H:%M) waiting for cv_breit_de"
# wait for the completion message, never for process names (see pipeline_v3b.sh)
until grep -q "^\[DONE\] cv_breit_de" "$LOG_DIR/cv-breit.log" 2>/dev/null; do
    grep -q "Traceback" "$LOG_DIR/cv-breit.log" 2>/dev/null && { echo "[v4] cv_breit ERROR"; exit 1; }
    sleep 30
done
grep "^\[DONE\]" "$LOG_DIR/cv-breit.log"

# finalize rewrites the splits and the meta files - keep the ones of run 3
S=$BREEZE_TRAINING_DIR/runs/de-en-v3/daten_v3
mkdir -p "$S" && cp $BREEZE_TRAINING_DIR/splits_*.txt $BREEZE_TRAINING_DIR/dataset_meta.jsonl \
    $BREEZE_TRAINING_DIR/summary.json "$S"/ && echo "[v4] splits of run 3 backed up in $S"

python3 - "$CORPORA_JSON" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
an = {"hui_de", "hifitts2_en", "thorsten_rare_de", "cml_rare_de", "cv_rare_de", "cv_breit_de"}
for c in d: c["enabled"] = c["name"] in an
p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print("[v4] enabled:", [c["name"] for c in d if c["enabled"]])
PY

echo "[v4] $(date -u +%H:%M) encoding cv_breit_de"
"$RUN" de_lora/core/prepare_dataset.py process --device cuda || { echo "[v4] process ERROR"; exit 1; }
echo "[v4] $(date -u +%H:%M) finalize"
"$RUN" de_lora/core/prepare_dataset.py finalize --gold 8 --parity-device cuda || { echo "[v4] finalize ERROR"; exit 1; }
if ! grep -q "parity FAILED for 0/" "$LOG_DIR/pipeline-v4.log"; then
    echo "[v4] parity check not clean, training does NOT start"; grep parity "$LOG_DIR/pipeline-v4.log"; exit 1
fi
echo "[v4] $(date -u +%H:%M) parity: 0 failures - training starts"

# Mix: with weight 1, Common Voice would be about 70 % of the draws and would
# dominate the sound. With these weights roughly: CV 50 %, HUI 26 %,
# English 17 % (as in run 3), Thorsten 6 %, CML 1 %.
exec "$RUN" de_lora/core/train_lora.py --run de-en-v4 --epochs 1 \
    --batch 8 --grad-acc 4 --lr 3e-5 --ref-edit-frac 0.9 --val-items 96 \
    --sample-every-steps 500 \
    --corpus-weights '{"hui_de": 1.5, "thorsten_rare_de": 1.5, "hifitts2_en": 2.0, "cv_breit_de": 0.6}' \
    --wort-ausgleich "$WORDS_DIR/seltene_woerter.json" \
    --wa-ziel 50 --wa-max 4 --wa-min-zipf 4.0 \
    --resume-adapter "$BREEZE_TRAINING_DIR/runs/de-en-v3/checkpoints/step1000"
