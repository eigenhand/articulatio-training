#!/bin/bash
# Run 2: HUI (de) + HiFiTTS-2 (en), continued from step4500 of run 1.
# This part prepares the data; recipes/train_v2.sh starts the training.
#
# Separate training directory, so that nothing from the first run leaks in:
# the code store and dataset_meta.jsonl in <artifacts>/training/ still contain
# all 103,734 CML entries, and finalize reads the whole meta file. With a shared
# directory CML would be back in training via the splits.
#
# Chain: wait for the end of the ingestion -> encode -> finalize.
# Started with nohup, so it survives the end of the terminal session:
#
#   nohup scripts/run.sh de_lora/data/build_corpus_hui.py --hours 200 --min-snr 20 > "$LOG_DIR/hui-ingest.log" 2>&1 &
#   nohup scripts/run.sh de_lora/data/build_corpus_hifitts2.py --hours 200 > "$LOG_DIR/en-ingest.log" 2>&1 &
#   nohup recipes/pipeline_v2.sh > "$LOG_DIR/pipeline-v2.log" 2>&1 &
#   nohup recipes/train_v2.sh    > "$LOG_DIR/train-v2.log" 2>&1 &
#
# (Both builders register their corpus as enabled in corpora.json; disable
# cml_de there before running this, so that run 2 trains on HUI + HiFiTTS-2 only.)
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export BREEZE_TRAINING_DIR="$PTBR_ARTIFACTS/training-v2"
mkdir -p "$BREEZE_TRAINING_DIR"

echo "[v2] $(date -u +%H:%M) waiting for the ingestion"
while pgrep -f "[b]uild_corpus_hui.py" >/dev/null || pgrep -f "[b]uild_corpus_hifitts2.py" >/dev/null; do
    sleep 30
done
echo "[v2] $(date -u +%H:%M) ingestion finished"
tail -3 "$LOG_DIR/hui-ingest.log"; tail -3 "$LOG_DIR/en-ingest.log"

echo "[v2] $(date -u +%H:%M) encoding"
"$RUN" de_lora/core/prepare_dataset.py process --device cuda || { echo "[v2] process ERROR"; exit 1; }

echo "[v2] $(date -u +%H:%M) finalize"
"$RUN" de_lora/core/prepare_dataset.py finalize --gold 8 --parity-device cuda || { echo "[v2] finalize ERROR"; exit 1; }

# NOTE: recipes/train_v2.sh waits for this line ("DONE - ready") or "ERROR".
echo "[v2] $(date -u +%H:%M) DONE - ready for training"
