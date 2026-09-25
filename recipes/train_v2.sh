#!/bin/bash
# Training of run 2: HUI (de, ~113 h) + HiFiTTS-2 (en, ~55 h), continued from
# step4500 of run 1. Separate training directory, see pipeline_v2.sh.
#
# Starts only when pipeline_v2.sh has finished AND the parity check reports no
# failure. The check verifies that the training prefix matches the official
# inference path token for token - without it you might train for 20 hours on
# a wrongly assembled format.
#
#   nohup recipes/train_v2.sh > "$LOG_DIR/train-v2.log" 2>&1 &
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export BREEZE_TRAINING_DIR="$PTBR_ARTIFACTS/training-v2"

# Only count lines of the pipeline itself. Previously the bare word "FERTIG"
# ("DONE") was enough - and that also appears in the ingestion logs, which
# pipeline_v2.sh copies into its own log at the start via tail
# ("[hui] DONE: 56580 clips"). The watcher therefore took the pipeline for
# finished before anything was encoded.
until grep -qE "^\[v2\] .*(DONE - ready|ERROR)" "$LOG_DIR/pipeline-v2.log" 2>/dev/null; do sleep 60; done
if grep -qE "^\[v2\] .*ERROR" "$LOG_DIR/pipeline-v2.log"; then
    echo "[train-v2] pipeline ended with an error, training does NOT start"; exit 1
fi
if ! grep -q "parity FAILED for 0/" "$LOG_DIR/pipeline-v2.log"; then
    echo "[train-v2] parity check not clean, training does NOT start:"
    grep "parity" "$LOG_DIR/pipeline-v2.log"; exit 1
fi
echo "[train-v2] $(date -u +%H:%M) parity: 0 failures - training starts"

# Run 1 (de-r64-e2) lives in the default training directory <artifacts>/training.
exec "$RUN" de_lora/core/train_lora.py --run de-en-v2 --epochs 2 \
    --batch 8 --grad-acc 4 --lr 3e-5 --ref-edit-frac 0.9 --val-items 96 \
    --sample-every-steps 500 \
    --resume-adapter "$PTBR_ARTIFACTS/training/runs/de-r64-e2/checkpoints/step4500"
