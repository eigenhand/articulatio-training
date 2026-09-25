#!/bin/bash
# Run 5: a second pass over the data of run 4, with noise-augmented references.
#   waits for: the final checkpoint of run 4 (checkpoints/final) and the degraded WAVs
#   then:      stop the run-4 process, deploy and test the final checkpoint (CPU and
#              a second GPU in the original setup), at the same time encode the
#              degraded references (training GPU), then run 5
#
#   BREEZE_TRAINING_DIR=<artifacts>/training-v3 \
#     nohup scripts/run.sh tools/rauschvorlagen.py audio > "$LOG_DIR/rausch-audio.log" 2>&1 &
#   nohup recipes/pipeline_v5.sh > "$LOG_DIR/pipeline-v5.log" 2>&1 &
#
# (`rauschvorlagen.py audio` needs the run-3/4 splits, hence the training-v3 directory.)
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export BREEZE_TRAINING_DIR="$PTBR_ARTIFACTS/training-v3"
R4=$BREEZE_TRAINING_DIR/runs/de-en-v4
FIN=$R4/checkpoints/final
echo "[v5] $(date -u +%H:%M) waiting for the final checkpoint of run 4"
until [ "$(stat -c %s $FIN/adapter_model.safetensors 2>/dev/null || echo 0)" = "$ADAPTER_BYTES" ] \
      && [ -f $FIN/adapter_config.json ]; do
    grep -q "Traceback" "$LOG_DIR/pipeline-v4.log" && { echo "[v5] run 4 ERROR"; exit 1; }
    sleep 30
done
sleep 20
echo "[v5] $(date -u +%H:%M) final checkpoint present: $(ls $R4/checkpoints | grep -E '^epoch' | tr '\n' ' ')"

# After the final checkpoint the trainer tends to hang in the WER evaluation -
# the checkpoint has long been written by then. Wait up to 20 min, then stop it.
# Find the process by its exact command line, never with pgrep -f on a pattern.
# (python[0-9.]*: the interpreter may be .venv/bin/python or python3, see scripts/run.sh.)
for i in $(seq 1 40); do
    P=$(ps -eo pid,args | awk '$2 ~ /python[0-9.]*$/ && /train_lora\.py/ && /--run de-en-v4 / {print $1}')
    [ -z "$P" ] && break
    sleep 30
done
P=$(ps -eo pid,args | awk '$2 ~ /python[0-9.]*$/ && /train_lora\.py/ && /--run de-en-v4 / {print $1}')
[ -n "$P" ] && { echo "[v5] run-4 trainer hangs after the final checkpoint, stopping $P"; kill $P; sleep 15; }

# Deploy and test the final checkpoint - runs on the CPU (and a server on another
# GPU), in parallel to the encoding below. Both parts are optional: the GGUF
# build needs BREEZE_CPP_DIR (scripts/deploy_checkpoint.sh), the listening test
# needs TTS_URL - an HTTP endpoint that takes the form fields voice_id and text
# and returns raw 24 kHz 16-bit mono PCM. Voices: TEST_VOICES (default thorsten).
(
  if [ -n "${BREEZE_CPP_DIR:-}" ]; then
    "$BREEZE_ROOT/scripts/deploy_checkpoint.sh" "$(readlink -f $FIN)" --activate > "$LOG_DIR/deploy-v4-final.log" 2>&1
    echo "exit $?" >> "$LOG_DIR/deploy-v4-final.log"
  fi
  if [ -n "${TTS_URL:-}" ]; then
    OUT="$LISTEN_DIR/v4-final"; mkdir -p "$OUT"; rm -f "$OUT"/*.wav; i=0
    while IFS= read -r t; do i=$((i+1)); for v in $TEST_VOICES; do
      # retry: a busy server answers e.g. 409 - wait instead of losing the sample
      for versuch in $(seq 1 30); do
        code=$(curl -s -o "$OUT/x.pcm" -w "%{http_code}" --form-string "voice_id=$v" --form-string "text=$t" "$TTS_URL")
        [ "$code" = 200 ] && break; sleep 4
      done
      ffmpeg -loglevel error -y -f s16le -ar 24000 -ac 1 -i "$OUT/x.pcm" "$OUT/$(printf %02d $i)_$v.wav"; rm -f "$OUT/x.pcm"
    done; done <<'EOF'
Die Temperaturen steigen heute auf 20 Grad.
Bitte ruf mich später auf dem Telefon an.
Weitere Informationen zum Projekt finden Sie online.
Die Straßenbahnhaltestelle vor dem Hauptbahnhof wird umgebaut.
Am Bahnhof wartet schon der Zug.
The weather today is sunny with a gentle breeze from the east.
EOF
    # ASR check with faster-whisper on the CPU; sentence 06 is the English probe.
    nice -n 5 $PIN "$RUN" - "$OUT" <<'PY' > "$LOG_DIR/v4-final-test.log" 2>&1
import glob, sys
from faster_whisper import WhisperModel
m = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=4)
for f in sorted(glob.glob(f"{sys.argv[1]}/*.wav")):
    lang = "en" if "/06_" in f else "de"
    segs, _ = m.transcribe(f, language=lang, beam_size=5)
    print(f"  {f.rsplit('/',1)[1]:<28} {' '.join(s.text for s in segs).strip()}")
PY
  fi
  echo "[v5] $(date -u +%H:%M) final checkpoint deployed and tested" >> "$LOG_DIR/deploy-v4-final.log"
) &

echo "[v5] $(date -u +%H:%M) waiting for the degraded WAVs"
until grep -q "^\[DONE\] audio" "$LOG_DIR/rausch-audio.log" 2>/dev/null; do
    grep -q "Traceback" "$LOG_DIR/rausch-audio.log" 2>/dev/null && { echo "[v5] noisy references ERROR"; exit 1; }
    sleep 30
done
grep "^\[DONE\]" "$LOG_DIR/rausch-audio.log"
echo "[v5] $(date -u +%H:%M) encoding the noisy references"
"$RUN" tools/rauschvorlagen.py kodieren --device cuda || { echo "[v5] kodieren ERROR"; exit 1; }
echo "[v5] $(date -u +%H:%M) training starts"

# Second pass: same data and weights as run 4, slightly lower learning rate
# (2e-5 instead of 3e-5), so that the fine-tuning from the end of the first
# pass is not overwritten right away.
exec "$RUN" de_lora/core/train_lora.py --run de-en-v5 --epochs 1 \
    --batch 8 --grad-acc 4 --lr 2e-5 --ref-edit-frac 0.9 --val-items 96 \
    --sample-every-steps 500 \
    --corpus-weights '{"hui_de": 1.5, "thorsten_rare_de": 1.5, "hifitts2_en": 2.0, "cv_breit_de": 0.6}' \
    --wort-ausgleich "$WORDS_DIR/seltene_woerter.json" \
    --wa-ziel 50 --wa-max 4 --wa-min-zipf 4.0 \
    --rausch-codes $BREEZE_TRAINING_DIR/codes_rauschen --rausch-anteil 0.35 --rausch-sauber 0.25 \
    --resume-adapter $FIN
