#!/bin/bash
# Turn a LoRA checkpoint into a quantized GGUF for the companion C++ engine
# (Breeze-TTS-2.cpp) and optionally switch a running server over to it.
#
#   scripts/deploy_checkpoint.sh step4500 [--activate]           (name inside RUN_DIR; run 1)
#   scripts/deploy_checkpoint.sh /path/to/checkpoints/step2500 --activate
#
# Chain: merge (CPU) -> GGUF f16 -> Q8_0 -> optionally repoint the "active" symlink.
# Without --activate the model is only built; switching stays a deliberate
# decision because it restarts the server.
#
# The merge deliberately runs on the CPU: the GPU may still be busy with a
# training run, and a second 3B model next to it would compete for memory and
# compute. With CPU_PIN (e.g. CPU_PIN=10-13) all steps are additionally pinned
# to cores the training does not use.
#
# Configuration (environment):
#   BREEZE_CPP_DIR  checkout of the companion C++ engine (required); must contain
#                   scripts/convert_hf_to_gguf.py and build/breeze-quantize
#   RUN_DIR         run folder for checkpoint names (default: <training>/runs/de-r64-e2)
#   GGUF_DIR        output folder for the .gguf files (default: <PTBR_ARTIFACTS>/gguf)
#   TAG             file name tag (default: <run>-<checkpoint>)
#   ACTIVE_NAME     symlink inside GGUF_DIR that --activate repoints
#                   (default: breeze-tts-2-active.gguf); point your server at it
#   RESTART_CMD     optional command run after --activate to reload the server,
#                   e.g. RESTART_CMD="sudo systemctl restart my-tts-server"
#   HEALTH_URL      optional URL polled after the restart until it answers
#   CPU_PIN         optional taskset core list
set -euo pipefail

CKPT_ARG="${1:?missing checkpoint: a name (e.g. step4500) or a full path}"
ACTIVATE="${2:-}"
: "${BREEZE_CPP_DIR:?set BREEZE_CPP_DIR to your checkout of the companion C++ engine (Breeze-TTS-2.cpp)}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/scripts/run.sh"
PIN=${CPU_PIN:+taskset -c $CPU_PIN}
# Resolve locations exactly like de_lora/core/paths.py (environment > .env > defaults).
_paths() { "$RUN" -c "import sys; sys.path.insert(0, 'de_lora/core'); import paths; print(paths.$1)"; }
ARTIFACTS="$(_paths ARTIFACTS)"
TRAINING="$(_paths TRAINING)"

# Previously hard-wired to run 1. Now: a full path, or a name inside RUN_DIR.
RUN_DIR="${RUN_DIR:-$TRAINING/runs/de-r64-e2}"
if [ -d "$CKPT_ARG" ]; then CKPT="$(readlink -f "$CKPT_ARG")"; else CKPT="$RUN_DIR/checkpoints/$CKPT_ARG"; fi
CKPT_NAME="$(basename "$CKPT")"
RUN_NAME="$(basename "$(dirname "$(dirname "$CKPT")")")"
# Run name in the file name, otherwise step2500 of run 1 and of run 2 could not
# be told apart.
TAG="${TAG:-$RUN_NAME-$CKPT_NAME}"
HF=$ARTIFACTS/models/Breeze-TTS-2-$TAG
GGUF_DIR="${GGUF_DIR:-$ARTIFACTS/gguf}"
ACTIVE_NAME="${ACTIVE_NAME:-breeze-tts-2-active.gguf}"
F16=$GGUF_DIR/breeze-tts-2-$TAG-f16.gguf
Q8=$GGUF_DIR/breeze-tts-2-$TAG-q8_0.gguf

[ -f "$CKPT/adapter_model.safetensors" ] || { echo "no adapter in $CKPT"; exit 1; }
mkdir -p "$GGUF_DIR"

echo "== 1/4 merge: $CKPT_NAME =="
$PIN "$RUN" "$ROOT/scripts/merge_lora.py" "$CKPT" "$HF"

echo "== 2/4 convert to GGUF f16 =="
$PIN "$RUN" \
  "$BREEZE_CPP_DIR/scripts/convert_hf_to_gguf.py" "$HF" -o "$F16" --dtype f16

echo "== 3/4 quantize to Q8_0 =="
$PIN "$BREEZE_CPP_DIR/build/breeze-quantize" "$F16" "$Q8" q8_0 > /dev/null

echo "== 4/4 clean up =="
# f16 and the HF folder are not needed after quantizing; together they are
# 13 GB per checkpoint. The Q8_0 file stays.
rm -f "$F16"
rm -rf "$HF"
ls -la "$Q8" | awk '{printf "  %s  %d MB\n",$9,$5/1024/1024}'

if [ "$ACTIVATE" = "--activate" ]; then
    ln -sfn "$(basename "$Q8")" "$GGUF_DIR/$ACTIVE_NAME"
    echo "active: $(basename "$(readlink -f "$GGUF_DIR/$ACTIVE_NAME")")"
    if [ -n "${RESTART_CMD:-}" ]; then
        eval "$RESTART_CMD"
        if [ -n "${HEALTH_URL:-}" ]; then
            until curl -sf -o /dev/null "$HEALTH_URL" 2>/dev/null; do sleep 2; done
        fi
        echo "server up"
    else
        echo "symlink updated; restart your server to load it (or set RESTART_CMD)"
    fi
else
    echo "built, but not activated. To switch:"
    echo "  ln -sfn $(basename "$Q8") $GGUF_DIR/$ACTIVE_NAME   # then restart your server"
fi
