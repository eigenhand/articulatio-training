#!/bin/bash
# Entry point for everything in this project: sets up the ROCm/MIOpen
# environment, changes into the repository root and runs Python with the
# given arguments, e.g.
#
#   scripts/run.sh de_lora/core/prepare_dataset.py process --device cuda
#   scripts/run.sh - some args <<'PY'   (program on stdin)
#
# MIOPEN_FIND_MODE=2 (FAST) is not optional here. The audio encoder, and later
# the training step, see a different input length for almost every clip. In
# the default mode MIOpen searches the kernels anew for every new shape:
#
#   same length, warm              28 ms
#   varying length, default      2270 ms
#   varying length, FAST           92 ms
#
# Measured on 2026-09-21 on gfx1201 with the Qwen3TTSTokenizerV2 encoder.
# Without this line, preprocessing 200 h takes about 42 hours instead of just
# over 3. (On CUDA systems the MIOpen variables are simply ignored.)
export MIOPEN_FIND_MODE=2

# Repository root: BREEZE_ROOT, or the parent directory of this script.
BREEZE_ROOT="${BREEZE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# MIOpen kernel database inside the project instead of the home directory
# (on the original machine the home directory shares a small disk with /).
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$BREEZE_ROOT/.miopen}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

# Interpreter: BREEZE_PY, else <repo>/.venv/bin/python, else python3 from PATH.
PY="${BREEZE_PY:-}"
if [ -z "$PY" ]; then
    if [ -x "$BREEZE_ROOT/.venv/bin/python" ]; then PY="$BREEZE_ROOT/.venv/bin/python"; else PY=python3; fi
fi

cd "$BREEZE_ROOT"
exec "$PY" "$@"
