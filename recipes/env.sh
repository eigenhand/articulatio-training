# Shared settings for the run recipes in this folder. Sourced by every recipe;
# do not execute it directly.
#
# The Python side resolves its paths in de_lora/core/paths.py (environment >
# <repo>/.env > repository-local defaults). The values below are taken from
# there, so that shell and Python always agree.
#
# Environment variables used by the recipes (all optional):
#   BREEZE_ROOT     repository root (default: the parent of recipes/)
#   LOG_DIR         where the recipes read and write their logs
#                   (default: <PTBR_ARTIFACTS>/logs); the recipes coordinate
#                   through these log files, see the header of each recipe
#   CPU_PIN         core list for side jobs that must not disturb a running
#                   training (taskset -c), e.g. CPU_PIN=10-13; empty = no pinning
#   ADAPTER_BYTES   size of a complete adapter_model.safetensors; the run-5
#                   recipe waits until the file has this size (default: the
#                   r=64 / all-targets adapter of these runs, 148.3 M fp32 params)
#   BREEZE_CPP_DIR  companion C++ engine, needed by scripts/deploy_checkpoint.sh
#   TTS_URL         optional HTTP endpoint for the listening test in pipeline_v5.sh
#   TEST_VOICES     space-separated voice ids for that test (default: thorsten)
#   LISTEN_DIR      where the listening-test WAVs go (default: <PTBR_ARTIFACTS>/listening)

BREEZE_ROOT="${BREEZE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN="$BREEZE_ROOT/scripts/run.sh"

# Ask de_lora/core/paths.py for a resolved location, e.g. `_paths ARTIFACTS`.
_paths() { "$RUN" -c "import sys; sys.path.insert(0, 'de_lora/core'); import paths; print(paths.$1)"; }

PTBR_ARTIFACTS="$(_paths ARTIFACTS)"
export PTBR_ARTIFACTS
CORPORA_JSON="$(_paths CORPORA_JSON)"
WORDS_DIR="$(_paths WORDS_DIR)"
LOG_DIR="${LOG_DIR:-$PTBR_ARTIFACTS/logs}"
LISTEN_DIR="${LISTEN_DIR:-$PTBR_ARTIFACTS/listening}"
ADAPTER_BYTES="${ADAPTER_BYTES:-593168056}"
TEST_VOICES="${TEST_VOICES:-thorsten}"
mkdir -p "$LOG_DIR"

# Optional CPU pinning, used as a command prefix: $PIN some-command ...
PIN=${CPU_PIN:+taskset -c $CPU_PIN}
