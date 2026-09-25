# Installation

## Requirements

| Item | Used for the German runs / minimum |
|---|---|
| OS | Linux (the pt-BR template was developed on Windows) |
| Python | 3.12 (the template supports 3.10 – 3.12) |
| GPU | AMD Radeon AI PRO R9700 (gfx1201, 32 GB) with ROCm; ~15 GB peak VRAM with batch 8. NVIDIA/CUDA works too (the template trained on a 16 GB RTX 4060 Ti). |
| Disk | several hundred GB for raw corpora, processed WAVs and codes, plus ~8 GB for the base model |
| System tools | `ffmpeg` (with libmp3lame and libopus), `espeak-ng`, `pigz`, `libsndfile` |

## Steps

```bash
git clone <this repository> breeze-de-lora-training
cd breeze-de-lora-training

python3.12 -m venv .venv          # scripts/run.sh uses .venv/bin/python automatically
source .venv/bin/activate
pip install --upgrade pip

# 1) PyTorch for your platform first, e.g. the ROCm build used for the German runs:
pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/rocm6.4
#    (CUDA: use the matching index from pytorch.org instead)

# 2) the rest
pip install -r requirements.txt
```

- `requirements.txt` lists the training dependencies derived from the imports
  (engine runtime, PEFT, data tools, evaluation). Optional packages for the pt-BR
  podcast scraping pipeline are commented out at the end.
- `requirements-engine.txt` is the upstream engine's pinned list (inference API,
  Docker image, engine unit tests).

## ROCm notes

Always start Python through `scripts/run.sh` (or export the same variables):
it sets `MIOPEN_FIND_MODE=2` and a project-local MIOpen kernel database. Without
FIND_MODE=2, MIOpen searches kernels again for every new input length, which
makes the variable-length audio encoding about 25x slower (see the top-level
README).

## Base checkpoint

The base model is **not** part of the repository. Download `BreezeBlue/Breeze-TTS-2`
from Hugging Face into `<PTBR_ARTIFACTS>/models/Breeze-TTS-2`:

```bash
python -c "from huggingface_hub import snapshot_download as s; \
s('BreezeBlue/Breeze-TTS-2', local_dir='<PTBR_ARTIFACTS>/models/Breeze-TTS-2')"
```

If the repository is *gated*, accept the terms and set `HF_TOKEN`. The weights
are licensed under the BreezeBlue Research and Non-Commercial License Agreement.

## Checking the installation

```bash
# byte-compile the package (fast, no GPU)
python -m compileall -q de_lora tools scripts

# read-only introspection of the base model (phase A)
scripts/run.sh de_lora/tools/analyze_model.py
```

## Next step

Configure the paths in [CONFIGURATION.md](CONFIGURATION.md) and prepare the data
as described in [DATASETS.md](DATASETS.md).
