# Upstream

This repository is a fork of
[`EdnilsonMonteiro/breeze-tts2-ptbr-lora-training`](https://github.com/EdnilsonMonteiro/breeze-tts2-ptbr-lora-training)
(the pt-BR template), which itself is a fork of
[`breezeblue-ai/breeze-tts`](https://github.com/breezeblue-ai/breeze-tts) that
adds LoRA training/evaluation code.

| Field | Value |
|---|---|
| Template (direct upstream) | `https://github.com/EdnilsonMonteiro/breeze-tts2-ptbr-lora-training.git` |
| Template commit | `6c414bdbab4d0be5f6953f66ebc81fed8dbf17ee` (2026-09-17) |
| Engine upstream | `https://github.com/breezeblue-ai/breeze-tts.git` |
| Engine base commit | `008f769016b0a24711becd7a4925030bc93f608c` (`main`) |
| German fork | `de_lora/` (renamed from `ptbr_lora/`), `tools/`, `recipes/`, `scripts/` |

The engine (Apache-2.0 code: `breeze_infer/`, `models/`, `infer.py`, `configs/`,
`docker/`, `tests/`) sits at the repository root and is **unchanged**, except
that `docker/Dockerfile` installs `requirements-engine.txt` (the engine's
original `requirements.txt`, renamed). The project code lives in `de_lora/`
and the directories listed above. The weights (Breeze-TTS-2) are **not** part of
the repository — download them from the official source (see `README.md`). The
model license (`BreezeBlue Research and Non-Commercial`) is at
https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE (upstream
removed the `MODEL_LICENSE` file from the code repository).

## Syncing with upstream

```bash
git remote add upstream https://github.com/breezeblue-ai/breeze-tts.git
git fetch upstream
git rebase upstream/main        # or merge, as you prefer
```

When rebasing, check `de_lora/core/common_breeze.py`: it depends on internal
engine APIs (`breeze_infer.templates._prepare_one`, `_encode_prompt_audio`,
`_tts_instruction_segments`, `_ref_edit_tata_segments`) that may change between
versions. The parity check of `prepare_dataset.py finalize` will report such a
change as failures.
