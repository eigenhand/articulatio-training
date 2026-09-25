# Documentation — German LoRA training for Breeze TTS 2

User-facing documentation: how to install, configure and run the training and
evaluation pipeline of the German (plus English) LoRA adapter on top of
Breeze TTS 2. The overview, the run history and the data licences are in the
top-level [`../README.md`](../README.md).

| Document | Content |
|---|---|
| [INSTALL.md](INSTALL.md) | environment, dependencies, base checkpoint |
| [CONFIGURATION.md](CONFIGURATION.md) | `PTBR_ARTIFACTS` and all environment variables |
| [DATASETS.md](DATASETS.md) | source corpora, corpus builders, rare-word and Common Voice tools, `prepare_dataset.py` |
| [TRAINING.md](TRAINING.md) | `train_lora.py`, its options, `auto_train.py` |
| [EVALUATION.md](EVALUATION.md) | val loss, WER/CER, speaker similarity |

## Typical flow

```
[INSTALL] → [CONFIGURATION] → [DATASETS: build corpora, process + finalize]
          → [TRAINING: smoke + full] → [EVALUATION] → adapter in training/runs/<run>/checkpoints/
          → scripts/merge_lora.py / scripts/deploy_checkpoint.sh (GGUF for the C++ engine)
```

The adapter (a folder with `adapter_config.json` + `adapter_model.safetensors`)
is a PEFT LoRA for `models.breeze.BreezeForConditionalGeneration`.

> **License:** the Breeze TTS 2 weights and all derivatives (including adapters
> trained with this code) are *research/non-commercial*
> (https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE). See `../NOTICE`.
