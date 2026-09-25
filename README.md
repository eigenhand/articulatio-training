# Articulatio-DE: German LoRA fine-tuning for Breeze TTS 2

Training code and run recipes for **Articulatio-DE**, a German (plus English) LoRA
adapter on top of [Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2), BreezeBlue's
codec language model for text-to-speech. Beyond plain fine-tuning it covers:

- **rare-word coverage** — finding everyday words the model has hardly ever
  heard (word frequency, tokenizer coverage, stress position) and mining clips
  that contain them from Thorsten-Voice, CML-TTS and Common Voice;
- **Common Voice selection** — streaming the 37 GB archive once, with quality
  filters (SNR, mains hum, band edge) and a per-speaker cap;
- **sampler weighting** — corpus weights plus a per-clip re-weighting by the
  rarest word of each transcript;
- **noise-augmented reference prompts** — synthetic degradations of the
  reference clip with a clean target and a "clean studio recording" instruction.

This repository is a fork of
[breeze-tts2-ptbr-lora-training](https://github.com/EdnilsonMonteiro/breeze-tts2-ptbr-lora-training)
(Brazilian Portuguese), which in turn forks the official engine
[breezeblue-ai/breeze-tts](https://github.com/breezeblue-ai/breeze-tts). See
[`UPSTREAM.md`](UPSTREAM.md) and [`NOTICE`](NOTICE).

> **License:** code Apache-2.0; model weights and adapters are research and
> non-commercial only (see [License](#license)). No weights, audio or datasets
> are included.

**Result:** on 130 held-out test sentences, cloned German speech reaches 4.2 % word error rate against
4.4 % for the real recordings, with the same speaker similarity (0.72 vs 0.715), and runs at RTF 0.49 on
an RTX 4070. The model is on Hugging Face as
[eigenhand/Articulatio-DE](https://huggingface.co/eigenhand/Articulatio-DE) (adapter),
[-GGUF](https://huggingface.co/eigenhand/Articulatio-DE-GGUF) and
[-MLX](https://huggingface.co/eigenhand/Articulatio-DE-MLX); method and all numbers in
[`docs/RESULTS.md`](docs/RESULTS.md).

## Demo

https://github.com/user-attachments/assets/aaee58ac-1228-48b6-b9f9-7ac4150b0338

13 s of German in a voice the model created itself, one sentence per piece, `speed` 1.25 with the pitch
preserved and 150 ms between sentences: *"Guten Morgen! Heute ist Donnerstag, der fünfundzwanzigste
September. Draußen sind es achtzehn Grad, am Nachmittag zieht von Westen ein Gewitter auf. Vergiss also
den Regenschirm nicht, wenn du später noch zum Bahnhof fährst."*

Synthetic speech generated with Articulatio-DE (GGUF Q8_0, articulatio.cpp); the voice belongs to no real
person. Derived from Breeze TTS 2 by BreezeBlue and licensed for research and non-commercial use only: the
clip is an output of the model and falls under the
[BreezeBlue Research and Non-Commercial License](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE),
not under the Apache 2.0 license of this repository.

## Repository layout

```
de_lora/        training package (renamed from the template's ptbr_lora/)
├─ core/        paths.py (configuration), common_breeze.py, codestore.py,
│               prepare_dataset.py (encode + finalize), train_lora.py
├─ data/        corpus builders: build_corpus_de.py (CML-TTS German),
│               build_corpus_hui.py, build_corpus_hifitts2.py; pt-BR builders of the template
├─ eval/        full val loss, WER/CER, speaker similarity
├─ tools/       model/corpus diagnostics (template)
├─ train/       auto_train.py (template)
└─ scraping/    pt-BR podcast pipeline (template, not used here)
tools/          German data tools: rare words, capitalization, Common Voice, noisy references
recipes/        the pipelines of runs 2-5 as documented shell scripts (+ env.sh)
scripts/        run.sh (entry point), merge_lora.py, deploy_checkpoint.sh
docs/           detailed documentation
breeze_infer/, models/, infer.py, configs/, docker/, tests/   upstream engine (unchanged)
```

## Pipeline

```
raw corpora ──► 1. build corpora ──► 2. encode to codec codes ──► 3. finalize splits
               (de_lora/data,         (prepare_dataset.py           (prepare_dataset.py finalize:
                tools/)                process: Qwen3 audio          90/5/5 per corpus + parity
                                       tokenizer, 16 codebooks)      check, must report 0 failures)
            ──► 4. train LoRA ──► 5. merge ──► 6. convert to GGUF
               (train_lora.py)    (scripts/     (scripts/deploy_checkpoint.sh with the
                                   merge_lora.py) companion C++ engine: f16 -> Q8_0)
```

1. **Build corpora.** Each builder writes `datasets/<corpus>/` (`wavs/`,
   `texts.csv` in the format `wavs/<file>.wav==text`, `speakers.jsonl`) and
   registers the corpus in `datasets/corpora.json`. See
   [`docs/DATASETS.md`](docs/DATASETS.md).
2. **Encode.** `prepare_dataset.py process` trims, resamples to 24 kHz,
   normalizes and encodes every enabled corpus once with the model's audio
   tokenizer; the codes go into one consolidated store (`codes/codes.i16` +
   `index.jsonl`) instead of one file per utterance.
3. **Finalize.** `prepare_dataset.py finalize` writes the splits and checks
   that the assembled training sequences match the official inference template
   token for token (`[finalize] parity FAILED for 0/16`). The recipes refuse to
   train otherwise.
4. **Train.** `train_lora.py` trains a PEFT LoRA on backbone, depth decoder and
   text encoder (the codec stays frozen), writes checkpoints, audio samples and
   a WER check per checkpoint. See [`docs/TRAINING.md`](docs/TRAINING.md).
5. **Merge.** `scripts/merge_lora.py` merges the adapter into the base weights
   on the CPU in float32 and saves an fp16 Hugging Face folder.
6. **Convert.** `scripts/deploy_checkpoint.sh` runs the merge, then the GGUF
   converter and quantizer of the companion C++ engine ([articulatio.cpp](https://github.com/eigenhand/articulatio.cpp), not
   part of this repository; set `BREEZE_CPP_DIR`), and can optionally switch a
   running server to the new file.

## Quick start

Training needs one GPU with at least 16 GB of VRAM (CUDA or ROCm); the runs here used an AMD Radeon AI
PRO R9700 at about 12–14 s per optimizer step.

```bash
# install: PyTorch for your platform first, then the rest
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/rocm6.4   # AMD
# pip install torch==2.9.1 torchaudio==2.9.1                                                    # NVIDIA (CUDA)
pip install -r requirements.txt

# configure (or put these into .env, see .env.example)
export PTBR_ARTIFACTS=/data/breeze-artifacts          # datasets/, training/, models/ go here
export CML_DE_ROOT=/data/raw/cml_tts_dataset_german_v0.1
# base model -> $PTBR_ARTIFACTS/models/Breeze-TTS-2 (see docs/INSTALL.md)

# run 1, abbreviated: build the corpus, encode, finalize, smoke test, train
scripts/run.sh de_lora/data/build_corpus_de.py --hours 200
scripts/run.sh de_lora/core/prepare_dataset.py process --device cuda
scripts/run.sh de_lora/core/prepare_dataset.py finalize --gold 8 --parity-device cuda
scripts/run.sh de_lora/core/train_lora.py --run smoke --smoke --steps 30
scripts/run.sh de_lora/core/train_lora.py --run de-r64-e2 --epochs 2 \
    --rank 64 --alpha 64 --targets all --use-rslora \
    --batch 8 --grad-acc 4 --lr 3e-5 --ref-edit-frac 0.9 --val-items 96

# merge + GGUF for the C++ engine
BREEZE_CPP_DIR=/path/to/articulatio.cpp scripts/deploy_checkpoint.sh final
```

Always start Python through `scripts/run.sh`; it sets the MIOpen variables
(see *Hardware notes*) and runs from the repository root.

## Run history

Six runs, each continuing the adapter of the previous one: CML-TTS German first, then English data,
rare-word clips, a broad Common Voice selection (463.5 hours in the final mix), a second pass
with noise-augmented references, and a short learning-rate decay. The last run (`de-en-v5-abklingen`,
German for *decay*) is the released model. Every run, its data and its settings, plus the recipe
scripts that ran them: [`docs/RUNS.md`](docs/RUNS.md).

## Data sources and licenses

No audio or derived corpora are distributed with this code; the tools read the
sources from local copies (locations configurable, see
[`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)).

| Source | Corpora (names in `corpora.json`) | License | Attribution |
|---|---|---|---|
| CML-TTS German (`cml_tts_dataset_german_v0.1`) | `cml_de`, `cml_rare_de` | CC BY 4.0 | required: credit CML-TTS (Oliveira et al., 2023), link the license, state that the audio was selected, trimmed, resampled and re-encoded |
| HiFiTTS-2 (`nvidia/hifitts-2` manifests; LibriVox audio from archive.org) | `hifitts2_en` | CC BY 4.0 | required: credit HiFiTTS-2 (NVIDIA), link the license, state the changes |
| HUI-Audio-Corpus-German (IISYS; used via the Hugging Face upload `Paradoxia/opendata-iisys-hui`) | `hui_de` | CC0 (Puchtler et al., 2021, Table 1; the `license: mit` tag of the Hugging Face upload does not match the original release) | not required (citing the paper appreciated) |
| Thorsten-Voice (`Thorsten-Voice/TV-44kHz-Full`, subsets TV-2021.02-Neutral and TV-2022.10-Neutral) | `thorsten_rare_de` | CC0 | not required (credit to Thorsten Müller / thorsten-voice.de appreciated) |
| Mozilla Common Voice 27.0 German | `cv_rare_de`, `cv_breit_de` | CC0 | not required; Common Voice's terms forbid attempts to identify the speakers |

## Hardware notes

- The German runs used one **AMD Radeon AI PRO R9700** (32 GB) with ROCm
  (PyTorch 2.9.1, ROCm 6.4 wheels): peak ~15 GB VRAM at batch 8 × grad-acc 4,
  ~12–14 s per optimizer step. On CUDA the code runs unchanged.
- **`MIOPEN_FIND_MODE=2` is essential on ROCm.** In the default mode MIOpen
  benchmarks kernels for every new tensor shape, and audio clips almost never
  repeat a length: encoding took 2270 ms per clip instead of 92 ms, i.e. about
  42 hours instead of 3 for 200 h of audio. `scripts/run.sh` sets it.

## Gotchas

- **Corpus names must end in `_de` or `_en`.** The number expansion
  (`num2words`) and the instruction language are chosen from that suffix; any
  other name falls back to the template's Portuguese number words and to German
  instructions for English text.
- **The trainer may hang in the WER evaluation after the final checkpoint.**
  The adapter is already written at that point. `recipes/pipeline_v5.sh` waits
  up to 20 minutes and then stops the process, found by its exact command line.
- **One training directory per data generation.** `finalize` reads the whole
  `dataset_meta.jsonl`, so previously encoded corpora would end up in the splits.
- **Resuming keeps the adapter shape.** With `--resume-adapter`, `--rank`,
  `--alpha`, `--targets` and `--use-rslora` are ignored.
- **German names in the code.** CLI flags, file names, corpus and run names
  are kept as they were used in the runs (e.g. `--wort-ausgleich` = rare-word
  re-weighting, `--rausch-*` = noise augmentation, `--trocken` = dry run,
  `seltene_woerter.json` = rare words, `rauschvorlagen` = noisy references,
  `cv_breit_de` = broad Common Voice selection, `abklingen` = learning-rate
  decay); the docstrings explain them in English.

## Configuration

All paths come from environment variables or `.env` (see
[`.env.example`](.env.example)), resolved in
[`de_lora/core/paths.py`](de_lora/core/paths.py); `PTBR_ARTIFACTS` (name
inherited from the template) is the root of datasets, training runs and models.
The full list is in [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md).

## Documentation

| Doc | Content |
|---|---|
| [`docs/INSTALL.md`](docs/INSTALL.md) | environment, dependencies, base checkpoint |
| [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) | all variables, artifact tree, corpus registry |
| [`docs/DATASETS.md`](docs/DATASETS.md) | source layouts, builders, rare words, Common Voice, noisy references, preparation |
| [`docs/TRAINING.md`](docs/TRAINING.md) | trainer options, sampler weighting, noise augmentation |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | val loss, WER/CER, speaker similarity |
| [`docs/RUNS.md`](docs/RUNS.md) | the six training runs: data, settings, recipes |
| [`docs/RESULTS.md`](docs/RESULTS.md) | release evaluation of the final model: method, findings, tables |

## License

- Code: Apache License 2.0 ([`LICENSE`](LICENSE)); see [`NOTICE`](NOTICE) for
  the attribution of the pt-BR template and the Breeze TTS 2 notice.
- Breeze TTS 2 weights and derivatives (including LoRA adapters trained with
  this code): BreezeBlue Research and Non-Commercial License Agreement
  (https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE). Not
  included here; commercial use requires a separate license. If you
  distribute an adapter, include the NOTICE and state: *"Derived from Breeze
  TTS 2 by BreezeBlue and licensed for research and non-commercial use only."*
- Datasets: see *Data sources and licenses*.

## Acknowledgements

Ednilson Monteiro for the pt-BR LoRA training template, BreezeBlue for Breeze
TTS 2 and its engine, and the creators of CML-TTS, HiFiTTS-2, the
HUI-Audio-Corpus-German, Thorsten-Voice and Mozilla Common Voice.
