# Training

The trainer applies **LoRA (PEFT)** to the frozen Breeze TTS 2. The audio codec
is always frozen (gate G1 asserts that nothing outside the backbone, depth
decoder and text encoder is trainable). Adapters are saved under
`<training>/runs/<run>/checkpoints/<tag>/`.

## Smoke test

Checks the loader, labels, VRAM and saving in a few steps:

```bash
scripts/run.sh de_lora/core/train_lora.py --run smoke --smoke --steps 30
```

## Full training

The first German run (new adapter, CML-TTS German):

```bash
scripts/run.sh de_lora/core/train_lora.py --run de-r64-e2 --epochs 2 \
  --rank 64 --alpha 64 --targets all --use-rslora \
  --batch 8 --grad-acc 4 --lr 3e-5 --ref-edit-frac 0.9 --val-items 96
```

Runs 2–5 continued from an earlier checkpoint with `--resume-adapter`; see
`recipes/` and the run table in the top-level README. When resuming, the LoRA
shape (rank, alpha, targets, rsLoRA) comes from the adapter and
`--rank/--alpha/--targets/--use-rslora` are ignored.

### Options

| Flag | Default | Description |
|---|---|---|
| `--run` | (required) | run name (folder in `training/runs/`) |
| `--smoke` / `--steps N` | — / 30 | short validation mode |
| `--epochs` | 3 | epochs |
| `--batch` / `--grad-acc` | 4 / 8 (`2/2` in smoke mode) | effective batch = batch × grad-acc |
| `--lr` | 2e-4 | learning rate (AdamW, warmup + cosine) |
| `--warmup` | 50 | warmup steps |
| `--rank` / `--alpha` | 16 / 32 | LoRA rank and alpha |
| `--targets` | `attn` | `attn` = q/k/v/o · `all` = + gate/up/down (MLPs) |
| `--use-rslora` | off | rank-stabilized scaling (`α/√r`) |
| `--ref-edit-frac` | 0.9 | fraction of examples with a speaker reference |
| `--val-items` | 96 | items of the stratified validation during training |
| `--corpus-weights` | template weights | JSON `{corpus: weight}`; corpora not listed get 1.0 |
| `--wort-ausgleich PATH` | off | rare-word re-weighting with `seltene_woerter.json` (see below) |
| `--wa-ziel` / `--wa-max` / `--wa-min-zipf` | 50 / 8 / 4.0 | target examples per rare word / cap per clip / frequency floor |
| `--rausch-codes DIR` | off | code store of the degraded references (`codes_rauschen`) |
| `--rausch-anteil` / `--rausch-sauber` | 0.35 / 0.25 | share with degraded reference + clean instruction / share with the clean instruction only |
| `--rausch-korpora` | `hui_de,thorsten_rare_de,hifitts2_en` | clean corpora that get degraded references |
| `--no-tensorboard` / `--no-wer` | off | disable TensorBoard / WER per checkpoint |
| `--sample-every-steps` | 500 | checkpoint + audio samples every N optimizer steps |
| `--resume-adapter` | — | adapter folder to continue training from |

(The German flag names are kept for compatibility with the recorded runs:
"Wortausgleich" = word balancing, "Ziel" = target, "Rausch" = noise,
"Anteil" = share, "sauber" = clean, "Korpora" = corpora.)

### Sampler weighting

Every epoch draws `len(train)` items with replacement from a
`WeightedRandomSampler`. The weight of a clip is its corpus weight
(`--corpus-weights`), multiplied by the rare-word factor if `--wort-ausgleich` is
set: a word that occurs `c` times in the training split gives the factor
`wa-ziel / c`, clamped to `[1, wa-max]`, and a clip takes the factor of its
rarest word. Only words with Zipf ≥ `--wa-min-zipf` count - with all 14,500
rare words, 37 % of the clips contained one and 78 % of the draws went to
"rare" clips. This is duplication in expectation, without duplicate files and
without leaking duplicates into val/test.

### Noise-augmented references

With `--rausch-codes`, reference-conditioned examples (`ref_edit_auto`) of the
clean corpora are assigned deterministically (hash of the idx) to one of three
groups: degraded reference + the instruction
*"Saubere Studioaufnahme ohne Hintergrundgeraeusche, klar und natuerlich
gesprochen."* (English corpora: *"Clean studio recording without background
noise, spoken clearly and naturally."*), clean reference + that instruction, or
unchanged. The target audio always stays clean, so the instruction learns to
mean "clean output" regardless of the reference. Validation never uses the
augmentation.

### Run outputs

```
training/runs/<run>/
├─ config.json              # effective hyperparameters + trainable parameters
├─ log.csv                  # step, losses, val_loss, lr, VRAM, s/step
├─ trainable_modules.txt
├─ checkpoints/<tag>/       # LoRA adapter (adapter_config.json + adapter_model.safetensors)
├─ samples/                 # validation WAVs per checkpoint + wer.json
└─ tb/                      # TensorBoard  (tensorboard --logdir training/runs/<run>/tb)
```

Checkpoints are written every `--sample-every-steps` steps (`step<N>`), after
every epoch (`epoch<E>_val<loss>`) and at the end (`final`).

### Training variants

- `tts_instruction`: `[instruction] + [text] + [target audio]` (no reference).
- `ref_edit_auto` / `ref_edit_tata`: `[ref text, ref audio, text] + [target audio]`
  (cloning with a reference from the same speaker when available).
- The share with a reference is controlled by `--ref-edit-frac`.
- The instruction language follows the corpus suffix (`_en` → English, otherwise German).

## Chained runs (`auto_train.py`)

Launches `train_lora.py` in rounds with a decaying LR and stops on a plateau or crash
(inherited from the template, not used for the German runs):

```bash
scripts/run.sh de_lora/train/auto_train.py \
  --resume "<training>/runs/r64_02/checkpoints/step4000" \
  --max-rounds 6 --epochs 2 --lr 1e-4 --decay 0.5 --epsilon 0.002 \
  --rank 64 --alpha 64 --targets all --use-rslora
```

To stop between rounds, create `<training>/AUTO_STOP`.

## Tips

- VRAM: batch 8 × grad-acc 4 with gradient checkpointing (always on) peaked at
  ~15 GB on the R9700; the template used batch 4 × grad-acc 8 on 16 GB cards.
  Watch `vram_peak_gb` in `log.csv`.
- The loss has two branches (`backbone_loss` + `depth_loss`, λ=1); both should go down.
- The target text is **context** (label `-100`); only the audio tokens are trained.
- After the final checkpoint the process can hang in the WER evaluation; the
  adapter is already written by then (`recipes/pipeline_v5.sh` handles this).
