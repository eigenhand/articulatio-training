# Evaluation

Three axes: **generalization** (val loss), **intelligibility** (WER/CER) and
**voice identity** (speaker similarity).

## Val loss on the complete split

Evaluates and ranks adapters with the same loss as in training (lower = better):

```bash
# base + adapters; <adapter> can be a path relative to the training directory or absolute
scripts/run.sh de_lora/eval/eval_val_full.py --adapters runs/de-en-v4/checkpoints/final --out results.json

# smoke (fast)
scripts/run.sh de_lora/eval/eval_val_full.py --limit 64 --adapters <adapter> --out results.json
```

Options: `--adapters` (list), `--batch` (default 4), `--limit`, `--out`.
The base model (no adapter) is always evaluated first as the reference. The
output is a JSON list sorted by `val_full`.

## WER/CER (intelligibility)

During training, the probe sentences (`SAMPLE_TEXTS` in `train_lora.py`: seven
German sentences plus an English and a Chinese regression probe) are
synthesized at every checkpoint and transcribed with faster-whisper on the CPU;
the language comes from the name suffix (`-de`, `-en`, `-zh`). To re-run it on
a sample folder:

```bash
scripts/run.sh de_lora/eval/eval_wer.py --dir <training>/runs/<run>/samples/checkpoint-epoch0 --size large-v3 --device cpu
```

Options: `--dir` (required), `--size` (default `large-v3`), `--device`
(`cpu`/`cuda`), `--compute-type` (default `int8`). With only nine sentences the
numbers are noisy; use them to catch regressions (e.g. the English probe
drifting away), not to rank close checkpoints.

## Speaker similarity (ECAPA)

Compares ECAPA embeddings (cosine) between reference(s) and generated audio.
By default it discovers the references `ref_*.wav` and the generations `*.wav`
(except the references) in `<training>/clone_out/`:

```bash
scripts/run.sh de_lora/eval/spk_similarity.py
scripts/run.sh de_lora/eval/spk_similarity.py --dir <folder> --refs a.wav b.wav --gens c.wav
```

Options: `--dir`, `--refs`, `--gens`, `--device` (`cuda`/`cpu`), `--savedir`.
Rule of thumb for the cosine: **~0.7+** ≈ same speaker; **< 0.3** ≈ a different person.

## Following the training

```bash
tensorboard --logdir <training>/runs/<run>/tb
```

## Listening

No metric replaces listening for timbre and pronunciation. For listening tests
through a running server see the optional `TTS_URL` block in
`recipes/pipeline_v5.sh`; to use an adapter with the companion C++ engine,
merge and convert it with `scripts/deploy_checkpoint.sh`.

## Per-corpus val loss and the release evaluation

`de_lora/eval/eval_val_per_corpus.py` evaluates the complete val split per corpus (base model
first, then each adapter; about 45 minutes per model on an R9700). The release evaluation of the
final model — WER/CER, speaker similarity, degraded references, long passages, Chinese and speed
for every GGUF and MLX variant — is in [`de_lora/eval/release/`](../de_lora/eval/release/README.md),
its results in [`RESULTS.md`](RESULTS.md).
