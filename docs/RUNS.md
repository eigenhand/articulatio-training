# Run history

All runs used batch 8 × gradient accumulation 4, `--ref-edit-frac 0.9` and one
LoRA shape: r=64, α=64, rsLoRA, targets q/k/v/o + gate/up/down in backbone,
depth decoder and text encoder (148.3 M trainable parameters, 4.1 %), created in
run 1 and continued from there. The hours count the audio that remained after preparation
(trimming and filtering), as reported in `summary.json`.

| Run | Name | Training data | Starts from | Changes |
|---|---|---|---|---|
| 1 | `de-r64-e2` | CML-TTS German: 103,734 clips, 174.3 h | base model | new adapter; lr 3e-5, 2 epochs (5,836 steps, ~22 h) |
| 2 | `de-en-v2` | HUI-Audio-Corpus-German 100.9 h + HiFiTTS-2 (English) 51.1 h = 152.0 h | run 1, step 4500 | English data added, because after run 1 the English probe had degraded (probe WER 16.7 % vs 0 % for German); CML-TTS left out; lr 3e-5, 2 epochs planned (4,710 steps) |
| 3 | `de-en-v3` | run 2 + rare-word clips: Thorsten 15.9 h, CML 7.1 h, Common Voice 59.5 h = 234.5 h | run 2, step 2500 | per-clip rare-word re-weighting (target 50 examples per word, cap 4, Zipf ≥ 4.0); 1 epoch planned (4,100 steps) |
| 4 | `de-en-v4` | run 3 + broad Common Voice (≤ 100 clips per speaker) 229.0 h = 463.5 h, 338,055 clips | run 3, step 1000 | corpus weights `hui_de` 1.5, `thorsten_rare_de` 1.5, `hifitts2_en` 2.0, `cv_breit_de` 0.6 (≈ CV 50 %, HUI 26 %, English 17 %, Thorsten 6 %, CML 1 % of the draws); 1 epoch (9,508 steps, ~34 h) |
| 5 | `de-en-v5` | same data as run 4 | run 4, final | second pass with noise-augmented references (35 % degraded reference + "clean studio" instruction, 25 % instruction only) on the clean corpora HUI, Thorsten, HiFiTTS-2; lr 2e-5 (instead of 3e-5) so the end of the first pass is not overwritten; stopped at step 3,000 of 9,508 |
| 6 | `de-en-v5-abklingen` | same as run 5 | run 5, step 3000 | 500 steps, lr 1.5e-5 decaying to 0; **the released model**, evaluated in [`RESULTS.md`](RESULTS.md) |

The scripts in [`recipes/`](../recipes/) ran runs 2–5 (`pipeline_v2.sh` + `train_v2.sh`,
`pipeline_v3.sh` + `pipeline_v3b.sh`, `pipeline_v4.sh`, `pipeline_v5.sh`), with paths replaced by
configuration. Run 6 was started by hand with the run-5 settings plus `--resume-adapter`,
`--max-steps 500`, `--lr 1.5e-5` and `--warmup 25`. The recipes
are documented examples, not a one-click reproduction: they hard-code run names
and checkpoints and coordinate through log files in `LOG_DIR`, e.g.

```bash
nohup scripts/run.sh tools/cv_breit.py > "$LOG_DIR/cv-breit.log" 2>&1 &
nohup recipes/pipeline_v4.sh          > "$LOG_DIR/pipeline-v4.log" 2>&1 &
```

Each recipe header lists the commands whose logs it waits for. Two steps were
done by hand: disabling `cml_de` in `corpora.json` before run 2, and copying
`codes/` and `dataset_meta.jsonl` from `training-v2` to `training-v3` before run 3.
