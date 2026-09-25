# Results: Articulatio-DE (released model, run 6)

Evaluated model: **`de-en-v5-abklingen/final`**, the released Articulatio-DE. Run 5 was stopped at
step 3,000 of 9,508; run 6 then continued it for 500 steps with the learning rate decaying from 1.5e-5
to 0 (same data, same noise-augmented references; *abklingen* is German for *decay*). See
[`RUNS.md`](RUNS.md). Measured on 25 September 2026.
Only this model was tested; there is no comparison with the base model or earlier runs.

Released variants (from the same merged checkpoint): GGUF **Q8_0** (3.3 GB) and **Q4_K** (2.4 GB)
for articulatio.cpp, and the adapter itself; **MLX 4 bit** (2.9 GB) is the conversion measured for
articulatio-mlx. F16, Q6_K, MLX 8 bit and MLX 4 bit with an 8-bit depth decoder were measured as
well and were not better than these (WER within the same intervals); they are not released.

## Summary

- **Intelligibility is at the level of real recordings.** Cloned German speech reaches 4.2 %
  (Q8_0), 5.8 % (Q4_K) and 4.6 % (MLX 4 bit) WER against 4.4 % for the real recordings of the
  same sentences (Whisper's own error on real speech); English 2.6–3.1 % against 2.6 %. The
  differences between the variants are within the 95 % intervals.
- **Voice similarity matches two real clips of the same speaker**: 0.71–0.72 ECAPA cosine, the
  real pairs give 0.715.
- **Q8_0 is the safest GGUF**; Q4_K has the widest interval and the only failed clone of the GGUF
  files, but is usable.
- **Degraded references do not make the output noisy**: the output SNR stays at 37.6–40.8 dB,
  at most 4.5 dB below cloning from the clean reference.
- **The clean-studio instruction has no measurable effect** (SNR, high-frequency share, WER), at cfg 1
  or 3. What a degraded reference does cost is voice similarity: ~0.66 instead of ~0.73.
- **Long text must be split into sentences.** Generated in one piece, ~70 s passages derail in
  every variant after 15–40 s (WER 17–71 %, several stop early). Split into sentences
  (`split_chars` 120), the same passages come out at 1.6–6.2 % WER over 71–82 s, with stable
  voice and brightness to the end.
- **Chinese is broken** (CER 58–73 %): partly right, partly English or invented syllables. The
  fine-tune never saw Chinese.
- **Speed:** RTF 0.49 (Q8_0) and 0.43 (Q4_K) on an NVIDIA RTX 4070, 0.69 and 0.65 on an AMD
  R9700 (both Vulkan), 0.69 on an M3 Pro with articulatio-mlx (0.87 with guidance at cfg 3).

## Method

Test set: 130 utterances of the **test split**, 3–10 s, from all six corpora (HUI 25, Common Voice
broad 25, Common Voice rare words 25, CML-TTS rare words 15, Thorsten rare words 15, HiFiTTS-2
English 25), 104 speakers, 12 min of audio. Each utterance is cloned from a *different* clip of its
speaker (val or test split), with the standard instruction of training (`Sprich klar und
natuerlich.` / `Speak clearly and naturally.`), cfg 1, one fixed seed per item shared by all
variants, whole text in one piece.

Further sets: 30 references of HUI, Thorsten and HiFiTTS-2 degraded with the run-5 degradations
(reverb 14, mains hum 7, colored noise 13, microphone coloration 4, telephone band 3, low-pass 3,
Opus/MP3 5, clipping 1; 1–3 per clip), each with the plain instruction, the clean-studio
instruction at cfg 1 and at cfg 3; 10 Chinese sentences by voice design; two passages of 174 and
185 words (HUI reader, Thorsten) in one piece.

Scoring: Whisper large-v3 (mlx-whisper, greedy, language fixed), text normalized on both sides
(lowercase, no punctuation, digits written out, ß→ss); WER/CER are corpus-level with a 95 %
bootstrap interval over items. Speaker similarity: ECAPA (speechbrain spkrec-ecapa-voxceleb)
cosine to the clean reference clip. SNR: 95th minus 10th percentile of 20 ms frame energies, a
reference-free proxy for background noise. HF ratio: share of energy at 4–12 kHz. Scripts and
commands: [`de_lora/eval/release/`](../de_lora/eval/release/README.md); raw numbers:
[`docs/results/v5-final/`](results/v5-final/).

## Tables

### Clone set (130 test utterances, 104 speakers)

| Variant | WER de | CER de | WER en | Speaker sim. to reference | to real recording | SNR dB | HF ratio | Clips with WER > 50 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recordings | 4.4 % (2.9–5.9) | 1.3 % (0.8–2.0) | 2.6 % (1.2–4.1) | 0.715 | – | 43.4 | 0.042 | 0 / 130 |
| Q8_0 | 4.2 % (2.7–6.0) | 1.2 % (0.7–1.9) | 2.6 % (1.1–4.3) | 0.716 | 0.698 | 40.7 | 0.035 | 0 / 130 |
| Q4_K | 5.8 % (3.4–8.6) | 2.5 % (1.0–5.0) | 2.6 % (1.1–4.2) | 0.714 | 0.698 | 40.2 | 0.035 | 1 / 130 |
| MLX-4bit | 4.6 % (3.1–6.3) | 1.2 % (0.7–1.9) | 3.1 % (1.6–4.6) | 0.706 | 0.696 | 40.5 | 0.040 | 0 / 130 |

#### WER per corpus

| Variant | cml_rare_de | cv_breit_de | cv_rare_de | hifitts2_en | hui_de | thorsten_rare_de |
|---|---:|---:|---:|---:|---:|---:|
| recordings | 5.8 % | 4.7 % | 3.8 % | 2.6 % | 3.4 % | 5.6 % |
| Q8_0 | 2.6 % | 3.9 % | 4.5 % | 2.6 % | 4.2 % | 6.3 % |
| Q4_K | 8.4 % | 5.9 % | 4.2 % | 2.6 % | 6.4 % | 3.5 % |
| MLX-4bit | 5.2 % | 3.9 % | 5.3 % | 3.1 % | 4.2 % | 4.9 % |

### Degraded references (30 items of HUI, Thorsten, HiFiTTS-2)

Same items cloned from the clean reference (clone set) as the target quality.

| Variant | Condition | SNR dB | HF ratio | Speaker sim. | WER |
|---|---|---:|---:|---:|---:|
| Q8_0 | clean reference | 41.6 | 0.036 | 0.724 | 3.6 % (1.4–6.6) |
| Q8_0 | degraded, plain instruction | 39.5 | 0.039 | 0.668 | 2.3 % (0.7–4.4) |
| Q8_0 | degraded, clean instruction, cfg 1 | 38.1 | 0.031 | 0.655 | 1.7 % (0.4–3.2) |
| Q8_0 | degraded, clean instruction, cfg 3 | 39.3 | 0.032 | 0.656 | 2.3 % (0.8–4.0) |
| Q4_K | clean reference | 40.2 | 0.030 | 0.731 | 1.3 % (0.4–2.5) |
| Q4_K | degraded, plain instruction | 38.2 | 0.037 | 0.661 | 3.4 % (1.5–5.7) |
| Q4_K | degraded, clean instruction, cfg 1 | 38.2 | 0.030 | 0.665 | 1.5 % (0.6–2.5) |
| Q4_K | degraded, clean instruction, cfg 3 | 38.1 | 0.039 | 0.641 | 1.9 % (0.7–3.2) |
| MLX-4bit | clean reference | 42.1 | 0.040 | 0.727 | 3.0 % (1.3–4.8) |
| MLX-4bit | degraded, plain instruction | 37.6 | 0.037 | 0.674 | 2.3 % (0.9–4.1) |
| MLX-4bit | degraded, clean instruction, cfg 1 | 38.9 | 0.036 | 0.654 | 2.3 % (0.9–4.3) |
| MLX-4bit | degraded, clean instruction, cfg 3 | 38.7 | 0.024 | 0.641 | 7.8 % (1.7–17.7) |

### Chinese regression (10 sentences, voice design)

| Variant | CER |
|---|---:|
| Q8_0 | 73.4 % (41.8–111.5) |
| Q4_K | 58.4 % (36.2–85.6) |
| MLX-4bit | 70.1 % (50.6–88.6) |

### Long passages (~70 s in one piece, no sentence splitting)

Per 10 s window; first = mean of the first two windows, last = mean of the last two.

| Variant | Passage | Duration | WER | HF ratio first → last | Speaker sim. first → last | Whisper log-prob first → last |
|---|---|---:|---:|---:|---:|---:|
| Q8_0 | long-hui_de | 35 s | 70.8 % | 0.023 → 0.019 | 0.701 → 0.700 | -0.21 → -0.42 |
| Q8_0 | long-thorsten_rare_de | 71 s | 17.0 % | 0.011 → 0.023 | 0.834 → 0.861 | -0.23 → -0.11 |
| Q4_K | long-hui_de | 43 s | 52.4 % | 0.017 → 0.006 | 0.613 → 0.642 | -0.13 → -0.31 |
| Q4_K | long-thorsten_rare_de | 55 s | 35.2 % | 0.020 → 0.033 | 0.831 → 0.829 | -0.09 → -0.22 |
| MLX-4bit | long-hui_de | 39 s | 52.4 % | 0.012 → 0.006 | 0.639 → 0.625 | -0.07 → -0.35 |
| MLX-4bit | long-thorsten_rare_de | 60 s | 42.6 % | 0.014 → 0.014 | 0.781 → 0.793 | -0.18 → -0.11 |

### Long passages split into sentences (Q8_0, `split_chars` 120, 2 seeds)

| Passage | Seed | Duration | WER | HF ratio first → last | Speaker sim. first → last | Whisper log-prob first → last |
|---|---:|---:|---:|---:|---:|---:|
| long-hui_de | 1 | 71 s | 3.8 % | 0.012 → 0.023 | 0.726 → 0.716 | -0.05 → -0.04 |
| long-hui_de | 2 | 72 s | 1.6 % | 0.024 → 0.020 | 0.731 → 0.763 | -0.07 → -0.06 |
| long-thorsten_rare_de | 1 | 78 s | 6.2 % | 0.019 → 0.014 | 0.848 → 0.834 | -0.10 → -0.08 |
| long-thorsten_rare_de | 2 | 82 s | 6.2 % | 0.033 → 0.025 | 0.799 → 0.831 | -0.15 → -0.13 |

### Speed

One server at a time, no other model on the GPU, 12 clone requests after a warm-up
(includes encoding the uploaded reference each time), Vulkan backend:

| Variant | AMD Radeon AI PRO R9700 | NVIDIA GeForce RTX 4070 |
|---|---:|---:|
| Q8_0 | 0.69 | 0.49 |
| Q4_K | 0.65 | 0.43 |

The RTX 4070 used the NVIDIA Vulkan driver 595.91.07 and shared the card with a desktop
display; the R9700 was otherwise idle.

Apple M3 Pro, articulatio-mlx: see [`results/v5-final/speed_mlx_m3pro.md`](results/v5-final/speed_mlx_m3pro.md)
(4 bit: RTF 0.69, 0.87 with cfg 3; mlx-audio for comparison: 2.0 / 3.7).

## Engine differences found

- On MLX the same degraded-reference job at cfg 3 (`hifitts2_en-en-002245`)
  returned no audio: EOS as the first frame. articulatio.cpp blocks EOS for the first
  `max(4, letters/2)` steps of a piece; articulatio-mlx does not yet, and it does not split long
  text into sentences either. Both explain MLX-only failures here.

## Not measured, limits

- **Val loss per corpus** on the full val split was not completed. The 96-item stratified val loss
  from training is available: 6.338 at the end of run 4, 6.406 for this model (same items, lower is
  better).
- No base-model or run-4 comparison, so the Chinese result says *broken*, not *how much worse*.
- One seed per item: single-clip differences between variants are sampling luck; only the
  aggregates with their intervals are meaningful.
- The SNR proxy sees stationary noise and hum; it does not capture reverberation or codec artifacts.
  Judging the degraded-reference outputs by ear is still worthwhile.
- Whisper errors on names and rare words count as model errors on both sides (that is why the real
  recordings are not at 0 %).
