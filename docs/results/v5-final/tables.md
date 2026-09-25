## Clone set (130 test utterances, 104 speakers)

| Variant | WER de | CER de | WER en | Speaker sim. to reference | to real recording | SNR dB | HF ratio | Clips with WER > 50 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recordings | 4.4 % (2.9–5.9) | 1.3 % (0.8–2.0) | 2.6 % (1.2–4.1) | 0.715 | – | 43.4 | 0.042 | 0 / 130 |
| Q8_0 | 4.2 % (2.7–6.0) | 1.2 % (0.7–1.9) | 2.6 % (1.1–4.3) | 0.716 | 0.698 | 40.7 | 0.035 | 0 / 130 |
| Q4_K | 5.8 % (3.4–8.6) | 2.5 % (1.0–5.0) | 2.6 % (1.1–4.2) | 0.714 | 0.698 | 40.2 | 0.035 | 1 / 130 |
| MLX-4bit | 4.6 % (3.1–6.3) | 1.2 % (0.7–1.9) | 3.1 % (1.6–4.6) | 0.706 | 0.696 | 40.5 | 0.040 | 0 / 130 |

### WER per corpus

| Variant | cml_rare_de | cv_breit_de | cv_rare_de | hifitts2_en | hui_de | thorsten_rare_de |
|---|---:|---:|---:|---:|---:|---:|
| recordings | 5.8 % | 4.7 % | 3.8 % | 2.6 % | 3.4 % | 5.6 % |
| Q8_0 | 2.6 % | 3.9 % | 4.5 % | 2.6 % | 4.2 % | 6.3 % |
| Q4_K | 8.4 % | 5.9 % | 4.2 % | 2.6 % | 6.4 % | 3.5 % |
| MLX-4bit | 5.2 % | 3.9 % | 5.3 % | 3.1 % | 4.2 % | 4.9 % |

## Degraded references (30 items of HUI, Thorsten, HiFiTTS-2)

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

## Chinese regression (10 sentences, voice design)

| Variant | CER |
|---|---:|
| Q8_0 | 73.4 % (41.8–111.5) |
| Q4_K | 58.4 % (36.2–85.6) |
| MLX-4bit | 70.1 % (50.6–88.6) |

## Long passages (~70 s in one piece, no sentence splitting)

Per 10 s window; first = mean of the first two windows, last = mean of the last two.

| Variant | Passage | Duration | WER | HF ratio first → last | Speaker sim. first → last | Whisper log-prob first → last |
|---|---|---:|---:|---:|---:|---:|
| Q8_0 | long-hui_de | 35 s | 70.8 % | 0.023 → 0.019 | 0.701 → 0.700 | -0.21 → -0.42 |
| Q8_0 | long-thorsten_rare_de | 71 s | 17.0 % | 0.011 → 0.023 | 0.834 → 0.861 | -0.23 → -0.11 |
| Q4_K | long-hui_de | 43 s | 52.4 % | 0.017 → 0.006 | 0.613 → 0.642 | -0.13 → -0.31 |
| Q4_K | long-thorsten_rare_de | 55 s | 35.2 % | 0.020 → 0.033 | 0.831 → 0.829 | -0.09 → -0.22 |
| MLX-4bit | long-hui_de | 39 s | 52.4 % | 0.012 → 0.006 | 0.639 → 0.625 | -0.07 → -0.35 |
| MLX-4bit | long-thorsten_rare_de | 60 s | 42.6 % | 0.014 → 0.014 | 0.781 → 0.793 | -0.18 → -0.11 |

## Speed during the evaluation

Wall time / audio duration over all jobs. Only comparable within one machine and run.

| Variant | RTF | Audio generated |
|---|---:|---:|
| Q8_0 | 0.77 | 24.1 min |
| Q4_K | 0.74 | 24.0 min |
| MLX-4bit | 1.15 | 24.1 min |
