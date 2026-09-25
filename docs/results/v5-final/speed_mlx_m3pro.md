MLX, M3 Pro 18 GB, MLX 0.32.2, articulatio-mlx d2128d8, `python scripts/bench.py MODEL --ref ref.wav --ref-text ...`
(same German sentence, ~7 s audio, seed 1, after warm-up, nothing else running). RTF = compute / audio.

| Variant | mlx-audio RTF (design / clone / clone+cfg3) | articulatio-mlx RTF (design / clone / clone+cfg3) | first audio, streaming |
|---|---|---|---|
| mlx-4bit | 2.02 / 1.97 / 3.74 | 0.69 / 0.69 / 0.87 | 0.34 / 0.38 / 0.61 s |
