# Release evaluation

Scripts behind [`docs/RESULTS.md`](../../../docs/RESULTS.md). Every model variant gets the
same 232 generation jobs with the same seeds; outputs are scored with the same ASR and
speaker encoder.

| Script | Does |
|---|---|
| `make_testset.py` | 130 utterances of the **test split** (never trained on), 3–10 s, from all six corpora, 104 speakers; each is cloned from another clip of its speaker. 30 references of the clean corpora also get a degraded copy (`tools/rauschvorlagen.py`). |
| `jobs.py` | The job list: `clone` (130), `noisy` (30 × 3 conditions), `zh` (10 Chinese sentences, voice design), `long` (two passages of ~180 words in one piece). |
| `synth_server.py` | Runs the jobs against a `breeze-server` of articulatio.cpp (GGUF). |
| `synth_mlx.py` | Runs the jobs with articulatio-mlx (Apple Silicon). |
| `score.py` | WER/CER (Whisper large-v3, greedy), speaker similarity (ECAPA), SNR estimate, share of energy at 4–12 kHz; for long passages also per 10 s window. |
| `report.py` | Corpus-level WER/CER with 95 % bootstrap intervals and all tables. |

```bash
# on the training machine (paths from .env)
python make_testset.py --out testset/
python -c "import jobs, json; json.dump(jobs.build('testset', '<training>/dataset_meta.jsonl'), open('testset/jobs.json', 'w'), ensure_ascii=False)"
breeze-server model.gguf --port 18090 --ws-port -1 &
python synth_server.py --url http://127.0.0.1:18090 --testset testset/ --out out/q8_0
# on a Mac
python synth_mlx.py --model <mlx model> --jobs testset/jobs.json --out out/mlx-4bit --remap <old>=<new>
python score.py --testset testset/ --ground-truth --variant out/q8_0 --variant out/mlx-4bit
python report.py --testset testset/ --variant out/q8_0 --variant out/mlx-4bit --out results/
```

Needs `faster-whisper` (`--asr ct2`) or `mlx-whisper` (`--asr mlx`, default), `speechbrain`,
`jiwer`, `num2words`, `librosa`, `soundfile`. The two back ends gave the same WER on a
spot check of real recordings; all numbers in RESULTS.md come from mlx-whisper.
