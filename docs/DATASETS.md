# Data: sources, corpus builders and preparation

The pipeline turns audio + text into **pre-extracted codec codes** and
**splits** that are ready for training. Processed corpora live in
`<PTBR_ARTIFACTS>/datasets/`, codes and splits in the training directory
(`BREEZE_TRAINING_DIR`). Start every command through `scripts/run.sh` (ROCm
settings, repository root as working directory).

## 1. Raw sources and their expected layout

All locations are configurable (see [CONFIGURATION.md](CONFIGURATION.md));
the defaults live under `<datasets>/_raw/`.

| Source | Variable | Expected content |
|---|---|---|
| CML-TTS German | `CML_DE_ROOT` | `train.csv`, `dev.csv`, `test.csv` (pipe-separated: `wav_filename`, `duration`, `transcript`, `client_id`, `levenshtein`) and the 24 kHz WAVs they reference |
| HUI-Audio-Corpus-German | `HUI_GLOB` | Parquet files with the columns `speaker`, `audio`, `text` (the runs used the Hugging Face re-upload `Paradoxia/opendata-iisys-hui`) |
| HiFiTTS-2 (English) | `HIFITTS2_ROOT` | `selected_chapters.jsonl` (manifest rows of `nvidia/hifitts-2` with `spk`, `path`, `utts[offset, duration, audio_filepath]`), `chapters/<spk>__<book>__<stem>.mp3` (LibriVox audio from archive.org), `texte.tsv` (utterance id in column 1, transcript in column 4). The selection/download script that produced this layout (873 chapters, 200 h, bandwidth ≥ 15 kHz) is not part of this repository. |
| Thorsten-Voice | `THORSTEN_DIR` | `TV-2021.02-Neutral/train-*.parquet` and `TV-2022.10-Neutral/train-*.parquet` from `Thorsten-Voice/TV-44kHz-Full` (columns `id`, `text`, `durationSeconds`, `audio`) |
| Common Voice 27.0 German | `CV_DIR` | `tsv/validated.tsv` and `tsv/clip_durations.tsv`, plus the release archive split into four byte ranges `teil.0` … `teil.3` |

Common Voice is never unpacked (about a million MP3 files); the tools stream the
archive once. To prepare `CV_DIR` from the downloaded `.tar.gz`:

```bash
cd "$CV_DIR"
split -n 4 -d -a 1 cv-corpus-27.0-*-de.tar.gz teil.      # -> teil.0 .. teil.3
mkdir -p tsv && cat teil.0 teil.1 teil.2 teil.3 | pigz -dc \
  | tar -x --wildcards '*.tsv' --strip-components=2 -C tsv
```

## 2. Corpus builders (German runs)

```bash
scripts/run.sh de_lora/data/build_corpus_de.py --hours 200                  # cml_de       (run 1)
scripts/run.sh de_lora/data/build_corpus_hui.py --hours 200 --min-snr 20    # hui_de       (run 2)
scripts/run.sh de_lora/data/build_corpus_hifitts2.py --hours 200            # hifitts2_en  (run 2)
```

Each builder writes `datasets/<corpus>/` with `wavs/`, `texts.csv` and
`speakers.jsonl` and registers the corpus (enabled) in `corpora.json`.
Filters: duration 4.0–10.2 s, text density 4–30 chars/s, NFC; the German
builders spell numbers out with num2words (HiFiTTS-2 transcripts are already
normalized). CML additionally requires a transcript/ASR
Levenshtein agreement ≥ 0.95 and is **not copied**: `wavs/` is a symlink to
`CML_DE_ROOT`. HUI is filtered for SNR ≥ 20 dB and mains hum < 10 dB (CML's
median SNR is 18 dB, HUI's 29 dB).

## 3. Rare words (run 3)

Audiobook corpora rarely contain everyday words such as *Temperaturen*,
*Telefon*, *Informationen* or *online*. The tools find such words and add clips
that contain them:

```bash
scripts/run.sh tools/analyse_seltene_woerter.py   # -> seltene_woerter.json/.tsv (needs espeak-ng, wordfreq)
scripts/run.sh tools/grossschreibung.py           # -> grossschreibung.json (capitalization table)
scripts/run.sh tools/suche_seltene_woerter.py     # -> thorsten_rare_de, cml_rare_de + rare_de_abdeckung.json
scripts/run.sh tools/cv_seltene_woerter.py        # -> cv_rare_de + rare_de_abdeckung_mit_cv.json (needs pigz)
```

- `analyse_seltene_woerter.py` lists words with Zipf ≥ 3 (wordfreq) that were
  heard ≤ 5 times in the CML + HUI transcripts, or whose rarest Gemma token was
  heard ≤ 5 times, plus their stress position (espeak-ng).
- `grossschreibung.py` builds a capitalization table from correctly written
  transcripts (counted mid-sentence only); 38 % of the selected Thorsten
  sentences are entirely lowercase and the tokenizer is case-sensitive.
- `suche_seltene_woerter.py` greedily selects clips per word (Thorsten before
  CML, `--quote` clips per word); `--trocken` is a dry run.
- `cv_seltene_woerter.py` tops the examples per word up from Common Voice
  (stricter filters: SNR ≥ 20 dB, hum < 10 dB, band edge ≥ 10 kHz, at most
  `--pro-sprecher` clips per speaker).

These tools register their corpora **disabled**; `recipes/pipeline_v3.sh` and
`recipes/pipeline_v3b.sh` enable them. At training time
`train_lora.py --wort-ausgleich <seltene_woerter.json>` re-weights clips by their
rarest word (see [TRAINING.md](TRAINING.md)).

## 4. Common Voice in breadth (run 4)

```bash
scripts/run.sh tools/cv_breit.py          # -> cv_breit_de (at most 100 clips per speaker)
```

Same filters as the rare-word pass; clips that the rare-word pass already
decoded are skipped. Without the per-speaker cap the few most prolific speakers (the
largest has 56,886 recordings) would dominate the sound.

## 5. Noise-augmented references (run 5)

```bash
BREEZE_TRAINING_DIR=<artifacts>/training-v3 scripts/run.sh tools/rauschvorlagen.py audio      # CPU
BREEZE_TRAINING_DIR=<artifacts>/training-v3 scripts/run.sh tools/rauschvorlagen.py kodieren   # GPU
```

`audio` writes a degraded copy (1–3 of: coloured noise, reverb, telephone band /
low-pass, MP3/Opus codec, mains hum, clipping, microphone colouration; needs
ffmpeg) of every training clip of the clean corpora (`hui_de`,
`thorsten_rare_de`, `hifitts2_en`); `kodieren` encodes them into
`<training>/codes_rauschen`. `train_lora.py --rausch-codes` then uses them as
references (see [TRAINING.md](TRAINING.md)).

## 6. Preparation for training

Two commands (`process` is resumable — it skips items already in the code store):

```bash
scripts/run.sh de_lora/core/prepare_dataset.py process [--limit N] [--device cuda]
scripts/run.sh de_lora/core/prepare_dataset.py finalize [--gold N] [--parity-device cuda]
```

- **process**: reads the `texts.csv` of every enabled corpus, trims silence,
  resamples to 24 kHz, peak-normalizes, cuts clips longer than 10.2 s at a
  silence gap (with a proportional text cut), encodes the audio once with the
  `Qwen3TTSTokenizer` (16 codebooks) and appends the codes to
  `<training>/codes/` (one `codes.i16` stream + `index.jsonl`, instead of one
  file per utterance) and a record to `dataset_meta.jsonl`.
- **finalize**: split **90/5/5 per corpus** (`splits_{train,val,test}.txt`; corpora
  with fewer than 40 items go to train), `manifest.csv`, `summary.json`, gold
  samples and the **parity check** against the official template. It prints
  `[finalize] parity FAILED for N/M`; the recipes only start training when N is 0.

Use a separate training directory per data generation (the recipes use
`training`, `training-v2`, `training-v3`): `finalize` reads the whole
`dataset_meta.jsonl`, so corpora encoded earlier in the same directory would
end up in the splits again.

## `texts.csv` format

One line per item, `wavs/<path>.wav==text` (sub-paths are allowed):

```
wavs/hui-000123.wav==Die Temperaturen steigen heute auf zwanzig Grad.
wavs/train/audio/<speaker>/<book>/<file>.wav==Am Bahnhof wartet schon der Zug.
```

`speakers.jsonl` (optional) adds `{"idx","speaker"}`; `ref_map.jsonl`
(optional) `{"idx","ref_idx","cos"}`.

## pt-BR template corpora (inherited)

`de_lora/data/download_datasets.py` (TAGARELA, CML-TTS PT, CETUC from Hugging
Face), `de_lora/data/build_corpus.py --corpus tagarela|cml_pt|cetuc`,
`de_lora/data/tagarela_speakers.py` (ECAPA speaker clustering) and the podcast
pipeline in `de_lora/scraping/` are kept from the template. They were not used
for the German runs; `build_corpus.py` still provides `clean_text()` and the
duration limits for the German builders.

## Next step

[`TRAINING.md`](TRAINING.md).
