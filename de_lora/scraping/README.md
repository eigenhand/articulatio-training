# dataScrapping — pt-BR podcast dataset pipeline for TTS

> Inherited from the pt-BR template and not used for the German runs. It is
> kept for completeness; all numbers below are from the template's run.

Turns raw podcast episodes (YouTube, 48 kHz stereo, 1–3 h) into **24 kHz mono
PCM16 training chunks + transcripts** in the project format
(`datasets/podcast/texts.csv` → consumed by `prepare_dataset.py`).

## Pipeline

```
data/*.wav (48 kHz stereo)
   │  00_preprocess.py      mono mix + resampling to 16 kHz PCM16 (for the analysis models)
   ▼
work/16k/{tag}.wav ────────► work/episodes.json (manifest)
   │  01_diarize.py         pyannote segmentation-3.0 (speech vs overlap)
   │                        + SpeechBrain ECAPA (identity) + 2-pass clustering
   ▼
work/rttm/{tag}.rttm       turns per speaker
work/diar/{tag}.json       CLEAN intervals (no overlap) per speaker + hours
   │  02_slice.py           Silero VAD ∩ diarization → 1.4–10 s chunks
   │                        175 ms padding, speech ≥ 70 %, cut from the 48k original → 24 kHz
   ▼
datasets/podcast/wavs/{tag}_{N}.wav
work/chunks_index.jsonl    {chunk, tag, speaker, dur, speech_ratio}
   │  03_transcribe.py      X.ai API (4 workers, resumable, retry with backoff)
   │                        QC: language=pt, 6–25 chars/s, ≤ 30 % digits, NFC,
   │                        punctuation cleanup, numbers → words (num2words)
   ▼
datasets/podcast/texts.csv (format wavs/{chunk}.wav==text)
   │  04_qc_audit.py        report + 30 random pairs for human listening
   ▼
pilot_audit/playlist.txt   GATE: listen before scaling up / using the data
   05_report_diar.py       table of hours per speaker/episode (occasional use)
   api_check.py            health check of the transcription API
```

## Prerequisites

1. The project venv (see `../../docs/INSTALL.md`).
2. A `.env` in the artifacts root (`<PTBR_ARTIFACTS>/.env`) with:
   - `XAI_TRANSCRIBE_KEY` — key for the transcription API (https://console.x.ai)
   - `HF_TOKEN` — Hugging Face token with the terms **accepted** at:
     https://hf.co/pyannote/segmentation-3.0
3. Installed dependencies: `pyannote.audio silero-vad num2words speechbrain`
   (see the optional block in `../../requirements.txt`).

## Usage

```bash
# 1) put the raw .wav files into <PTBR_ARTIFACTS>/dataScrapping/data/
#    (names ... [youtubeid].wav; tag = youtubeid)
python 00_preprocess.py     # ~6 s per hour of audio; resumable (skips existing 16k files)
python 01_diarize.py        # ~40 s per episode on a GPU; writes work/diar/
python 02_slice.py          # ~2 min per hour of audio; resumable (skips tags already sliced)
python 03_transcribe.py     # ~550 chunks/min; resumable via work/transcribe_state.jsonl
python 04_qc_audit.py       # report + listening package
python api_check.py         # (optional) tests the API before large rounds
```

Interrupted? Run the same script again — every phase is
**idempotent/resumable** (existing codes, tags already sliced and chunks
already transcribed are skipped). Note: `api_check.py` still uses Windows path
separators, as in the template.

## Numbers of the template's real run (Aug 2026)

| Step | Value |
|---|---|
| Raw episodes | 18 (36 h, 23 GB, 48 kHz stereo) |
| Exclusive speech after diarization | 26.3 h |
| Sliced chunks | 23,134 (22.4 h new + pilot) |
| Accepted by transcription | **22,228 / 96.1 % — 23.2 h** |
| Total pipeline time | ~2.5 h (diarization ~12 min on GPU; transcription ~40 min) |
| Rejections | 3.9 % — extreme text density (< 6 or > 25 chars/s) or empty |

Rejections for high density (25–30 c/s) are real, fast speech; to recover them,
raise `CHARS_PER_S` in `03_transcribe.py`, remove the corresponding lines from
`work/transcribe_state.jsonl` and run step 03 again.

## Formats

- `work/chunks_index.jsonl` — `{"chunk","tag","speaker","dur","speech_ratio","spans"}`
  (`speaker` does NOT share identity across episodes — every tag has its own SPEAKER_00…)
- `datasets/podcast/texts.csv` — lines `wavs/{chunk}.wav==text` (the parser of
  `prepare_dataset.py` accepts them directly; `podcast_*` names do not collide with Tata's `sample-*`)
- `work/transcribe_state.jsonl` — API manifest:
  `{"chunk","status":"ok|rejected","text"|"reason"}`

## Technical decisions

- **Diarization without a ready-made pipeline**: `pyannote.audio` 4.x resolves
  `speaker-diarization-3.1` → the gated repo *community-1* (one more consent), and
  3.3.2 (the classic stack) breaks with `torchaudio>=2.9`. We use the minimal
  official recipe: `segmentation-3.0` (activation > 0.6, exclusive frames = no
  overlap) + ECAPA embeddings + agglomerative clustering (threshold 0.45) with
  dissolution of clusters < 60 s.
- **Chunks are cut from the 48k original** (not from the 16k intermediate) —
  maximum fidelity when resampling to 24 kHz.
- **No denoising**: studio speech is clean; VAD + diarization already discard
  music and crosstalk. (DeepFilterNet was dropped: its Rust build fails on Windows.)
- **Numbers → words** (`num2words` pt-BR): the Tata corpus is 99.8 % free of
  digits — consistency with the pre-training.
- **Disk**: the intermediates in `work/16k/` (~4 GB) can be deleted after phase 2;
  the full pipeline used ~11 GB (chunks + tokens + wavs24). Check free space before
  large rounds (a 3 h episode produces ~2 GB of chunk wavs).

## Integration with training

`prepare_dataset.py process` (with the `podcast` corpus registered or
`BREEZE_DATASET_DIR=…/datasets/podcast`) encodes the chunks, `finalize` makes the
split stratified by corpus with the `speaker` field, and training uses the
`ref_edit_auto` variant (reference from another clip of the SAME speaker).
