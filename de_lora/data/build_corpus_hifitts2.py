"""build_corpus_hifitts2.py — cut HiFiTTS-2 (English) utterances out of chapter MP3s.

Purpose: mix in English material so that the German LoRA does not eat up the
original language. In the first run English had become measurably worse
(WER 0.167 vs 0.0 on the German probes).

The source is a selection of 873 chapter MP3s; `selected_chapters.jsonl` holds,
per chapter, the list `utts` with `offset` and `duration`. Each chapter is
loaded once and then cut into utterances - the other way round it would be
870 times the same decoding.

Expected layout of HIFITTS2_ROOT (see core/paths.py):
  selected_chapters.jsonl   one chapter per line: {"spk", "path", "utts": [...]}
                            (manifest rows of nvidia/hifitts-2)
  chapters/<spk>__<book>__<stem>.mp3   chapter audio (LibriVox via archive.org)
  texte.tsv                 utterance id <TAB> ... ; column 4 = transcript
The selection/download script that produced this layout is not part of this
repository.

Usage:
  python de_lora/data/build_corpus_hifitts2.py --hours 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from build_corpus import ALLOWED, CHARS_PER_S, MAX_DUR, MIN_DUR  # noqa: E402

CORPUS = "hifitts2_en"
ROOT = paths.HIFITTS2_ROOT
SR = 24000


def clean_en(t: str, dur: float) -> str | None:
    """English cleanup. clean_text() from build_corpus spells numbers out via
    num2words - for English that would be the wrong language, and HiFiTTS-2
    already has normalized transcripts anyway."""
    import re
    import unicodedata
    t = unicodedata.normalize("NFC", t or "")
    t = t.replace("…", "...").replace("–", "-").replace("—", "-")
    t = ALLOWED.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip(" -")
    if not t:
        return None
    if not (CHARS_PER_S[0] <= len(t) / max(dur, 1e-6) <= CHARS_PER_S[1]):
        return None
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, required=True)
    args = ap.parse_args()

    import librosa

    texte = {}
    with (ROOT / "texte.tsv").open(encoding="utf-8") as fh:
        for ln in fh:
            teile = ln.rstrip("\n").split("\t")
            if len(teile) >= 4:
                texte[teile[0]] = teile[3]
    print(f"[en] {len(texte)} transcripts loaded")

    kapitel = [json.loads(l) for l in
               (ROOT / "selected_chapters.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[en] {len(kapitel)} chapters in the manifest")

    out_dir = paths.DATASETS_ROOT / CORPUS
    wavs = out_dir / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    tf = (out_dir / "texts.csv").open("w", encoding="utf-8")
    spk_f = (out_dir / "speakers.jsonl").open("w", encoding="utf-8")

    ziel_s = args.hours * 3600
    total_s = 0.0
    n = 0
    # skip counters: mp3 = chapter missing/undecodable, text = no transcript,
    # dauer = duration, leer = empty/silent (keys appear as-is in the log)
    fehlt = {"mp3": 0, "text": 0, "dauer": 0, "leer": 0}
    sprecher = set()

    for k, kap in enumerate(kapitel):
        if total_s >= ziel_s:
            break
        stamm = os.path.splitext(os.path.basename(kap["path"]))[0]
        mp3 = ROOT / "chapters" / f"{kap['spk']}__{stamm.split('_')[1]}__{stamm}.mp3"
        if not mp3.is_file():
            treffer = list((ROOT / "chapters").glob(f"*{stamm}.mp3"))
            if not treffer:
                fehlt["mp3"] += 1
                continue
            mp3 = treffer[0]
        try:
            # decode once per chapter, directly at the target rate
            voll, _ = librosa.load(str(mp3), sr=SR, mono=True)
        except Exception as exc:                        # noqa: BLE001
            print(f"[en] {mp3.name}: {type(exc).__name__}")
            fehlt["mp3"] += 1
            continue

        for u in kap.get("utts", []):
            if total_s >= ziel_s:
                break
            dur = float(u["duration"])
            if not (MIN_DUR <= dur <= MAX_DUR):
                fehlt["dauer"] += 1
                continue
            uid = os.path.splitext(os.path.basename(u["audio_filepath"]))[0]
            roh = texte.get(uid)
            if not roh:
                fehlt["text"] += 1
                continue
            text = clean_en(roh, dur)
            if not text:
                fehlt["leer"] += 1
                continue
            a = int(float(u["offset"]) * SR)
            b = a + int(dur * SR)
            w = voll[a:b]
            if len(w) < int(MIN_DUR * SR):
                fehlt["dauer"] += 1
                continue
            peak = float(np.max(np.abs(w)))
            if peak < 1e-4:
                fehlt["leer"] += 1
                continue
            w = (w * (0.95 / peak)).clip(-1, 1).astype("float32")

            idx = f"en-{n:06d}"
            sf.write(str(wavs / f"{idx}.wav"), w, SR, subtype="PCM_16")
            tf.write(f"wavs/{idx}.wav=={text}\n")
            spk_f.write(json.dumps({"idx": idx, "speaker": f"en:{kap['spk']}"}) + "\n")
            sprecher.add(kap["spk"])
            total_s += len(w) / SR
            n += 1
        if (k + 1) % 50 == 0:
            tf.flush(); spk_f.flush()
            print(f"[en] chapter {k+1}/{len(kapitel)}: {n} clips, "
                  f"{total_s/3600:.1f} h, {len(sprecher)} speakers", flush=True)

    tf.close(); spk_f.close()
    print(f"\n[en] DONE: {n} clips, {total_s/3600:.2f} h, {len(sprecher)} speakers")
    print(f"[en] skipped: {fehlt}")

    cj = paths.CORPORA_JSON
    data = json.loads(cj.read_text(encoding="utf-8")) if cj.exists() else []
    data = [c for c in data if c.get("name") != CORPUS]
    data.append({"name": CORPUS, "root": CORPUS, "csv": "texts.csv",
                 "speakers": "speakers.jsonl", "enabled": True})
    cj.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[en] {cj} updated: {[c['name'] for c in data]}")


if __name__ == "__main__":
    main()
