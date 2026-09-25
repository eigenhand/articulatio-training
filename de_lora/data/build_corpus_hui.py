"""build_corpus_hui.py — ingest HUI-Audio-Corpus-German from Parquet.

Unlike CML (WAV files on disk, referenced via a symlink), HUI comes as Parquet
with embedded 44.1 kHz audio, so it has to be written out. We take advantage of
that and measure the signal quality in the same pass — decoding twice would be
wasteful.

Filters:
  duration       4.0 to 10.2 s    (as in the rest of the chain)
  text density   4 to 30 chars/s
  SNR            default 20 dB    (median of CML is 18 dB, of HUI 29 dB)
  mains hum      below 10 dB      (narrow peak at 50 Hz)

Input: the Parquet files of the Hugging Face re-upload used for the runs
(Paradoxia/opendata-iisys-hui, columns speaker/audio/text); set HUI_GLOB, see
core/paths.py.

Usage:
  python de_lora/data/build_corpus_hui.py --hours 200 --min-snr 20
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from build_corpus import MAX_DUR, MIN_DUR, clean_text  # noqa: E402

CORPUS = "hui_de"
HUI_GLOB = paths.HUI_GLOB
SR = 24000


def signal_masse(w: np.ndarray, sr: int) -> tuple[float, float]:
    """(SNR in dB, mains hum in dB). An 8192-sample window = 2.9 Hz per bin,
    needed to separate narrow mains hum from the fundamental of a voice.

    SNR: the quietest fifth of the frames counts as noise, the loudest half as
    speech. Hum: the 48-52 Hz peak against its 30-70 Hz neighbourhood.
    ("Signalmasse" = signal measures; also used by tools/.)"""
    n, hop = 8192, 4096
    anz = (len(w) - n) // hop
    if anz < 3:
        return 0.0, 99.0
    fen = np.hanning(n)
    S = np.abs(np.fft.rfft(np.stack([w[i * hop:i * hop + n] * fen for i in range(anz)]), axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    idx = np.argsort(S.sum(axis=1))
    rausch = S[idx[:max(1, anz // 5)]].mean(axis=0)
    sprache = S[idx[max(1, anz // 2):]].mean(axis=0)
    snr = 10 * np.log10((sprache.sum() + 1e-12) / (rausch.sum() + 1e-12))
    spitze = sprache[(f > 48) & (f < 52)].max() if ((f > 48) & (f < 52)).any() else 0.0
    umfeld = sprache[((f > 30) & (f < 46)) | ((f > 54) & (f < 70))].mean() + 1e-20
    return float(snr), float(10 * np.log10(spitze / umfeld + 1e-20))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, required=True)
    ap.add_argument("--min-snr", type=float, default=20.0)
    ap.add_argument("--max-hum", type=float, default=10.0)
    args = ap.parse_args()

    import librosa

    out_dir = paths.DATASETS_ROOT / CORPUS
    wavs = out_dir / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)

    dateien = sorted(glob.glob(HUI_GLOB))
    print(f"[hui] {len(dateien)} parquet files, target {args.hours} h, "
          f"SNR >= {args.min_snr} dB")

    tf = (out_dir / "texts.csv").open("w", encoding="utf-8")
    spk_f = (out_dir / "speakers.jsonl").open("w", encoding="utf-8")
    qual_f = (out_dir / "qualitaet.jsonl").open("w", encoding="utf-8")

    ziel_s = args.hours * 3600
    total_s = 0.0
    n = 0
    # rejection counters: dauer = duration, text = text filter, snr, brumm = hum,
    # fehler = decode error (keys appear as-is in the log)
    verworfen = {"dauer": 0, "text": 0, "snr": 0, "brumm": 0, "fehler": 0}
    sprecher = set()

    for fp in dateien:
        if total_s >= ziel_s:
            break
        t = pq.read_table(fp, columns=["speaker", "audio", "text"])
        d = t.to_pydict()
        for spk, au, txt in zip(d["speaker"], d["audio"], d["text"]):
            if total_s >= ziel_s:
                break
            try:
                w, sr = sf.read(io.BytesIO(au["bytes"]), dtype="float32", always_2d=True)
                w = w.mean(axis=1)
            except Exception:                              # noqa: BLE001
                verworfen["fehler"] += 1
                continue
            dur = len(w) / sr
            if not (MIN_DUR <= dur <= MAX_DUR):
                verworfen["dauer"] += 1
                continue
            text = clean_text(txt, dur, CORPUS + "_de")     # _de -> num2words in German
            if not text:
                verworfen["text"] += 1
                continue
            snr, brumm = signal_masse(w, sr)
            if snr < args.min_snr:
                verworfen["snr"] += 1
                continue
            if brumm > args.max_hum:
                verworfen["brumm"] += 1
                continue

            w24 = librosa.resample(w, orig_sr=sr, target_sr=SR)
            peak = float(np.max(np.abs(w24)))
            w24 = (w24 * (0.95 / max(peak, 1e-9))).clip(-1, 1).astype("float32")

            idx = f"hui-{n:06d}"
            sf.write(str(wavs / f"{idx}.wav"), w24, SR, subtype="PCM_16")
            tf.write(f"wavs/{idx}.wav=={text}\n")
            spk_f.write(json.dumps({"idx": idx, "speaker": f"hui:{spk}"},
                                   ensure_ascii=False) + "\n")
            qual_f.write(json.dumps({"idx": idx, "snr_db": round(snr, 1),
                                     "brumm_db": round(brumm, 1),
                                     "dur": round(dur, 2)}) + "\n")
            sprecher.add(spk)
            total_s += len(w24) / SR
            n += 1
            if n % 2000 == 0:
                tf.flush(); spk_f.flush(); qual_f.flush()
                print(f"[hui] {n} clips, {total_s/3600:.1f} h, "
                      f"{len(sprecher)} speakers", flush=True)

    tf.close(); spk_f.close(); qual_f.close()
    print(f"\n[hui] DONE: {n} clips, {total_s/3600:.2f} h, {len(sprecher)} speakers")
    print(f"[hui] discarded: {verworfen}")

    cj = paths.CORPORA_JSON
    data = json.loads(cj.read_text(encoding="utf-8")) if cj.exists() else []
    data = [c for c in data if c.get("name") != CORPUS]
    data.append({"name": CORPUS, "root": CORPUS, "csv": "texts.csv",
                 "speakers": "speakers.jsonl", "enabled": True})
    cj.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[hui] {cj} updated: {[c['name'] for c in data]}")


if __name__ == "__main__":
    main()
