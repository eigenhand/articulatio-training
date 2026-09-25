"""build_corpus_de.py — ingestion of CML-TTS German, without copying the audio.

Difference to the pt-BR template (build_corpus.py): there the corpora are
Parquet files, so every selected clip has to be decoded and written out as its
own WAV. The German CML set is already available unpacked as 24 kHz mono WAV
(CML_DE_ROOT). Writing it out again would be ~35 GB of duplicates in more than
100,000 files — expensive on virtiofs and completely unnecessary.

Instead: `wavs/` becomes a symlink to the original dataset and texts.csv points
into it with sub-paths (LINE_RE allows slashes). Only three small text files
are written.

Selection as in the template: round-robin over speakers, so that the hours are
not dominated by the few speakers with the most recordings.

Usage:
  python de_lora/data/build_corpus_de.py --hours 200
"""
from __future__ import annotations

import argparse
import csv as _csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from build_corpus import MAX_DUR, MIN_DUR, clean_text  # noqa: E402

CORPUS = "cml_de"
# Extracted CML-TTS German (the folder with train.csv/dev.csv/test.csv and
# train/audio/...); set CML_DE_ROOT, see core/paths.py.
CML_DE_ROOT = paths.CML_DE_ROOT

# Agreement between transcript and wav2vec ASR. Low values are silent
# transcription errors — you do not want them in training, because the model
# then maps text onto audio that does not belong to it.
MIN_LEVENSHTEIN = 0.95


def scan() -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    kept = dropped = 0
    for split in ("train", "dev"):
        csv_p = CML_DE_ROOT / f"{split}.csv"
        if not csv_p.is_file():
            print(f"[de] {csv_p} missing — skipped")
            continue
        with csv_p.open(encoding="utf-8", newline="") as fh:
            for row in _csv.DictReader(fh, delimiter="|"):
                try:
                    dur = float(row["duration"])
                    lev = float(row.get("levenshtein") or 0.0)
                except (TypeError, ValueError, KeyError):
                    dropped += 1
                    continue
                if not (MIN_DUR <= dur <= MAX_DUR) or lev < MIN_LEVENSHTEIN:
                    dropped += 1
                    continue
                text = clean_text(row["transcript"], dur, CORPUS)
                if not text:
                    dropped += 1
                    continue
                rel = row["wav_filename"]           # e.g. train/audio/252/1552/x.wav
                idx = Path(rel).stem
                g = f"cmlde:{row['client_id']}"
                groups.setdefault(g, []).append(
                    {"rel": rel, "idx": idx, "group": g, "dur": dur, "text": text})
                kept += 1
    print(f"[de] scan: {kept} usable, {dropped} discarded, {len(groups)} speakers")
    return groups


def select(groups: dict[str, list[dict]], hours: float) -> list[dict]:
    # Shuffle deterministically within each speaker (hash instead of random),
    # so that a repeated run makes the same selection.
    for g in groups:
        groups[g].sort(key=lambda c: hashlib.sha1(c["rel"].encode()).hexdigest())
    order = sorted(groups)
    ptr = {g: 0 for g in order}
    target = hours * 3600
    out: list[dict] = []
    total = 0.0
    exhausted = False
    while total < target and not exhausted:
        exhausted = True
        for g in order:
            if total >= target:
                break
            if ptr[g] >= len(groups[g]):
                continue
            exhausted = False
            c = groups[g][ptr[g]]
            ptr[g] += 1
            out.append(c)
            total += c["dur"]
    n_spk = len({c["group"] for c in out})
    print(f"[de] selection: {len(out)} clips / {total/3600:.2f} h from {n_spk} speakers")
    return out


def write(selected: list[dict]) -> None:
    out_dir = paths.DATASETS_ROOT / CORPUS
    out_dir.mkdir(parents=True, exist_ok=True)

    link = out_dir / "wavs"
    if link.is_symlink() or link.exists():
        if not (link.is_symlink() and link.resolve() == CML_DE_ROOT.resolve()):
            sys.exit(f"[de] {link} exists and does not point to {CML_DE_ROOT}")
    else:
        link.symlink_to(CML_DE_ROOT, target_is_directory=True)
        print(f"[de] symlink {link} -> {CML_DE_ROOT}")

    with (out_dir / "texts.csv").open("w", encoding="utf-8") as tf, \
         (out_dir / "speakers.jsonl").open("w", encoding="utf-8") as sf, \
         (out_dir / "selection.jsonl").open("w", encoding="utf-8") as mf:
        for c in selected:
            tf.write(f"wavs/{c['rel']}=={c['text']}\n")
            sf.write(json.dumps({"idx": c["idx"], "speaker": c["group"]},
                                ensure_ascii=False) + "\n")
            mf.write(json.dumps({"idx": c["idx"], "rel": c["rel"], "group": c["group"],
                                 "dur": round(c["dur"], 3)}, ensure_ascii=False) + "\n")
    print(f"[de] written to {out_dir}")

    # Register the corpus so that parse_csv() finds it.
    cj = paths.CORPORA_JSON
    data = json.loads(cj.read_text(encoding="utf-8")) if cj.exists() else []
    data = [c for c in data if c.get("name") != CORPUS]
    data.append({"name": CORPUS, "root": CORPUS, "csv": "texts.csv",
                 "speakers": "speakers.jsonl", "enabled": True})
    cj.parent.mkdir(parents=True, exist_ok=True)
    cj.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[de] {cj} updated: {[c['name'] for c in data]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, required=True)
    args = ap.parse_args()
    write(select(scan(), args.hours))


if __name__ == "__main__":
    main()
