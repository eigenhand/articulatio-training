"""make_testset.py — the release test set, drawn from the test split (never trained on).

Per corpus, target utterances of 3-10 s; each is cloned from a *different* clip of
the same speaker (3-10 s, from the val or test split), so the model has to carry
the voice to new text. Deterministic (sorted, hashed picks). Also writes degraded
copies of the references of the clean corpora (tools/rauschvorlagen.py, same
degradations as run 5 used in training) for the clean-instruction test.

  python make_testset.py --out testset/   (paths from de_lora/core/paths.py / .env)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "de_lora" / "core"))
sys.path.insert(0, str(REPO / "tools"))
import paths  # noqa: E402

PER_CORPUS = {"hui_de": 25, "thorsten_rare_de": 15, "cv_breit_de": 25, "cv_rare_de": 25,
              "cml_rare_de": 15, "hifitts2_en": 25}
CLEAN = {"hui_de", "thorsten_rare_de", "hifitts2_en"}   # the corpora run 5 degraded
NOISY_PER_CORPUS = 10


def h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "noisy_refs").mkdir(parents=True, exist_ok=True)

    split = {}
    for name in ("val", "test"):
        for l in open(paths.TRAINING / f"splits_{name}.txt", encoding="utf-8"):
            if l.strip():
                split[l.strip()] = name
    recs = [json.loads(l) for l in open(paths.TRAINING / "dataset_meta.jsonl", encoding="utf-8")]
    recs = [r for r in recs if r["idx"] in split and not r.get("truncated")]
    by_spk = defaultdict(list)
    for r in recs:
        by_spk[(r["corpus"], r["speaker"])].append(r)

    items = []
    for corpus, n in PER_CORPUS.items():
        cands = [r for r in recs if r["corpus"] == corpus and split[r["idx"]] == "test"
                 and 3.0 <= r["dur_proc_s"] <= 10.0]
        cands.sort(key=lambda r: h("target:" + r["idx"]))
        used_spk = defaultdict(int)
        for r in cands:
            if len([i for i in items if i["corpus"] == corpus]) >= n:
                break
            key = (corpus, r["speaker"])
            if used_spk[key] >= (n if corpus == "thorsten_rare_de" else 2):   # spread over speakers
                continue
            refs = [x for x in by_spk[key] if x["idx"] != r["idx"] and 3.0 <= x["dur_proc_s"] <= 10.0]
            if not refs:
                continue
            ref = min(refs, key=lambda x: h("ref:" + r["idx"] + x["idx"]))
            used_spk[key] += 1
            root = paths.DATASETS_ROOT / corpus
            items.append({
                "id": f"{corpus}-{r['idx']}", "corpus": corpus, "lang": corpus.rsplit("_", 1)[1],
                "speaker": r["speaker"], "text": r["text"], "target_wav": str(root / r["wav_rel"]),
                "target_dur": r["dur_proc_s"], "ref_idx": ref["idx"], "ref_wav": str(root / ref["wav_rel"]),
                "ref_text": ref["text"], "ref_dur": ref["dur_proc_s"],
            })

    from rauschvorlagen import SR, stoere
    import librosa
    noisy = defaultdict(int)
    for it in items:
        if it["corpus"] in CLEAN and noisy[it["corpus"]] < NOISY_PER_CORPUS:
            x, sr = sf.read(it["ref_wav"], dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != SR:
                x = librosa.resample(x, orig_sr=sr, target_sr=SR)
            y, kinds = stoere(x, "eval:" + it["ref_idx"])
            p = out / "noisy_refs" / f"{it['id']}.wav"
            sf.write(p, np.clip(y, -1, 1), SR)
            it["noisy_ref_wav"], it["noisy_kinds"] = str(p), kinds
            noisy[it["corpus"]] += 1

    with open(out / "testset.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    cnt = defaultdict(int)
    for it in items:
        cnt[it["corpus"]] += 1
    print(f"{len(items)} items", dict(cnt), "noisy refs:", dict(noisy),
          "speakers:", len({(i['corpus'], i['speaker']) for i in items}))


if __name__ == "__main__":
    main()
