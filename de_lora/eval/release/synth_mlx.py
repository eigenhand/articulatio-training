"""synth_mlx.py — run the release jobs with articulatio-mlx (Apple Silicon).

  python synth_mlx.py --model <mlx model dir> --jobs jobs.json --out out/<variant>/

jobs.json is written on the machine with the test set (python -c "import jobs, json;
json.dump(jobs.build('testset'), open('jobs.json', 'w'))"); --remap old=new rewrites the
reference paths for the copy of the test set on this machine.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--remap", action="append", default=[], help="OLD_PREFIX=NEW_PREFIX")
    ap.add_argument("--sets", default="clone,noisy,zh,long")
    a = ap.parse_args()
    from articulatio_mlx import SAMPLE_RATE, Articulatio

    eng = Articulatio.load(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    want = set(a.sets.split(","))
    remap = [r.split("=", 1) for r in a.remap]
    timing = open(out / "timing.jsonl", "a", encoding="utf-8")
    eng.generate("Hallo.", instruct="Sprich klar.", seed=0)          # compile / warm up
    for j in json.load(open(a.jobs, encoding="utf-8")):
        if j["set"] not in want:
            continue
        p = out / j["set"] / f"{j['id']}.wav"
        if p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        ref = j["ref_wav"]
        for old, new in remap:
            if ref and ref.startswith(old):
                ref = new + ref[len(old):]
        t0 = time.perf_counter()
        audio, st = eng.generate(j["text"], instruct=j["instruction"], ref_audio=ref,
                                 ref_text=j["ref_text"] if ref else None, cfg_scale=j["cfg"],
                                 seed=j["seed"], max_frames=j["max_frames"])
        dt = time.perf_counter() - t0
        sf.write(p, np.array(audio), SAMPLE_RATE)
        timing.write(json.dumps({"set": j["set"], "id": j["id"], "seconds": dt,
                                 "audio_s": audio.shape[0] / SAMPLE_RATE}) + "\n")
        timing.flush()


if __name__ == "__main__":
    main()
