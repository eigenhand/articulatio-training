"""synth_server.py — run the release jobs against a breeze-server (C++ engine, GGUF).

  python synth_server.py --url http://127.0.0.1:18090 --testset testset/ --out out/<variant>/

Writes <out>/<set>/<id>.wav (24 kHz mono) and <out>/timing.jsonl (wall time per job).
Whole text in one piece (split_chars 0), like the MLX synthesizer.
"""
from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path

import requests

from jobs import build


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--meta", help="dataset_meta.jsonl (only needed the first time, for long.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="clone,noisy,zh,long")
    a = ap.parse_args()
    out = Path(a.out)
    want = set(a.sets.split(","))
    out.mkdir(parents=True, exist_ok=True)
    timing = open(out / "timing.jsonl", "a", encoding="utf-8")
    for j in build(a.testset, a.meta):
        if j["set"] not in want:
            continue
        p = out / j["set"] / f"{j['id']}.wav"
        if p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {"text": j["text"], "instruction": j["instruction"], "cfg_scale": str(j["cfg"]),
                "seed": str(j["seed"]), "split_chars": "0", "max_new_tokens": str(j["max_frames"])}
        files = {}
        if j["ref_wav"]:
            data["ref_text"] = j["ref_text"]
            files["ref_audio"] = open(j["ref_wav"], "rb")
        t0 = time.perf_counter()
        r = requests.post(f"{a.url}/v1/audio/speech", data=data, files=files or None, timeout=900)
        r.raise_for_status()
        dt = time.perf_counter() - t0
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(r.content)
        timing.write(json.dumps({"set": j["set"], "id": j["id"], "seconds": dt,
                                 "audio_s": len(r.content) / 48000}) + "\n")
        timing.flush()


if __name__ == "__main__":
    main()
