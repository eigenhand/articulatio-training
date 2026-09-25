"""score.py — score generated audio of the release evaluation.

  python score.py --testset testset/ --variant out/v5-q8_0 [--variant ...] [--ground-truth]

Per WAV:
  wer, cer    Whisper large-v3 (greedy, language fixed per item; --asr mlx: mlx-whisper on
              Apple Silicon, --asr ct2: faster-whisper int8 on the CPU) against the
              text; both sides normalized (lowercase, no punctuation, digits written out
              with num2words, ß -> ss) so formatting choices of Whisper do not count as errors
  sim_ref     ECAPA cosine (speechbrain spkrec-ecapa-voxceleb) to the clean reference clip,
              i.e. how well the cloned voice matches; sim_target: to the real recording
  snr_db      level of loud frames (95th percentile of 20 ms frame energies) minus level of
              quiet frames (10th percentile): a reference-free estimate of background noise
  hf_ratio    share of spectral energy between 4 and 12 kHz, at the file's own sample rate
              (muffled audio has less)
Long passages additionally per 10 s window: hf_ratio, snr_db, sim_ref, whisper avg_logprob.
Results: <variant>/scores.json (per item) — aggregate with report.py.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
from num2words import num2words

SR = 16000


def load(path):
    """-> (16 kHz audio for Whisper/ECAPA, native audio, native rate)"""
    import librosa
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return (librosa.resample(x, orig_sr=sr, target_sr=SR) if sr != SR else x), x, sr


def load16(path):
    return load(path)[0]


def _num(m, lang):
    t = m.group()
    v = float(t.replace(",", ".")) if re.search(r"[.,]", t) else int(t)
    return " " + num2words(v, lang=lang) + " "


def normalize(s: str, lang: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = s.replace("%", " prozent " if lang == "de" else " percent ").replace("€", " euro ")
    if lang in ("de", "en"):
        s = re.sub(r"\d+([.,]\d+)?", lambda m: _num(m, lang), s)
    s = s.replace("ß", "ss")
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def audio_stats(x, sr):
    n = int(sr * 0.02)
    frames = x[: len(x) // n * n].reshape(-1, n)
    e = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)
    snr = float(np.percentile(e, 95) - np.percentile(e, 10)) if len(e) > 10 else float("nan")
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    hf = float(spec[(f >= 4000) & (f <= 12000)].sum() / max(spec[f <= 12000].sum(), 1e-12))
    return snr, hf


def item_id(job: dict) -> str:
    if job["set"] == "noisy":
        for suffix in ("-plain", "-clean-cfg1", "-clean-cfg3"):
            if job["id"].endswith(suffix):
                return job["id"][: -len(suffix)]
    return job["id"]


class _Seg:
    def __init__(self, d):
        self.text, self.avg_logprob = d["text"], d["avg_logprob"]


class Scorer:
    def __init__(self, threads, asr="mlx"):
        from speechbrain.inference.speaker import EncoderClassifier
        self.backend = asr
        if asr == "ct2":
            from faster_whisper import WhisperModel
            self.asr = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=threads)
        self.spk = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                  savedir=str(Path.home() / ".cache/spkrec-ecapa"),
                                                  run_opts={"device": "cpu"})
        self._emb = {}

    def emb(self, path=None, x=None):
        if path is not None and path in self._emb:
            return self._emb[path]
        x = load16(path) if x is None else x
        with torch.no_grad():
            e = self.spk.encode_batch(torch.from_numpy(np.ascontiguousarray(x))[None]).squeeze()
        e = e / e.norm()
        if path is not None:
            self._emb[path] = e
        return e

    @staticmethod
    def sim(a, b):
        return float((a * b).sum())

    def transcribe(self, x, lang):
        if self.backend == "mlx":
            import mlx_whisper
            r = mlx_whisper.transcribe(np.ascontiguousarray(x, dtype=np.float32),
                                       path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                                       language=lang, temperature=0.0,
                                       condition_on_previous_text=False, verbose=None)
            return [_Seg(d) for d in r["segments"]]
        segs, _ = self.asr.transcribe(x, language=lang, beam_size=1,
                                      condition_on_previous_text=False, vad_filter=False)
        return list(segs)

    def score(self, wav, text, lang, ref_wav=None, target_wav=None, long=False):
        x, xn, sr = load(wav)
        r = {"dur": len(x) / SR}
        if len(x) < SR * 0.3:
            return dict(r, wer=1.0, cer=1.0, hyp="", empty=True)
        segs = self.transcribe(x, lang)
        hyp = " ".join(s.text.strip() for s in segs)
        ref_n, hyp_n = normalize(text, lang), normalize(hyp, lang)
        r["hyp"] = hyp
        if lang == "zh":
            r["wer"] = float("nan")
            r["cer"] = jiwer.cer(ref_n.replace(" ", ""), hyp_n.replace(" ", "") or "_")
        else:
            r["wer"] = jiwer.wer(ref_n, hyp_n or "_")
            r["cer"] = jiwer.cer(ref_n, hyp_n or "_")
        r["snr_db"], r["hf_ratio"] = audio_stats(xn, sr)
        e = self.emb(x=x)
        if ref_wav:
            r["sim_ref"] = self.sim(e, self.emb(ref_wav))
        if target_wav:
            r["sim_target"] = self.sim(e, self.emb(target_wav))
        if long:
            win = []
            for k in range(0, int(len(x) / SR), 10):
                w = x[k * SR:(k + 10) * SR]
                if len(w) < 5 * SR:
                    break
                snr, hf = audio_stats(xn[k * sr:(k + 10) * sr], sr)
                ws = self.transcribe(w, lang)
                lp = float(np.mean([s.avg_logprob for s in ws])) if ws else float("nan")
                win.append({"t0": k, "hf_ratio": hf, "snr_db": snr, "avg_logprob": lp,
                            "sim_ref": self.sim(self.emb(x=w), self.emb(ref_wav)) if ref_wav else None})
            r["windows"] = win
        return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", required=True)
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--ground-truth", action="store_true", help="also score the real recordings")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--asr", choices=["mlx", "ct2"], default="mlx")
    ap.add_argument("--remap", action="append", default=[], help="OLD_PREFIX=NEW_PREFIX for audio paths")
    a = ap.parse_args()
    from jobs import build
    ts = Path(a.testset)
    items = {i["id"]: i for i in (json.loads(l) for l in open(ts / "testset.jsonl", encoding="utf-8"))}
    jobs = build(str(ts))
    remap = [r.split("=", 1) for r in a.remap]

    def rp(p):
        for old, new in remap:
            if p and p.startswith(old):
                return new + p[len(old):]
        return p

    for it in items.values():
        for k in ("target_wav", "ref_wav", "noisy_ref_wav"):
            if k in it:
                it[k] = rp(it[k])
    for j in jobs:
        j["ref_wav"] = rp(j["ref_wav"])
    sc = Scorer(a.threads, a.asr)

    if a.ground_truth:
        out = ts / "ground_truth_scores.json"
        res = json.loads(out.read_text()) if out.exists() else {}
        for it in items.values():
            if it["id"] in res:
                continue
            # sim_ref here compares two real clips of the same speaker: the natural ceiling
            r = sc.score(it["target_wav"], it["text"], it["lang"], ref_wav=it["ref_wav"])
            res[it["id"]] = dict(r, corpus=it["corpus"])
            out.write_text(json.dumps(res, ensure_ascii=False, indent=0), encoding="utf-8")
        print(f"ground truth: {len(res)} scored")

    for v in a.variant:
        vdir = Path(v)
        out = vdir / "scores.json"
        res = json.loads(out.read_text()) if out.exists() else {}
        for j in jobs:
            key = f"{j['set']}/{j['id']}"
            wav = vdir / j["set"] / f"{j['id']}.wav"
            if key in res or not wav.exists():
                continue
            it = items.get(item_id(j))
            clean_ref = it["ref_wav"] if it else j["ref_wav"]   # identity against the clean clip
            r = sc.score(str(wav), j["text"], j["lang"], ref_wav=clean_ref,
                         target_wav=it["target_wav"] if it and j["set"] == "clone" else None,
                         long=j["set"] == "long")
            res[key] = dict(r, set=j["set"], corpus=it["corpus"] if it else j["set"])
            out.write_text(json.dumps(res, ensure_ascii=False, indent=0), encoding="utf-8")
        print(f"{v}: {len(res)} scored")


if __name__ == "__main__":
    main()
