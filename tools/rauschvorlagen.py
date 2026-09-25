"""rauschvorlagen.py — degraded references for run 5 (noisy reference, clean target).

("Rauschvorlagen" = noisy reference prompts.) Until now the model learns: if the
reference sounds noisy, the output sounds noisy too - in Common Voice, reference
and target of the same speaker are both noisy. Here the CLEAN corpora (HUI,
Thorsten, HiFiTTS-2) get a degraded version of every training clip. In training
it serves as the reference, the target stays clean, and a dedicated instruction
("Saubere Studioaufnahme ..." / "Clean studio recording ...") marks such
examples - so it stays controllable whether the output takes on the sound of
the reference or not.

Only synthetic degradations, no third-party noise collections (no licensing
questions): coloured noise, mains hum, room reverb, telephone and headset sound,
lossy compression (MP3/Opus), clipping, microphone colouration. 1-3 of them per
clip, random but reproducible per idx.

    audio      create the degradations (CPU), WAVs to <datasets>/rauschvorlagen
    kodieren   (= encode) convert the WAVs to codec codes (GPU) -> <training>/codes_rauschen

<training> is BREEZE_TRAINING_DIR (recipes/pipeline_v5.sh sets it), <datasets>
is BREEZE_DATASETS_DIR (see de_lora/core/paths.py). Needs ffmpeg with libmp3lame
and libopus. The final lines start with "[DONE] audio" (recipes/pipeline_v5.sh
waits for that) and "[DONE] kodieren".
"""
from __future__ import annotations

import argparse, hashlib, io, json, subprocess, sys, time, wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "de_lora" / "core"))
import paths  # noqa: E402  (all locations are configurable, see de_lora/core/paths.py)

TRAINING = paths.TRAINING      # BREEZE_TRAINING_DIR; the run-5 recipe points it at training-v3
AUS = paths.DATASETS_ROOT / "rauschvorlagen"
KORPORA = {"hui_de", "thorsten_rare_de", "hifitts2_en"}
SR = 24000


def _rng(idx: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha1(("rausch:" + idx).encode()).hexdigest()[:16], 16))


# farbe (colour): "weiss" = white, "rosa" = pink, "braun" = brown noise
def _farbrauschen(n: int, farbe: str, rng) -> np.ndarray:
    w = rng.standard_normal(n)
    if farbe == "weiss":
        return w
    spec = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1 / SR); f[0] = f[1]
    spec /= np.sqrt(f) if farbe == "rosa" else f          # 1/f or 1/f^2 power
    x = np.fft.irfft(spec, n)
    return x / (np.std(x) + 1e-12)


# mix the disturbance st into x at the given SNR
def _mische(x: np.ndarray, st: np.ndarray, snr_db: float) -> np.ndarray:
    ps = np.mean(x ** 2) + 1e-12
    pn = np.mean(st ** 2) + 1e-12
    return x + st * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))


def _biquad(x, typ, f0, q, gain_db=0.0):
    from scipy.signal import lfilter
    A = 10 ** (gain_db / 40); w0 = 2 * np.pi * f0 / SR
    al = np.sin(w0) / (2 * q); c = np.cos(w0)
    if typ == "peak":
        b = [1 + al * A, -2 * c, 1 - al * A]; a = [1 + al / A, -2 * c, 1 - al / A]
    elif typ == "lp":
        b = [(1 - c) / 2, 1 - c, (1 - c) / 2]; a = [1 + al, -2 * c, 1 - al]
    else:  # hp
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]; a = [1 + al, -2 * c, 1 - al]
    return lfilter(np.array(b) / a[0], np.array(a) / a[0], x)


def _codec(x: np.ndarray, rng) -> tuple[np.ndarray, str]:
    """Send the audio through a real lossy codec (ffmpeg, in memory)."""
    if rng.random() < 0.5:
        art, arg = f"mp3 {int(rng.choice([16, 24, 32]))}k", ["-c:a", "libmp3lame", "-b:a", "", "-f", "mp3"]
        arg[3] = art.split()[1]
    else:
        art, arg = f"opus {int(rng.choice([8, 12, 16]))}k", ["-c:a", "libopus", "-b:a", "", "-f", "ogg"]
        arg[3] = art.split()[1]
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    enc = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                          *arg, "pipe:1"], input=pcm, capture_output=True, check=True).stdout
    dec = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le", "-ar", str(SR), "-ac", "1",
                          "pipe:1"], input=enc, capture_output=True, check=True).stdout
    y = np.frombuffer(dec, dtype="<i2").astype(np.float32) / 32767
    y = y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
    return y, art


# Degradation types ("arten"; the labels are written to manifest.jsonl as they are):
# rauschen = noise, hall = reverb, telefon = telephone band / low-pass ("tiefpass"),
# codec = MP3/Opus, brummen = mains hum, uebersteuert = clipping, mikrofon = mic colouration.
def stoere(x: np.ndarray, idx: str) -> tuple[np.ndarray, list[str]]:
    from scipy.signal import fftconvolve
    rng = _rng(idx)
    arten = ["rauschen", "hall", "telefon", "codec", "brummen", "uebersteuert", "mikrofon"]
    gew = np.array([0.30, 0.20, 0.15, 0.15, 0.08, 0.05, 0.07])
    k = int(rng.choice([1, 2, 3], p=[0.45, 0.40, 0.15]))
    wahl = list(rng.choice(arten, size=k, replace=False, p=gew))
    # reverb before noise, codec last - the order in which it happens in reality
    reihenfolge = ["mikrofon", "hall", "telefon", "brummen", "rauschen", "uebersteuert", "codec"]
    kette = []
    for art in [a for a in reihenfolge if a in wahl]:
        if art == "mikrofon":
            for _ in range(int(rng.integers(2, 4))):
                f0, g = float(rng.uniform(150, 7000)), float(rng.uniform(-7, 7))
                x = _biquad(x, "peak", f0, float(rng.uniform(0.7, 2.0)), g)
            kette.append("mikrofon")
        elif art == "hall":
            rt60 = float(rng.uniform(0.2, 0.9)); n = int(rt60 * SR)
            t = np.arange(n) / SR
            rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
            rir[0] = float(rng.uniform(1.5, 5.0)) * np.max(np.abs(rir))   # direct sound
            x = fftconvolve(x, rir)[:len(x)]
            kette.append(f"hall {rt60:.2f}s")
        elif art == "telefon":
            if rng.random() < 0.5:
                x = _biquad(_biquad(x, "hp", 300, 0.7), "lp", 3400, 0.7)
                kette.append("telefon 300-3400")
            else:
                fc = float(rng.uniform(3500, 7500))
                x = _biquad(_biquad(x, "lp", fc, 0.7), "lp", fc, 0.7)
                kette.append(f"tiefpass {fc:.0f}")
        elif art == "brummen":
            t = np.arange(len(x)) / SR
            f0 = 50.0 if rng.random() < 0.8 else 60.0
            h = sum(float(rng.uniform(0.2, 1.0)) / k_ * np.sin(2 * np.pi * f0 * k_ * t) for k_ in range(1, 6))
            snr = float(rng.uniform(12, 28))
            x = _mische(x, h, snr)
            kette.append(f"brummen {snr:.0f}dB")
        elif art == "rauschen":
            farbe = str(rng.choice(["weiss", "rosa", "braun"]))
            snr = float(rng.uniform(5, 25))
            x = _mische(x, _farbrauschen(len(x), farbe, rng), snr)
            kette.append(f"{farbe} {snr:.0f}dB")
        elif art == "uebersteuert":
            x = x / (np.max(np.abs(x)) + 1e-9)
            grenze = float(rng.uniform(0.3, 0.7))
            x = np.clip(x, -grenze, grenze)
            kette.append(f"clip {grenze:.2f}")
        elif art == "codec":
            x, art_ = _codec(x, rng)
            kette.append(art_)
    x = x * (0.95 / (np.max(np.abs(x)) + 1e-9))
    return x.astype(np.float32), kette


def _auftrag(args):
    """Worker: load a clip, prepare it like prepare_dataset.process, degrade it, write it."""
    idx, pfad, ziel = args
    import librosa, soundfile as sf
    try:
        w, sr = sf.read(pfad, dtype="float32", always_2d=True)
        w = w.mean(axis=1)
        w, _ = librosa.effects.trim(w, top_db=40)
        if sr != SR:
            w = librosa.resample(w, orig_sr=sr, target_sr=SR)
        pk = float(np.max(np.abs(w))) if len(w) else 0.0
        if pk < 0.30 or pk > 0.99:
            w = w * (0.95 / max(pk, 1e-9))
        y, kette = stoere(w.astype(np.float64), idx)
        b = io.BytesIO()
        with wave.open(b, "wb") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR)
            f.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
        with open(ziel, "wb", buffering=0) as f:     # a single write(): virtiofs
            f.write(b.getvalue())
        return idx, kette, None
    except Exception as exc:                          # noqa: BLE001
        return idx, None, f"{type(exc).__name__}: {exc}"


# Training-split clips of the clean corpora (untruncated), with their source WAV path.
def auswahl() -> list[dict]:
    train = {l.strip() for l in open(TRAINING / "splits_train.txt", encoding="utf-8") if l.strip()}
    wurzel = {c["name"]: paths.DATASETS_ROOT / c["root"]
              for c in json.load(open(paths.CORPORA_JSON, encoding="utf-8"))}
    recs = []
    for l in open(TRAINING / "dataset_meta.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["idx"] in train and r.get("corpus") in KORPORA and not r.get("truncated"):
            r["pfad"] = str(wurzel[r["corpus"]] / r["wav_rel"])
            recs.append(r)
    return recs


def audio(worker: int) -> None:
    import concurrent.futures as cf, multiprocessing as mp
    recs = auswahl()
    (AUS / "wavs").mkdir(parents=True, exist_ok=True)
    fertig = set()
    man = AUS / "manifest.jsonl"
    if man.exists():
        fertig = {json.loads(l)["idx"] for l in man.open(encoding="utf-8")}
    todo = [r for r in recs if r["idx"] not in fertig]
    print(f"[audio] {len(recs)} clean training clips, {len(fertig)} already done, {len(todo)} to do", flush=True)
    t0, n, fehler = time.time(), 0, 0
    with man.open("a", encoding="utf-8") as mf, \
         cf.ProcessPoolExecutor(worker, mp_context=mp.get_context("spawn")) as ex:
        for idx, kette, err in ex.map(_auftrag, [(r["idx"], r["pfad"], str(AUS / "wavs" / f"{r['idx']}.wav"))
                                                for r in todo], chunksize=16):
            if err:
                fehler += 1; print(f"[audio] ERROR {idx}: {err}", flush=True); continue
            mf.write(json.dumps({"idx": idx, "kette": kette}, ensure_ascii=False) + "\n")
            n += 1
            if n % 5000 == 0:
                print(f"[audio] {n}/{len(todo)} {time.time()-t0:.0f} s", flush=True)
    # NOTE: recipes/pipeline_v5.sh waits for a line starting with "[DONE] audio".
    print(f"[DONE] audio: {n} clips, {fehler} errors, {time.time()-t0:.0f} s", flush=True)


def kodieren(device: str) -> None:
    sys.path.insert(0, str(REPO / "de_lora/core"))
    import soundfile as sf
    import common_breeze as CB
    from codestore import CodeWriter
    ziel = TRAINING / "codes_rauschen"
    w = CodeWriter(ziel)                          # resumable: has() knows what is done
    eintraege = [json.loads(l) for l in (AUS / "manifest.jsonl").open(encoding="utf-8")]
    todo = [e for e in eintraege if not w.has(e["idx"])]
    print(f"[kodieren] {len(eintraege)} clips, {len(todo)} to do", flush=True)
    atok = CB.import_qwen_tts().from_pretrained(str(CB.CKPT / "audio_tokenizer"), device_map=device)
    t0 = time.time()
    for k, e in enumerate(todo):
        x, _ = sf.read(AUS / "wavs" / f"{e['idx']}.wav", dtype="float32")
        codes = atok.encode(x, sr=SR)["audio_codes"][0]
        codes = codes.detach().cpu().numpy().astype("int16") if hasattr(codes, "cpu") else codes
        assert codes.ndim == 2 and codes.shape[1] == 16, codes.shape
        w.append(e["idx"], codes)
        if (k + 1) % 2000 == 0:
            r = (k + 1) / (time.time() - t0)
            print(f"[kodieren] {k+1}/{len(todo)} {r:.1f}/s, {(len(todo)-k-1)/r/60:.0f} min left", flush=True)
    w.close()                                     # otherwise the last part of the index is missing
    print(f"[DONE] kodieren: {len(todo)} clips, {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("schritt", choices=["audio", "kodieren"])   # "Schritt" = step
    ap.add_argument("--worker", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    audio(a.worker) if a.schritt == "audio" else kodieren(a.device)
