"""cv_seltene_woerter.py — Common Voice 27.0 German: select clips with rarely heard
words and write them out as the corpus 'cv_rare_de'.

The archive (37.4 GB, ~1 million MP3s) is NOT unpacked - on virtiofs that would
have blown the host's file descriptor limit. Instead:
  1. Selection from the tables (validated.tsv, clip_durations.tsv): cleanly
     validated (>= 2 up votes, 0 down votes), 2.5-10.2 s, contains a rare word.
     Generous: twice the demand - a stream cannot be rewound.
  2. One pass through the archive; candidates are decoded in memory, checked,
     and written as WAV only if they are still needed.

Common Voice is recorded with consumer microphones, hence stricter than for
Thorsten: SNR >= 20 dB, hum < 10 dB, AND band edge >= 10 kHz - otherwise
telephone and headset recordings with a muffled sound get in.

Goal: QUOTE examples per word in total, together with what Thorsten and CML
already provide. At most PRO_SPRECHER clips per speaker - Common Voice has
prolific speakers with tens of thousands of recordings.

Layout of CV_DIR (see de_lora/core/paths.py):
  tsv/validated.tsv, tsv/clip_durations.tsv   extracted from the archive
  teil.0 ... teil.3                           the .tar.gz split into four byte
                                              ranges ("Teil" = part); streamed
                                              through `cat ... | pigz -dc` (needs pigz)
Options (German names): --quote / --quote-mittel = examples per word for
Zipf >= --zipf-hoch / below it, --min-zipf, --pro-sprecher = clips per speaker,
--stunden = hours budget, --min-kante = minimum band edge in Hz,
--nur-auswahl = only run the table selection, --worker = processes.
Reads seltene_woerter.json and rare_de_abdeckung.json from BREEZE_WORDS_DIR and
writes rare_de_abdeckung_mit_cv.json there. The corpus is registered DISABLED.
The final lines start with "[DONE] Common Voice" (recipes/pipeline_v3b.sh waits
for that).
"""
from __future__ import annotations
import argparse, collections, csv, io, json, re, subprocess, sys, tarfile, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "de_lora" / "core"))
import paths  # noqa: E402  (all locations are configurable, see de_lora/core/paths.py)

CV = paths.CV_DIR
WORT = re.compile(r"[A-Za-zÄÖÜäöüß]+")
csv.field_size_limit(10**9)


# Band edge ("Kante"): the highest frequency whose level is at most 35 dB below
# the mean level of the 300-3000 Hz speech band. Catches the low-pass edge of
# lossy codecs and narrow-band microphones rather than the natural roll-off.
def kante_hz(w, sr):
    n, hop = 8192, 4096
    anz = (len(w) - n) // hop
    if anz < 2:
        return 0.0
    S = np.abs(np.fft.rfft(np.stack([w[i*hop:i*hop+n] * np.hanning(n) for i in range(anz)]), axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    e = S.sum(axis=1); idx = np.argsort(e)
    sprache = S[idx[max(1, anz // 2):]].mean(axis=0)
    band = sprache[(f >= 300) & (f <= 3000)].mean() + 1e-20
    db = 10 * np.log10(sprache / band + 1e-20)
    ueber = np.where(db > -35)[0]
    return float(f[ueber[-1]]) if len(ueber) else 0.0



def _pruefe(auftrag):
    """Worker: decode, check, trim, convert to 24 kHz.
    Returns (name, reason_or_None, int16 bytes, seconds, measures).
    Rejection reasons: dekodieren = decode error, snr, brumm = hum,
    kante = band edge, kurz = too short after trimming."""
    import soundfile as sf, librosa
    sys.path.insert(0, str(REPO / "de_lora/data"))
    from build_corpus_hui import signal_masse
    name, daten, min_kante = auftrag
    try:
        w, sr = sf.read(io.BytesIO(daten), dtype="float32", always_2d=True)
        w = w.mean(axis=1)
    except Exception:                                  # noqa: BLE001
        return name, "dekodieren", None, 0.0, None
    snr, brumm = signal_masse(w, sr)
    ka = kante_hz(w, sr)
    m = {"snr": round(snr, 1), "brumm": round(brumm, 1), "kante": round(ka)}
    if snr < 20.0: return name, "snr", None, 0.0, m
    if brumm > 10.0: return name, "brumm", None, 0.0, m
    if ka < min_kante: return name, "kante", None, 0.0, m
    w, _ = librosa.effects.trim(w, top_db=40)
    if len(w) / sr < 1.5: return name, "kurz", None, 0.0, m
    w24 = librosa.resample(w, orig_sr=sr, target_sr=24000)
    pk = float(np.max(np.abs(w24)))
    w24 = (w24 * (0.95 / max(pk, 1e-9))).clip(-1, 1)
    return name, None, (w24 * 32767).astype("<i2").tobytes(), len(w24) / 24000, m


def main() -> None:
    ap = argparse.ArgumentParser()
    # Staggered by present-day frequency. Without staggering (all 23,000 words
    # from Zipf 3.0, 50 each) the pre-selection alone came to 396 h - almost
    # every sentence contains some rare word. Consistent with the re-weighting
    # in training, which also only applies from Zipf 4.0.
    ap.add_argument("--quote", type=int, default=50, help="examples per word with Zipf >= --zipf-hoch")
    ap.add_argument("--quote-mittel", type=int, default=20, help="examples per word with --min-zipf <= Zipf < --zipf-hoch")
    ap.add_argument("--zipf-hoch", type=float, default=4.0)
    ap.add_argument("--min-zipf", type=float, default=3.5)
    ap.add_argument("--pro-sprecher", type=int, default=150)
    ap.add_argument("--stunden", type=float, default=60.0)
    ap.add_argument("--min-kante", type=float, default=10000.0)
    ap.add_argument("--nur-auswahl", action="store_true")
    ap.add_argument("--worker", type=int, default=8)
    args = ap.parse_args()

    ziele = json.load(open(paths.WORDS_DIR / "seltene_woerter.json", encoding="utf-8"))
    ziele = [r for r in ziele if r["zipf"] >= args.min_zipf]
    prio = {r["wort"].casefold(): i for i, r in enumerate(ziele)}
    quote = {r["wort"].casefold(): (args.quote if r["zipf"] >= args.zipf_hoch else args.quote_mittel)
             for r in ziele}
    schon = json.load(open(paths.WORDS_DIR / "rare_de_abdeckung.json", encoding="utf-8"))  # Thorsten+CML
    bedarf = {w: max(0, quote[w] - schon.get(w, 0)) for w in prio}

    dauer = {}
    with (CV / "tsv/clip_durations.tsv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            dauer[r["clip"]] = int(r["duration[ms]"]) / 1000
    kand, n_valid = [], 0
    with (CV / "tsv/validated.tsv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            n_valid += 1
            try:
                ja, nein = int(r["up_votes"]), int(r["down_votes"])
            except (ValueError, KeyError):
                continue
            d = dauer.get(r["path"], 0.0)
            if nein != 0 or ja < 2 or not (2.5 <= d <= 10.2):
                continue
            z = sorted({w.casefold() for w in WORT.findall(r["sentence"])} & prio.keys(), key=prio.get)
            if z:
                kand.append(dict(path=r["path"], text=r["sentence"], spk=r["client_id"],
                                 dur=d, ziele=z))
    print(f"[selection] validated.tsv: {n_valid} clips, {len(kand)} with rare words "
          f"(cleanly validated, 2.5-10.2 s)", flush=True)

    # Pre-selection: up to TWICE the demand per word, so that enough is left after
    # the quality losses; a per-speaker cap for diversity.
    idx = collections.defaultdict(list)
    for j, k in enumerate(kand):
        for w in k["ziele"]:
            idx[w].append(j)
    vor, je_spk, reserve = set(), collections.Counter(), dict(bedarf)
    for w in sorted(idx, key=prio.get):
        brauch = 2 * reserve[w]
        for j in idx[w]:
            if brauch <= 0:
                break
            if j in vor or je_spk[kand[j]["spk"]] >= args.pro_sprecher:
                continue
            vor.add(j); je_spk[kand[j]["spk"]] += 1
            for v in kand[j]["ziele"]:
                reserve[v] -= 1
            brauch -= 1
    std = sum(kand[j]["dur"] for j in vor) / 3600
    wirkt = sum(1 for w in bedarf if bedarf[w] > 0 and any(True for j in idx.get(w, []) if j in vor))
    print(f"[selection] pre-selection {len(vor)} clips, {std:.1f} h, covers {wirkt} words with open demand, "
          f"{len(je_spk)} speakers", flush=True)
    if args.nur_auswahl:
        return

    # ----- pass 2: stream through the archive once
    import soundfile as sf, librosa
    sys.path.insert(0, str(REPO / "de_lora/data")); sys.path.insert(0, str(REPO / "de_lora/core"))
    from build_corpus import clean_text
    from build_corpus_hui import signal_masse
    gesucht = {kand[j]["path"]: kand[j] for j in vor}
    out = paths.DATASETS_ROOT / "cv_rare_de"; (out / "wavs").mkdir(parents=True, exist_ok=True)
    tf = (out / "texts.csv").open("w", encoding="utf-8")
    sp = (out / "speakers.jsonl").open("w", encoding="utf-8")
    qf = (out / "qualitaet.jsonl").open("w", encoding="utf-8")
    rest = dict(bedarf)
    verw, n, sek, gesehen, t0 = collections.Counter(), 0, 0.0, 0, time.time()
    cat = subprocess.Popen(f"cat {CV}/teil.0 {CV}/teil.1 {CV}/teil.2 {CV}/teil.3 | pigz -dc",
                           shell=True, stdout=subprocess.PIPE, bufsize=16 * 1024 * 1024)
    import concurrent.futures as cf, multiprocessing as mp
    import soundfile as sf
    offen = {}

    def ernte(fertig_bis):
        """Collect results ("ernten" = harvest) and decide here - in the main
        process: only the main process tracks the remaining demand, so that the
        bookkeeping stays correct. Extra reasons: budget = hours budget used up,
        gedeckt = word demand already covered."""
        nonlocal n, sek
        while len(offen) > fertig_bis:
            fut = next(cf.as_completed(list(offen)))
            k = offen.pop(fut)
            name, grund, roh, dauer_s, m = fut.result()
            if m is not None:
                qf.write(json.dumps({"path": name, **m}) + "\n")
            if grund:
                verw[grund] += 1; continue
            if sek >= args.stunden * 3600:
                verw["budget"] += 1; continue
            if not any(rest.get(w, 0) > 0 for w in k["ziele"]):
                verw["gedeckt"] += 1; continue
            text = clean_text(k["text"], dauer_s, "cv_rare_de")
            if not text:
                verw["text"] += 1; continue
            i = f"cvr-{n:06d}"
            sf.write(str(out / "wavs" / f"{i}.wav"), np.frombuffer(roh, dtype="<i2"), 24000,
                     subtype="PCM_16")
            tf.write(f"wavs/{i}.wav=={text}\n")
            sp.write(json.dumps({"idx": i, "speaker": f"cv:{k['spk']}"}) + "\n")
            n += 1; sek += dauer_s
            for v in k["ziele"]:
                if v in rest: rest[v] -= 1

    with cf.ProcessPoolExecutor(max_workers=args.worker, mp_context=mp.get_context("spawn")) as ex, \
         tarfile.open(fileobj=cat.stdout, mode="r|") as tar:
        for m in tar:
            gesehen += 1
            if gesehen % 100000 == 0:
                print(f"[stream] {gesehen} entries, {n} written ({sek/3600:.1f} h), "
                      f"discarded {dict(verw)}, {time.time()-t0:.0f} s", flush=True)
            name = m.name.rsplit("/", 1)[-1]
            k = gesucht.get(name)
            if k is None or not m.isfile():
                continue
            # already covered -> do not even decode
            if not any(rest.get(w, 0) > 0 for w in k["ziele"]):
                verw["gedeckt"] += 1; continue
            offen[ex.submit(_pruefe, (name, tar.extractfile(m).read(), args.min_kante))] = k
            if len(offen) >= 4 * args.worker:
                ernte(2 * args.worker)
        ernte(0)
    cat.wait()
    tf.close(); sp.close(); qf.close()

    cj = paths.CORPORA_JSON
    d = json.loads(cj.read_text(encoding="utf-8"))
    d = [c for c in d if c["name"] != "cv_rare_de"]
    d.append({"name": "cv_rare_de", "root": "cv_rare_de", "csv": "texts.csv",
              "speakers": "speakers.jsonl", "enabled": False})
    cj.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    gesamt = {w: schon.get(w, 0) + (bedarf[w] - rest[w]) for w in bedarf}
    json.dump(gesamt, open(paths.WORDS_DIR / "rare_de_abdeckung_mit_cv.json", "w", encoding="utf-8"), ensure_ascii=False)
    top = sorted(prio, key=prio.get)[:300]
    v = [gesamt[w] for w in top]
    # NOTE: recipes/pipeline_v3b.sh waits for a line starting with "[DONE] Common Voice".
    print(f"\n[DONE] Common Voice: {n} clips, {sek/3600:.1f} h, discarded {dict(verw)}")
    print(f"[DONE] 300 most frequent rare words, examples in total: 0: {sum(x==0 for x in v)}  "
          f"1-9: {sum(1<=x<10 for x in v)}  10-29: {sum(10<=x<30 for x in v)}  "
          f"30-49: {sum(30<=x<50 for x in v)}  50+: {sum(x>=50 for x in v)}")
    for w in ("online", "internet", "temperaturen", "telefon", "informationen", "projekt", "video", "super"):
        print(f"    {w:<14} {schon.get(w,0):>3} before  ->  {gesamt.get(w,0):>3} total")


if __name__ == "__main__":
    main()
