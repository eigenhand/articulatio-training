"""suche_seltene_woerter.py — find clips with rarely heard words in the permitted
corpora and write them out as additional corpora (thorsten_rare_de, cml_rare_de).

("suche seltene Woerter" = search for rare words.)
Sources (the licences permit non-commercial publication; Standard German):
  thorsten  TV-2021.02-Neutral + TV-2022.10-Neutral (CC0). Modern texts, hence
            the best source of everyday words. NOT Emotional (acted drunk/
            whispering) and NOT Hessisch (dialect).
  cml       CML-TTS German complete, train+dev+test (CC BY 4.0).
HUI is exhausted - everything suitable is already in run 2.

Greedy selection by priority: words by present-day frequency (Zipf),
descending; up to QUOTE new clips per word; a clip counts for ALL rare words
it contains. Thorsten before CML.

Quality: CML as before, SNR >= 20 dB (quietest 20 % of the frames count as
noise) and hum < 10 dB. For Thorsten this measure is useless - Thorsten speaks
without pauses, so the measure took speech for noise (18 dB measured, versus
~89 dB in the real silence before the sentence). Thorsten therefore only goes
through the hum test.

  --quote N        target clips per word (default 30)
  --cml-stunden H  cap on CML hours (real run)
  --trocken        dry run: only count, do not touch any audio; the selection
                   is limited by --stunden hours and written to rare_de_auswahl.json

Reads seltene_woerter.json and grossschreibung.json from BREEZE_WORDS_DIR and
writes rare_de_abdeckung.json (clips per word) there. The new corpora are
registered in corpora.json but DISABLED. The final lines start with "[DONE]"
(recipes/pipeline_v3.sh waits for that).
"""
from __future__ import annotations
import argparse, collections, csv, glob, hashlib, io, json, re, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "de_lora" / "core"))
import paths  # noqa: E402  (all locations are configurable, see de_lora/core/paths.py)

CML = paths.CML_DE_ROOT
THORSTEN = [str(paths.THORSTEN_DIR / "TV-2021.02-Neutral" / "train-*.parquet"),
            str(paths.THORSTEN_DIR / "TV-2022.10-Neutral" / "train-*.parquet")]
WORT = re.compile(r"[A-Za-zÄÖÜäöüß]+")
MIN_DUR, MAX_DUR = 4.0, 10.2
# Thorsten speaks short sentences (median 3.2 s): with a 4 s lower limit, three
# quarters of his recordings would drop out - precisely the ones with modern
# vocabulary. The pipeline copes with anything from ~1 s; 4 s was a choice of the
# pt-BR template.
MIN_DUR_THORSTEN = 2.5


def lade_pool():
    """-> list of dict(quelle=source, key, dur, text, spk, ort=location)"""
    import pyarrow.parquet as pq
    pool = []
    for muster in THORSTEN:
        for fp in sorted(glob.glob(muster)):
            t = pq.read_table(fp, columns=["id", "text", "durationSeconds"]).to_pydict()
            for row, (i, tx, d) in enumerate(zip(t["id"], t["text"], t["durationSeconds"])):
                d = float(d)
                if MIN_DUR_THORSTEN <= d <= MAX_DUR and tx:
                    pool.append(dict(quelle="thorsten", key=str(i), dur=d, text=tx,
                                     spk="thorsten", ort=(fp, row)))
    for split in ("train", "dev", "test"):
        with (CML / f"{split}.csv").open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh, delimiter="|"):
                try:
                    d = float(r["duration"]); lev = float(r.get("levenshtein") or 0)
                except ValueError:
                    continue
                if MIN_DUR <= d <= MAX_DUR and lev >= 0.95:
                    pool.append(dict(quelle="cml", key=r["wav_filename"], dur=d,
                                     text=r["transcript"], spk=f"cmlde:{r['client_id']}",
                                     ort=r["wav_filename"]))
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quote", type=int, default=30)
    ap.add_argument("--stunden", type=float, default=50.0)
    ap.add_argument("--trocken", action="store_true")
    ap.add_argument("--cml-stunden", type=float, default=25.0)
    args = ap.parse_args()

    ziele = json.load(open(paths.WORDS_DIR / "seltene_woerter.json", encoding="utf-8"))
    # casefold on both sides: ß == ss, otherwise "außerdem" finds nothing
    prio = {r["wort"].casefold(): i for i, r in enumerate(ziele)}   # sorted by Zipf
    info = {r["wort"].casefold(): r for r in ziele}

    pool = lade_pool()
    print(f"[pool] {sum(p['quelle']=='thorsten' for p in pool)} Thorsten clips, "
          f"{sum(p['quelle']=='cml' for p in pool)} CML clips within the length window", flush=True)

    index = collections.defaultdict(list)
    for j, p in enumerate(pool):
        p["ziele"] = sorted({w.casefold() for w in WORT.findall(p["text"])} & prio.keys(), key=prio.get)
        for w in p["ziele"]:
            index[w].append(j)
    gefunden = [w for w in prio if index.get(w)]
    print(f"[index] {len(gefunden)} of {len(prio)} rare words occur in the sources", flush=True)

    # candidates per word: Thorsten first, then CML in a stable shuffled order
    def ordnung(j):
        p = pool[j]
        return (0 if p["quelle"] == "thorsten" else 1,
                hashlib.sha1(p["key"].encode()).hexdigest())
    for w in index:
        index[w].sort(key=ordnung)

    if not args.trocken:
        schreibe(args, pool, index, prio, info)
        return

    # greedy selection by priority (without audio: an upper bound of the yield)
    bedarf = {w: args.quote for w in gefunden}
    gewaehlt, gesehen, total = [], set(), 0.0
    for w in sorted(gefunden, key=prio.get):
        if total >= args.stunden * 3600: break
        for j in index[w]:
            if bedarf[w] <= 0 or total >= args.stunden * 3600: break
            if j in gesehen: continue
            gesehen.add(j); gewaehlt.append(j); total += pool[j]["dur"]
            for v in pool[j]["ziele"]:
                if v in bedarf: bedarf[v] -= 1

    q = collections.Counter(pool[j]["quelle"] for j in gewaehlt)
    abgedeckt = sum(1 for w in gefunden if bedarf[w] <= 0)
    print(f"[selection] {len(gewaehlt)} clips, {total/3600:.1f} h  "
          f"(Thorsten {q['thorsten']}, CML {q['cml']})")
    print(f"[selection] quota {args.quote} fully reached for {abgedeckt} words, "
          f"partially for {sum(1 for w in gefunden if 0 < bedarf[w] < args.quote)}")
    print("\nsample (present-day frequency, descending): hits in the sources")
    for w in list(prio)[:25]:
        n = len(index.get(w, [])); th = sum(1 for j in index.get(w, []) if pool[j]["quelle"] == "thorsten")
        print(f"  {info[w]['wort']:<18} so far {info[w]['training']:>2}x  ->  {n:>5} hits (Thorsten {th})")
    for w in ("temperaturen", "telefon", "internet", "informationen", "bahnhof"):
        if w in prio or w == "bahnhof":
            n = len(index.get(w, []))
            print(f"  {w:<18} {'':>13}{n:>5} hits")
    if args.trocken:
        json.dump({"gewaehlt": [pool[j] | {"ort": list(pool[j]["ort"]) if isinstance(pool[j]["ort"], tuple) else pool[j]["ort"]}
                                for j in gewaehlt]},
                  open(paths.WORDS_DIR / "rare_de_auswahl.json", "w", encoding="utf-8"), ensure_ascii=False)
        return



# ------------------------------------------------------------------ real run
def schreibe(args, pool, index, prio, info) -> None:
    """Greedy selection WITH audio: every candidate is decoded and checked
    before it counts. Two target corpora, so that training can weight them
    separately - CML is band-limited to ~10 kHz, Thorsten is not."""
    import io, librosa, soundfile as sf, pyarrow.parquet as pq
    sys.path.insert(0, str(REPO / "de_lora" / "data"))
    sys.path.insert(0, str(REPO / "de_lora" / "core"))
    from build_corpus import clean_text
    from build_corpus_hui import signal_masse
    sys.path.insert(0, str(REPO / "tools"))
    from grossschreibung import richte
    tabelle = json.load(open(paths.WORDS_DIR / "grossschreibung.json", encoding="utf-8"))

    # Corpus names must end in _de: otherwise clean_text spells numbers out in
    # Portuguese (inherited from the pt-BR template).
    ziel = {"thorsten": "thorsten_rare_de", "cml": "cml_rare_de"}
    ds = paths.DATASETS_ROOT
    fh = {}
    for q, name in ziel.items():
        (ds / name / "wavs").mkdir(parents=True, exist_ok=True)
        fh[q] = ((ds / name / "texts.csv").open("w", encoding="utf-8"),
                 (ds / name / "speakers.jsonl").open("w", encoding="utf-8"))

    th_cache = {}
    def audio(p):
        if p["quelle"] == "thorsten":
            fp, row = p["ort"]
            if fp not in th_cache:
                th_cache[fp] = pq.read_table(fp, columns=["audio"]).column("audio").to_pylist()
            b = th_cache[fp][row]["bytes"]
            w, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
        else:
            w, sr = sf.read(str(CML / p["ort"]), dtype="float32", always_2d=True)
        return w.mean(axis=1), sr

    bedarf = {w: args.quote for w in index}
    stunden = collections.Counter()
    zahl = collections.Counter()
    # rejection counters: dekodieren = decode error, brumm = hum, snr, text, leer = silent
    verworfen = collections.Counter()
    geprueft = set()
    n_ok = 0
    for w in sorted(index, key=prio.get):
        if bedarf[w] <= 0:
            continue
        for j in index[w]:
            if bedarf[w] <= 0:
                break
            if j in geprueft:
                continue
            geprueft.add(j)
            p = pool[j]
            q = p["quelle"]
            if q == "cml" and stunden["cml"] >= args.cml_stunden * 3600:
                continue
            try:
                wav, sr = audio(p)
            except Exception:                                   # noqa: BLE001
                verworfen["dekodieren"] += 1
                continue
            snr, brumm = signal_masse(wav, sr)
            if brumm > 10.0:
                verworfen["brumm"] += 1
                continue
            if q == "cml" and snr < 20.0:        # the measure is useless for Thorsten
                verworfen["snr"] += 1
                continue
            text = clean_text(richte(p["text"], tabelle), len(wav) / sr, ziel[q])
            if not text:
                verworfen["text"] += 1
                continue
            w24 = librosa.resample(wav, orig_sr=sr, target_sr=24000) if sr != 24000 else wav
            pk = float(np.max(np.abs(w24)))
            if pk < 1e-4:
                verworfen["leer"] += 1
                continue
            w24 = (w24 * (0.95 / pk)).clip(-1, 1).astype("float32")
            idx = f"{'thr' if q == 'thorsten' else 'cmr'}-{zahl[q]:06d}"
            sf.write(str(ds / ziel[q] / "wavs" / f"{idx}.wav"), w24, 24000, subtype="PCM_16")
            fh[q][0].write(f"wavs/{idx}.wav=={text}\n")
            fh[q][1].write(json.dumps({"idx": idx, "speaker": p["spk"]}) + "\n")
            zahl[q] += 1
            stunden[q] += len(w24) / 24000
            for v in p["ziele"]:
                if v in bedarf:
                    bedarf[v] -= 1
            n_ok += 1
            if n_ok % 1000 == 0:
                for a, b in fh.values():
                    a.flush(); b.flush()
                print(f"[write] {n_ok} clips  Thorsten {stunden['thorsten']/3600:.1f} h  "
                      f"CML {stunden['cml']/3600:.1f} h  discarded {dict(verworfen)}", flush=True)
    for a, b in fh.values():
        a.close(); b.close()

    # Register the corpora, but DISABLED - only run 3 switches them on.
    cj = ds / "corpora.json"
    daten = json.loads(cj.read_text(encoding="utf-8"))
    daten = [c for c in daten if c["name"] not in ziel.values()]
    for name in ziel.values():
        daten.append({"name": name, "root": name, "csv": "texts.csv",
                      "speakers": "speakers.jsonl", "enabled": False})
    cj.write_text(json.dumps(daten, ensure_ascii=False, indent=2), encoding="utf-8")

    # NOTE: recipes/pipeline_v3.sh waits for lines starting with "[DONE]".
    print(f"\n[DONE] Thorsten {zahl['thorsten']} clips / {stunden['thorsten']/3600:.1f} h, "
          f"CML {zahl['cml']} clips / {stunden['cml']/3600:.1f} h")
    print(f"[DONE] discarded: {dict(verworfen)}")
    erreicht = {w: args.quote - bedarf[w] for w in bedarf}
    top = sorted(info, key=lambda w: -info[w]["zipf"])[:300]
    v = [erreicht.get(w, 0) for w in top]
    print(f"[DONE] 300 most frequent rare words: 0: {sum(x==0 for x in v)}  "
          f"1-9: {sum(1<=x<10 for x in v)}  10-29: {sum(10<=x<30 for x in v)}  30+: {sum(x>=30 for x in v)}")
    json.dump(erreicht, open(paths.WORDS_DIR / "rare_de_abdeckung.json", "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
