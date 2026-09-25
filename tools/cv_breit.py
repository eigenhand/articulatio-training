"""cv_breit.py — Common Voice 27.0 German in breadth: the corpus 'cv_breit_de'.

("breit" = broad.) cv_rare_de only took clips with rarely heard words (60 h).
This adds the rest, as far as it passes the quality filters - but at most
PRO_SPRECHER clips per speaker. Common Voice has prolific speakers with tens of
thousands of recordings (the largest has 56,886, the ten largest 22 %); without
a cap the model would sound like them afterwards.

Same filters as cv_seltene_woerter.py (_pruefe is imported from there): cleanly
validated (>= 2 up votes, 0 down votes), 2.5-10.2 s, SNR >= 20 dB, hum < 10 dB,
band edge >= 10 kHz. Clips that the rare-word pass has already decoded (its
qualitaet.jsonl) stay out: they are either already in training or failed the
filters.

As there, the archive is streamed only once and nothing is unpacked. A few more
candidates than the cap are reserved per speaker, because about 40 % fail the
filters; once a speaker is full, the rest is not even decoded.

WAVs go to disk in ONE write(). soundfile writes in small blocks, and on
virtiofs every write call costs a round trip - that was the cause of the
snail's pace of the first pass.

Options (German names): --pro-sprecher = clips per speaker (default 100),
--vorrat = candidates reserved per slot, --min-kante = minimum band edge in Hz,
--nur-auswahl = only run the table selection, --worker = processes.
Layout of CV_DIR: see cv_seltene_woerter.py. The corpus is registered DISABLED.
The final line starts with "[DONE] cv_breit_de" (recipes/pipeline_v4.sh waits
for that).
"""
from __future__ import annotations
import argparse, collections, csv, io, json, random, subprocess, sys, tarfile, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "de_lora" / "core"))
import paths  # noqa: E402  (all locations are configurable, see de_lora/core/paths.py)

CV = paths.CV_DIR
NAME = "cv_breit_de"          # must end in _de: otherwise numbers are spelled out in Portuguese
csv.field_size_limit(10**9)
sys.path.insert(0, str(REPO / "tools"))
from cv_seltene_woerter import _pruefe  # noqa: E402  (same filters, works with spawn)


def wav_bytes(pcm: bytes) -> bytes:
    import wave
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(pcm)
    return b.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pro-sprecher", type=int, default=100)
    ap.add_argument("--vorrat", type=float, default=1.9,
                    help="candidates reserved per slot (filter loss ~40 %%)")
    ap.add_argument("--min-kante", type=float, default=10000.0)
    ap.add_argument("--worker", type=int, default=10)
    ap.add_argument("--nur-auswahl", action="store_true")
    args = ap.parse_args()

    # ----- pass 1: selection from the tables
    schon = set()
    qpfad = paths.DATASETS_ROOT / "cv_rare_de/qualitaet.jsonl"
    for zeile in qpfad.open(encoding="utf-8"):
        schon.add(json.loads(zeile)["path"])
    dauer = {}
    with (CV / "tsv/clip_durations.tsv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            dauer[r["clip"]] = int(r["duration[ms]"]) / 1000
    je_spk = collections.defaultdict(list)
    with (CV / "tsv/validated.tsv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            try:
                ja, nein = int(r["up_votes"]), int(r["down_votes"])
            except (ValueError, KeyError):
                continue
            d = dauer.get(r["path"], 0.0)
            if nein != 0 or ja < 2 or not (2.5 <= d <= 10.2) or r["path"] in schon:
                continue
            je_spk[r["client_id"]].append(dict(path=r["path"], text=r["sentence"],
                                               spk=r["client_id"], dur=d))
    rnd = random.Random(27)
    gesucht, plaetze = {}, int(args.pro_sprecher * args.vorrat + 0.5)
    for spk, clips in je_spk.items():
        rnd.shuffle(clips)
        for k in clips[:plaetze]:
            gesucht[k["path"]] = k
    std = sum(k["dur"] for k in gesucht.values()) / 3600
    print(f"[selection] {len(je_spk)} speakers, {len(gesucht)} candidates ({std:.0f} h) reserved, "
          f"{len(schon)} left out from the rare-word pass", flush=True)
    if args.nur_auswahl:
        return

    # ----- pass 2: stream through the archive once
    sys.path.insert(0, str(REPO / "de_lora/data")); sys.path.insert(0, str(REPO / "de_lora/core"))
    from build_corpus import clean_text
    out = paths.DATASETS_ROOT / NAME
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    tf = (out / "texts.csv").open("w", encoding="utf-8")
    sp = (out / "speakers.jsonl").open("w", encoding="utf-8")
    qf = (out / "qualitaet.jsonl").open("w", encoding="utf-8")
    angenommen = collections.Counter()
    verw, n, sek, gesehen, t0 = collections.Counter(), 0, 0.0, 0, time.time()
    cat = subprocess.Popen(f"cat {CV}/teil.0 {CV}/teil.1 {CV}/teil.2 {CV}/teil.3 | pigz -dc",
                           shell=True, stdout=subprocess.PIPE, bufsize=16 * 1024 * 1024)
    import concurrent.futures as cf, multiprocessing as mp
    offen = {}

    # Collects worker results in the main process; "voll" = speaker already full.
    def ernte(fertig_bis):
        nonlocal n, sek
        while len(offen) > fertig_bis:
            fut = next(cf.as_completed(list(offen)))
            k = offen.pop(fut)
            name, grund, roh, dauer_s, m = fut.result()
            if m is not None:
                qf.write(json.dumps({"path": name, **m}) + "\n")
            if grund:
                verw[grund] += 1; continue
            if angenommen[k["spk"]] >= args.pro_sprecher:
                verw["voll"] += 1; continue
            text = clean_text(k["text"], dauer_s, NAME)
            if not text:
                verw["text"] += 1; continue
            i = f"cvb-{n:06d}"
            with open(out / "wavs" / f"{i}.wav", "wb", buffering=0) as f:
                f.write(wav_bytes(roh))
            tf.write(f"wavs/{i}.wav=={text}\n")
            sp.write(json.dumps({"idx": i, "speaker": f"cv:{k['spk']}"}) + "\n")
            angenommen[k["spk"]] += 1
            n += 1; sek += dauer_s

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
            if angenommen[k["spk"]] >= args.pro_sprecher:     # full: no more decoding
                verw["voll"] += 1; continue
            offen[ex.submit(_pruefe, (name, tar.extractfile(m).read(), args.min_kante))] = k
            if len(offen) >= 4 * args.worker:
                ernte(2 * args.worker)
        ernte(0)
    cat.wait()
    tf.close(); sp.close(); qf.close()

    cj = paths.CORPORA_JSON
    d = json.loads(cj.read_text(encoding="utf-8"))
    d = [c for c in d if c["name"] != NAME]
    d.append({"name": NAME, "root": NAME, "csv": "texts.csv", "speakers": "speakers.jsonl", "enabled": False})
    cj.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    voll = sum(1 for v in angenommen.values() if v >= args.pro_sprecher)
    # NOTE: recipes/pipeline_v4.sh waits for a line starting with "[DONE] cv_breit_de".
    print(f"\n[DONE] {NAME}: {n} clips, {sek/3600:.1f} h, {len(angenommen)} speakers "
          f"({voll} at the cap), discarded {dict(verw)}, {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
