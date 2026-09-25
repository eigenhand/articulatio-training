"""analyse_seltene_woerter.py — Which everyday words has the model hardly ever heard?

("seltene Woerter" = rare words.) Measures per word:
  zipf        frequency in present-day German (wordfreq; 3 = once per million
              words, 5 = very common). Separates what a chat assistant will say
              from audiobook rarities such as "Reservedame".
  training    how often the word occurs in the training transcripts (CML from
              run 1 + HUI from run 2).
  min_token   frequency of the word's RAREST Gemma token in training.
              Learning happens per token, not per word: "Temperaturen" is a
              single token whose sound the model has to know from audio;
              "Strassenbahnhaltestelle" splits into frequent pieces and is
              therefore uncritical, although the word is rare.
  betonung    (stress) syllable of the main stress according to espeak-ng.
              German mostly stresses the first syllable; stress further back
              (Temperaˈturen, Informaˈtion) is where an uncertain model goes wrong.

Output in BREEZE_WORDS_DIR (default <PTBR_ARTIFACTS>, see de_lora/core/paths.py):
  seltene_woerter.json  list sorted by Zipf (descending) with the keys
                        wort (word), zipf, training, min_token, tokens, ipa,
                        silben (syllables), ton (stressed syllable, 0-based),
                        ton_hinten (stress not on the first syllable); read by
                        suche_seltene_woerter.py, cv_seltene_woerter.py and
                        train_lora.py --wort-ausgleich
  seltene_woerter.tsv   the same as a table
Requires espeak-ng on PATH and the wordfreq package.
"""
from __future__ import annotations

import collections
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer
from wordfreq import top_n_list, zipf_frequency

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "de_lora" / "core"))
import paths  # noqa: E402  (all locations are configurable, see de_lora/core/paths.py)

OUT = paths.WORDS_DIR
TRAIN_CSV = [paths.DATASETS_ROOT / "cml_de/texts.csv",
             paths.DATASETS_ROOT / "hui_de/texts.csv"]
WORT = re.compile(r"[A-Za-zÄÖÜäöüß]+")
VOKALE = set("aeiouyɛɪɔʊəɐœøʏɑæɜɒʌɨ")

MIN_ZIPF = 3.0          # everyday: at least ~1x per million words
MAX_TRAIN = 5           # "rarely heard"
MAX_TOKEN = 5           # rarest token heard at most this often
MIN_LEN = 5


def silben_und_ton(ipa: str) -> tuple[int, int]:
    """(number of syllables, syllable of the main stress counted from 0) from espeak IPA."""
    silben, ton, in_vokal = 0, -1, False
    for ch in ipa:
        if ch == "ˈ":
            ton = silben
        if ch in VOKALE:
            if not in_vokal:
                silben += 1
            in_vokal = True
        elif ch not in "ː̃ˈˌ":
            in_vokal = False
    return silben, max(ton, 0)


def espeak_batch(woerter: list[str]) -> dict[str, str]:
    """One line per word. A single call for all of them - starting one process
    per word would be needlessly slow for a few thousand words."""
    text = "\n".join(woerter)
    out = subprocess.run(["espeak-ng", "-v", "de", "-q", "--ipa", "--stdin"],
                         input=text, capture_output=True, text=True).stdout
    zeilen = [z.strip() for z in out.splitlines() if z.strip()]
    if len(zeilen) != len(woerter):          # lines not 1:1 -> one call per word
        return {w: subprocess.run(["espeak-ng", "-v", "de", "-q", "--ipa", w],
                                  capture_output=True, text=True).stdout.strip()
                for w in woerter}
    return dict(zip(woerter, zeilen))


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(paths.CKPT))

    # --- 1) what has the model heard?
    wort_train = collections.Counter()
    token_train = collections.Counter()
    schreibung = collections.defaultdict(collections.Counter)
    saetze = []
    for p in TRAIN_CSV:
        for ln in p.read_text(encoding="utf-8").splitlines():
            if "==" not in ln:
                continue
            t = ln.split("==", 1)[1]
            saetze.append(t)
            for w in WORT.findall(t):
                # casefold instead of lower: folds ß to ss, exactly like wordfreq.
                # With lower(), "außerdem" was looked up as "ausserdem", not
                # found, and wrongly listed as rare.
                wort_train[w.casefold()] += 1
                schreibung[w.casefold()][w] += 1
    for i in range(0, len(saetze), 2000):
        for ids in tok(saetze[i:i + 2000], add_special_tokens=False)["input_ids"]:
            token_train.update(ids)
    import csv as _csv, glob as _glob
    import pyarrow.parquet as _pq
    quelltexte = []
    for split in ("train", "dev", "test"):
        with open(paths.CML_DE_ROOT / f"{split}.csv",
                  encoding="utf-8", newline="") as fh:
            quelltexte += [r["transcript"] for r in _csv.DictReader(fh, delimiter="|")]
    for fp in (_glob.glob(str(paths.THORSTEN_DIR / "TV-2021.02-Neutral" / "train-*.parquet"))
               + _glob.glob(str(paths.THORSTEN_DIR / "TV-2022.10-Neutral" / "train-*.parquet"))):
        quelltexte += [t for t in _pq.read_table(fp, columns=["text"]).column("text").to_pylist() if t]
    for t in quelltexte:
        for w in WORT.findall(t):
            schreibung[w.casefold()][w] += 1       # spelling only, does not count as heard
    print(f"[1] training: {len(saetze)} sentences, {len(wort_train)} word forms, "
          f"{len(token_train)} distinct tokens", flush=True)

    # --- 2) what do people say today?
    kand = [w for w in top_n_list("de", 200000)
            if len(w) >= MIN_LEN and WORT.fullmatch(w) and zipf_frequency(w, "de") >= MIN_ZIPF]
    print(f"[2] everyday vocabulary (Zipf >= {MIN_ZIPF}): {len(kand)} words", flush=True)

    # --- 3) rarely heard, as a word or via its rarest token
    risiko = []
    for w in kand:
        # wordfreq returns folded forms ("ausserdem"); the real spelling with ß
        # and capitalization comes from the training data + source corpora. Only
        # if the word appears nowhere there does the folded form remain.
        form = schreibung[w].most_common(1)[0][0] if schreibung[w] else w
        ids = tok(" " + form, add_special_tokens=False)["input_ids"]
        mt = min(token_train.get(i, 0) for i in ids)
        n = wort_train.get(w, 0)
        if n <= MAX_TRAIN or mt <= MAX_TOKEN:
            risiko.append(dict(wort=form, zipf=round(zipf_frequency(w, "de"), 2), training=n,
                               min_token=mt, tokens=tok.convert_ids_to_tokens(ids)))
    print(f"[3] rarely heard: {len(risiko)} words", flush=True)

    # --- 4) stress
    ipa = espeak_batch([r["wort"] for r in risiko])
    for r in risiko:
        r["ipa"] = ipa.get(r["wort"], "")
        r["silben"], r["ton"] = silben_und_ton(r["ipa"])
        r["ton_hinten"] = r["silben"] >= 2 and r["ton"] > 0

    risiko.sort(key=lambda r: (-r["zipf"]))
    OUT.mkdir(parents=True, exist_ok=True)
    aus = OUT / "seltene_woerter.tsv"
    with aus.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh, delimiter="\t")
        # columns: word, Zipf today, count in training, rarest token count, tokens,
        # IPA, syllables, stressed syllable (1-based), stress not on the first syllable
        wr.writerow(["wort", "zipf_heute", "im_training", "seltenstes_token", "tokens",
                     "ipa", "silben", "tonsilbe", "ton_nicht_vorn"])
        for r in risiko:
            wr.writerow([r["wort"], r["zipf"], r["training"], r["min_token"], " ".join(r["tokens"]),
                         r["ipa"], r["silben"], r["ton"] + 1, int(r["ton_hinten"])])
    json.dump(risiko, open(OUT / "seltene_woerter.json", "w", encoding="utf-8"), ensure_ascii=False)
    hinten = [r for r in risiko if r["ton_hinten"]]
    print(f"[4] of these, stress not on the first syllable: {len(hinten)}")
    print(f"    -> {aus}")


if __name__ == "__main__":
    main()
