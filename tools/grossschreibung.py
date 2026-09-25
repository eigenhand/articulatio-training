"""grossschreibung.py — restore proper capitalization of all-lowercase transcripts.

("Grossschreibung" = capitalization.) 38 % of the selected Thorsten sentences
are entirely lowercase. The Gemma tokenizer is case-sensitive: the model would
otherwise learn '▁geld' instead of '▁Geld' - tokens that never occur in real
(chat) text.

The table is built from all correctly capitalized transcripts, counted ONLY in
the middle of a sentence: at the start of a sentence everything is capitalized,
which would distort the table. Words that never occur there stay lowercase -
better a gap than a guess.

Run directly, it builds the table from CML German + HUI (processed corpora) and
the Thorsten Neutral subsets and writes BREEZE_WORDS_DIR/grossschreibung.json
(default <PTBR_ARTIFACTS>, see de_lora/core/paths.py). suche_seltene_woerter.py
imports richte() and loads that table.
"""
from __future__ import annotations
import collections, csv, glob, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "de_lora" / "core"))
import paths  # noqa: E402  (all locations are configurable, see de_lora/core/paths.py)

WORT = re.compile(r"[A-Za-zÄÖÜäöüß]+")
SATZENDE = re.compile(r"[.!?:;]\s*$")


# True if every word of t is lowercase ("klein" = lowercase).
def ist_klein(t: str) -> bool:
    ws = WORT.findall(t)
    return bool(ws) and all(w == w.lower() for w in ws)


# Builds the table: casefolded word -> most frequent spelling in mid-sentence position.
def baue_tabelle(texte) -> dict[str, str]:
    z = collections.defaultdict(collections.Counter)
    for t in texte:
        if ist_klein(t):
            continue                       # these say nothing about capitalization
        vorher = ""
        for m in WORT.finditer(t):
            # Do not count parts of hyphenated compounds: in "Online-Shop",
            # "Online" is capitalized because the whole compound is a noun. That
            # wrongly pulled "online" as an adverb to upper case.
            an_strich = t[m.start()-1:m.start()] == "-" or t[m.end():m.end()+1] == "-"
            if vorher and not an_strich and not SATZENDE.search(t[:m.start()]):
                z[m.group().casefold()][m.group()] += 1
            vorher = m.group()
    return {k: c.most_common(1)[0][0] for k, c in z.items()}


def richte(t: str, tabelle: dict[str, str]) -> str:
    """Only touch sentences that are entirely lowercase.

    Hyphen chains ("online-übersetzer") are handled as a unit: every part is
    looked up in the table, and if the last part is a noun (capitalized), the
    whole chain is capitalized - as is usual in German.
    """
    if not ist_klein(t):
        return t
    kette = re.compile(r"[A-Za-zÄÖÜäöüß]+(?:-[A-Za-zÄÖÜäöüß]+)*")
    aus, pos, satzanfang = [], 0, True
    for m in kette.finditer(t):
        aus.append(t[pos:m.start()])
        teile = [tabelle.get(x.casefold(), x) for x in m.group().split("-")]
        if len(teile) > 1 and teile[-1][:1].isupper():
            teile = [x[:1].upper() + x[1:] for x in teile]
        w = "-".join(teile)
        if satzanfang:
            w = w[:1].upper() + w[1:]
        aus.append(w)
        pos = m.end()
        satzanfang = bool(SATZENDE.search(t[m.end():m.end() + 2]))
    aus.append(t[pos:])
    return "".join(aus)


def alle_texte():
    root = paths.DATASETS_ROOT
    for p in (root / "cml_de/texts.csv", root / "hui_de/texts.csv"):
        for ln in p.read_text(encoding="utf-8").splitlines():
            if "==" in ln:
                yield ln.split("==", 1)[1]
    import pyarrow.parquet as pq
    for fp in (glob.glob(str(paths.THORSTEN_DIR / "TV-2021.02-Neutral" / "train-*.parquet"))
               + glob.glob(str(paths.THORSTEN_DIR / "TV-2022.10-Neutral" / "train-*.parquet"))):
        yield from (t for t in pq.read_table(fp, columns=["text"]).column("text").to_pylist() if t)
    for split in ("train", "dev", "test"):
        with open(paths.CML_DE_ROOT / f"{split}.csv",
                  encoding="utf-8", newline="") as fh:
            yield from (r["transcript"] for r in csv.DictReader(fh, delimiter="|"))


if __name__ == "__main__":
    tab = baue_tabelle(alle_texte())
    paths.WORDS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(tab, open(paths.WORDS_DIR / "grossschreibung.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"table: {len(tab)} words")
    # German test sentences (data): all lowercase on purpose.
    for t in ("desto mehr geld können die online-kriminellen damit verdienen.",
              "lange zeit hatten online-übersetzer mit schrägen satzkonstruktionen zu kämpfen.",
              "der programmcode der app steht seit einigen wochen online.",
              "rund ein drittel der unternehmen sei inzwischen auch online aktiv.",
              "die temperaturen steigen heute auf zwanzig grad. morgen wird es kühler."):
        print(f"  {richte(t, tab)}")
