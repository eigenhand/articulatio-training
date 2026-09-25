"""report.py — aggregate the release evaluation into Markdown tables and one JSON.

  python report.py --testset testset/ --variant out/final-q8_0 ... --out results/

WER/CER are corpus-level (all edits / all reference words or characters), with a
95 % bootstrap interval over items (1000 resamples, fixed seed). Similarity, SNR and
HF ratio are means over items.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from jobs import build
from score import item_id, normalize

RNG = np.random.default_rng(0)


def corpus_rate(rows, key):
    """rows: list of (errors, length) -> rate and 95 % bootstrap interval."""
    if not rows:
        return None
    e = np.array([r[0] for r in rows], float)
    n = np.array([r[1] for r in rows], float)
    rate = e.sum() / n.sum()
    idx = RNG.integers(0, len(rows), (1000, len(rows)))
    boot = e[idx].sum(1) / n[idx].sum(1)
    return {"rate": rate, "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)), "n": len(rows)}


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else None


def pct(r):
    return "–" if r is None else f"{100 * r['rate']:.1f} % ({100 * r['lo']:.1f}–{100 * r['hi']:.1f})"


def f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", required=True)
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ts = Path(a.testset)
    items = {i["id"]: i for i in (json.loads(l) for l in open(ts / "testset.jsonl", encoding="utf-8"))}
    jobs = {f"{j['set']}/{j['id']}": j for j in build(str(ts))}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def counts(text, lang, r):
        ref = normalize(text, lang)
        if lang == "zh":
            n = len(ref.replace(" ", ""))
            return (r["cer"] * n, n), (r["cer"] * n, n)
        nw, nc = len(ref.split()), len(ref)
        return (r["wer"] * nw, nw), (r["cer"] * nc, nc)

    summary = {}
    gt_path = ts / "ground_truth_scores.json"
    sources = []
    if gt_path.exists():
        sources.append(("recordings", {f"clone/{k}": dict(v, set="clone") for k, v in json.loads(gt_path.read_text()).items()}))
    for v in a.variant:
        sources.append((Path(v).name, json.loads((Path(v) / "scores.json").read_text())))

    md = []
    # ---- clone
    md.append("## Clone set (130 test utterances, 104 speakers)\n")
    md.append("| Variant | WER de | CER de | WER en | Speaker sim. to reference | to real recording | SNR dB | HF ratio | Clips with WER > 50 % |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    per_corpus_md = ["\n### WER per corpus\n", "| Variant | " + " | ".join(sorted({i['corpus'] for i in items.values()})) + " |",
                     "|---|" + "---:|" * len({i['corpus'] for i in items.values()})]
    for name, sc in sources:
        rows = [(k, r) for k, r in sc.items() if r.get("set") == "clone"]
        w = defaultdict(list); c = defaultdict(list); byc = defaultdict(list)
        for k, r in rows:
            it = items[k.split("/", 1)[1]]
            wc, cc = counts(it["text"], it["lang"], r)
            w[it["lang"]].append(wc); c[it["lang"]].append(cc); byc[it["corpus"]].append(wc)
        s = {"wer_de": corpus_rate(w["de"], "wer"), "cer_de": corpus_rate(c["de"], "cer"),
             "wer_en": corpus_rate(w["en"], "wer"),
             "sim_ref": mean([r.get("sim_ref") for _, r in rows]), "sim_target": mean([r.get("sim_target") for _, r in rows]),
             "snr_db": mean([r.get("snr_db") for _, r in rows]), "hf_ratio": mean([r.get("hf_ratio") for _, r in rows]),
             "bad": sum(1 for _, r in rows if r.get("wer", 0) > 0.5), "n": len(rows),
             "per_corpus": {k: corpus_rate(v, "wer") for k, v in byc.items()}}
        summary.setdefault(name, {})["clone"] = s
        md.append(f"| {name} | {pct(s['wer_de'])} | {pct(s['cer_de'])} | {pct(s['wer_en'])} | {f(s['sim_ref'])} | "
                  f"{f(s['sim_target'])} | {f(s['snr_db'], 1)} | {f(s['hf_ratio'])} | {s['bad']} / {s['n']} |")
        per_corpus_md.append(f"| {name} | " + " | ".join(
            "–" if s["per_corpus"].get(cn) is None else f"{100 * s['per_corpus'][cn]['rate']:.1f} %"
            for cn in sorted({i['corpus'] for i in items.values()})) + " |")
    md += per_corpus_md

    # ---- noisy
    md.append("\n## Degraded references (30 items of HUI, Thorsten, HiFiTTS-2)\n")
    md.append("Same items cloned from the clean reference (clone set) as the target quality.\n")
    md.append("| Variant | Condition | SNR dB | HF ratio | Speaker sim. | WER |")
    md.append("|---|---|---:|---:|---:|---:|")
    for name, sc in sources:
        if name == "recordings":
            continue
        noisy_ids = {item_id(jobs[k]) for k, r in sc.items() if r.get("set") == "noisy"}
        conds = [("clean reference", lambda k: k.startswith("clone/") and k[6:] in noisy_ids)] + [
            (lbl, (lambda suf: lambda k: k.startswith("noisy/") and k.endswith(suf))(suf))
            for lbl, suf in (("degraded, plain instruction", "-plain"), ("degraded, clean instruction, cfg 1", "-clean-cfg1"),
                             ("degraded, clean instruction, cfg 3", "-clean-cfg3"))]
        for lbl, pred in conds:
            rows = [(k, r) for k, r in sc.items() if pred(k)]
            if not rows:
                continue
            wc = [counts(jobs[k]["text"], jobs[k]["lang"], r)[0] for k, r in rows]
            s = {"snr_db": mean([r.get("snr_db") for _, r in rows]), "hf_ratio": mean([r.get("hf_ratio") for _, r in rows]),
                 "sim_ref": mean([r.get("sim_ref") for _, r in rows]), "wer": corpus_rate(wc, "wer"), "n": len(rows)}
            summary[name].setdefault("noisy", {})[lbl] = s
            md.append(f"| {name} | {lbl} | {f(s['snr_db'], 1)} | {f(s['hf_ratio'])} | {f(s['sim_ref'])} | {pct(s['wer'])} |")

    # ---- zh
    md.append("\n## Chinese regression (10 sentences, voice design)\n")
    md.append("| Variant | CER |")
    md.append("|---|---:|")
    for name, sc in sources:
        rows = [(k, r) for k, r in sc.items() if r.get("set") == "zh"]
        if rows:
            s = corpus_rate([counts(jobs[k]["text"], "zh", r)[1] for k, r in rows], "cer")
            summary[name]["zh"] = s
            md.append(f"| {name} | {pct(s)} |")

    # ---- long
    md.append("\n## Long passages (~70 s in one piece, no sentence splitting)\n")
    md.append("Per 10 s window; first = mean of the first two windows, last = mean of the last two.\n")
    md.append("| Variant | Passage | Duration | WER | HF ratio first → last | Speaker sim. first → last | Whisper log-prob first → last |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for name, sc in sources:
        for k, r in sorted(sc.items()):
            if r.get("set") != "long":
                continue
            wins = r.get("windows", [])
            fl = lambda key: (mean([w[key] for w in wins[:2]]), mean([w[key] for w in wins[-2:]]))
            hf, sim, lp = fl("hf_ratio"), fl("sim_ref"), fl("avg_logprob")
            summary[name].setdefault("long", {})[k] = {"dur": r["dur"], "wer": r["wer"], "windows": wins}
            md.append(f"| {name} | {k.split('/', 1)[1]} | {r['dur']:.0f} s | {100 * r['wer']:.1f} % | {f(hf[0])} → {f(hf[1])} | "
                      f"{f(sim[0])} → {f(sim[1])} | {f(lp[0], 2)} → {f(lp[1], 2)} |")

    # ---- speed
    md.append("\n## Speed during the evaluation\n")
    md.append("Wall time / audio duration over all jobs. Only comparable within one machine and run.\n")
    md.append("| Variant | RTF | Audio generated |")
    md.append("|---|---:|---:|")
    for v in a.variant:
        tf = Path(v) / "timing.jsonl"
        if tf.exists():
            t = [json.loads(l) for l in open(tf)]
            rtf = sum(x["seconds"] for x in t) / max(1e-9, sum(x["audio_s"] for x in t))
            summary[Path(v).name]["rtf"] = rtf
            md.append(f"| {Path(v).name} | {rtf:.2f} | {sum(x['audio_s'] for x in t) / 60:.1f} min |")

    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
