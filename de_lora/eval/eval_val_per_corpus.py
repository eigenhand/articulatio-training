"""eval_val_per_corpus.py — val loss on the complete val split, per corpus.

Same loss and the same deterministic item variants as quick_val_loss in
train_lora.py (reference clip, instruction, never noise-augmented), but over
every val item instead of 96, and reported per corpus as well as overall
(micro average over items). Base model first, then each adapter.

  scripts/run.sh de_lora/eval/eval_val_per_corpus.py --out val.json \
      --adapters runs/de-en-v4/checkpoints/final runs/de-en-v5/checkpoints/step3000
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

_PROJ = Path(__file__).resolve().parents[2]
for _p in (str(_PROJ / "de_lora" / "core"), str(_PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import common_breeze as CB  # noqa: E402
from train_lora import TrainDataset, load_split  # noqa: E402

DEV = "cuda"


def per_corpus_loss(model, ds, batch_size, limit_per_corpus=None):
    by = defaultdict(list)
    for i, idx in enumerate(ds.idxs):
        by[ds.records[idx]["corpus"]].append(i)
    model.eval()
    sums, counts = defaultdict(float), defaultdict(int)
    with torch.no_grad():
        for corp in sorted(by):
            lst = by[corp][:limit_per_corpus] if limit_per_corpus else by[corp]
            for j in range(0, len(lst), batch_size):
                chunk = lst[j:j + batch_size]
                batch = CB.collate([ds[k] for k in chunk])
                batch = {k: (v.to(DEV) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                # The model's loss is a mean over the batch; weight it by the batch size.
                sums[corp] += model(**batch).loss.item() * len(chunk)
                counts[corp] += len(chunk)
    per = {c: sums[c] / counts[c] for c in sums}
    overall = sum(sums.values()) / max(1, sum(counts.values()))
    return overall, per, dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit-per-corpus", type=int, default=None, help="smoke test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    raw = CB.load_breeze_model(DEV, attn="eager")
    ds = TrainDataset(load_split("val"), CB.load_text_tokenizer())
    print(f"[data] val={len(ds)} items", flush=True)
    results = []

    def run(label, model, path=None):
        t = time.time()
        overall, per, n = per_corpus_loss(model, ds, a.batch, a.limit_per_corpus)
        results.append({"label": label, "adapter": path, "overall": overall, "per_corpus": per,
                        "n": n, "seconds": round(time.time() - t)})
        print(f"[{label}] overall={overall:.4f} " + " ".join(f"{c}={v:.4f}" for c, v in per.items())
              + f" ({time.time() - t:.0f} s)", flush=True)
        Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf-8")

    run("base", raw)
    from peft import PeftModel
    for ad in a.adapters:
        p = Path(ad)
        p = p if p.is_absolute() else (CB.TRAINING / p).resolve()
        label = f"{p.parent.parent.name}/{p.name}"
        pm = PeftModel.from_pretrained(raw, str(p), is_trainable=False)
        run(label, pm, str(p))
        raw = pm.unload()
        del pm
        gc.collect()
        torch.cuda.empty_cache()
    print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
