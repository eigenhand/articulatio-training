"""smoke_forward_b.py — phase B final gate: a REAL forward pass with the checkpoint weights.

Validates:
 1) finite losses (backbone_loss, depth_decoder_loss, weighted loss);
 2) the semantics of -101: identical backbone loss, different depth loss (A/B experiment);
 3) gradient graph integrity (1 unfrozen weight -> backward OK);
 4) peak VRAM / timings.

Needs the gold samples written by `prepare_dataset.py finalize`.
Usage: python de_lora/tools/smoke_forward_b.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[2]
_CORE = _PROJ / "de_lora" / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

import common_breeze as CB
import torch

CB.GOLD_DIR.mkdir(parents=True, exist_ok=True)
DEV = "cuda"


def load_gold(name: str) -> dict:
    blob = torch.load(CB.GOLD_DIR / f"{name}.pt", map_location="cpu")
    return {
        "input_ids": blob["input_ids"].long(),
        "attention_mask": torch.ones_like(blob["input_ids"]).long(),
        "text_ids_mask": blob["text_ids_mask"],
        "text_ids_len": blob["text_ids_len"].long(),
        "input_values": blob["input_values"].int(),
        "labels": blob["labels"].long(),
    }


def relabel(item: dict, frame_policies: list[str]) -> dict:
    ex = {
        "input_ids": item["input_ids"].clone(),
        "text_ids_mask": item["text_ids_mask"].clone(),
        "text_ids_len": item["text_ids_len"].clone(),
        "audio_tokens": item["input_values"],
    }
    ex["labels"] = CB.make_labels(ex, frame_policies=frame_policies)
    out = dict(item)
    out["labels"] = ex["labels"]
    return out


def main() -> None:
    t0 = time.time()
    _ = CB.load_text_tokenizer()  # checks that the text tokenizer loads
    model = CB.load_breeze_model(DEV)
    print(f"[load] model ready in {time.time()-t0:.1f}s")

    items = [load_gold("gold0_tts_instruction"), load_gold("gold1_ref_edit_tata")]
    lens = [it["input_ids"].shape[1] for it in items]
    nvals = [it["input_values"].shape[1] for it in items]
    print(f"[data] seq_lens={lens} frames={nvals}")

    # ---------------------------------------------------------- 1) default forward
    batch = CB.collate(items)
    batch = {k: v.to(DEV) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    print("[shapes]", {k: tuple(v.shape) for k, v in batch.items()})

    model.train()
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    with torch.no_grad():
        out = model(**batch)
    fwd_s = time.time() - t1
    loss, bl, dl = out.loss.item(), out.backbone_loss.item(), out.depth_decoder_loss.item()
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    ok_fin = all(map(torch.isfinite, [out.loss, out.backbone_loss, out.depth_decoder_loss]))
    print(f"[forward] loss={loss:.4f} backbone={bl:.4f} depth={dl:.4f} "
          f"finite={ok_fin} t={fwd_s:.2f}s peak_vram={peak_gb:.2f}GB mode=train")

    # --------------------------------------------- 2) -101 experiment (A/B)
    it_alt = [items[0], relabel(items[1], ["train", "train"])]
    batch_alt = CB.collate(it_alt)
    batch_alt = {k: v.to(DEV) if isinstance(v, torch.Tensor) else v for k, v in batch_alt.items()}
    with torch.no_grad():
        out_a = model(**batch_alt)
    print(f"[exp-101] A(-101 ref): back={bl:.6f} depth={dl:.6f}")
    print(f"[exp-101] B(all-train): back={out_a.backbone_loss.item():.6f} "
          f"depth={out_a.depth_decoder_loss.item():.6f}")
    same_b = abs(out_a.backbone_loss.item() - bl) < 1e-4
    diff_d = abs(out_a.depth_decoder_loss.item() - dl) > 1e-3
    print(f"[exp-101] => backbone equal={same_b} | depth differs={diff_d} "
          f"(expected: True|True)")

    # ------------------------------------------------ 3) gradient integrity
    for p in model.parameters():
        p.requires_grad_(False)
    probe = model.backbone_model.layers[-1].self_attn.q_proj.weight
    probe.requires_grad_(True)
    model.eval()  # branch without weighting, to compare determinism
    torch.cuda.reset_peak_memory_stats()
    t2 = time.time()
    out2 = model(**batch)
    out2.backbone_loss.backward()
    g = probe.grad
    dt2 = time.time() - t2
    grad_ok = g is not None and bool(torch.isfinite(g).all())
    print(f"[grad] gradient reaches q_proj of the last layer: exists={g is not None} "
          f"finite={grad_ok} norm={float(g.norm()) if g is not None else float('nan'):.6f}")
    print(f"[grad] forward+backward t={dt2:.2f}s "
          f"peak_vram={torch.cuda.max_memory_allocated()/2**30:.2f}GB")

    model.zero_grad(set_to_none=True)
    del out, out_a, out2, batch, batch_alt
    torch.cuda.empty_cache()
    print(f"[end] max VRAM overall: {torch.cuda.max_memory_allocated()/2**30:.2f}GB")


if __name__ == "__main__":
    main()
