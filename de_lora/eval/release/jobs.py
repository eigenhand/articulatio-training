"""jobs.py — the generation jobs of the release evaluation, shared by both synthesizers.

Every job is one WAV: {"set", "id", "text", "lang", "ref_wav", "ref_text",
"instruction", "cfg", "seed", "max_frames"}. The same job list (and seed) goes to
every model variant, so differences between variants are not sampling luck of
different seeds per item (sampling noise per item remains; the sets average it).

Sets:
  clone  every test item, cloned from another clip of its speaker, standard instruction
  noisy  degraded reference: plain instruction (cfg 1) / clean-studio instruction at cfg 1 and 3
  zh     Chinese regression (voice design, no reference): the fine-tune never saw Chinese
  long   two ~70 s passages in one piece (no sentence splitting): drift over time
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

# The first entries of the training instruction pools (de_lora/core/common_breeze.py)
# and the clean-studio instructions of run 5 (de_lora/core/prepare_dataset.py), verbatim.
PLAIN = {"de": "Sprich klar und natuerlich.", "en": "Speak clearly and naturally."}
CLEAN = {"de": "Saubere Studioaufnahme ohne Hintergrundgeraeusche, klar und natuerlich gesprochen.",
         "en": "Clean studio recording without background noise, spoken clearly and naturally."}
ZH_INSTRUCTION = "一位成年男性，声音清晰自然，语速适中。"
ZH_TEXTS = [
    "今天天气晴朗，东边吹来一阵微风。",
    "请在下一个路口向左转，然后直走两百米。",
    "我们明天早上八点在火车站门口见面。",
    "这本书讲述了一个小村庄一百年来的变化。",
    "如果你有任何问题，请随时给我打电话。",
    "厨房里有新鲜的面包和一壶热茶。",
    "会议推迟到下周三下午两点举行。",
    "孩子们在公园里放风筝，笑声不断。",
    "这座城市的地铁每天运送三百万名乘客。",
    "他把钥匙忘在了办公室的桌子上。",
]


def seed_for(job_id: str) -> int:
    return int(hashlib.sha1(job_id.encode()).hexdigest()[:8], 16) % (2 ** 31)


def long_passages(items: list[dict], meta: list[dict]) -> list[dict]:
    """Two passages of consecutive sentences from one book reader (HUI) and Thorsten."""
    out = []
    for corpus in ("hui_de", "thorsten_rare_de"):
        it = next(i for i in items if i["corpus"] == corpus)
        same = sorted((m for m in meta if m["corpus"] == corpus and m["speaker"] == it["speaker"]),
                      key=lambda m: m["idx"])
        start = next(k for k, m in enumerate(same) if m["idx"] == it["target_wav"].rsplit("/", 1)[1][:-4])
        words, text = 0, []
        for m in same[start:]:
            text.append(m["text"].strip())
            words += len(m["text"].split())
            if words >= 170:        # ~70 s at a reading pace of ~2.4 words/s
                break
        out.append({"id": f"long-{corpus}", "lang": "de", "text": " ".join(text),
                    "ref_wav": it["ref_wav"], "ref_text": it["ref_text"]})
    return out


def build(testset_dir: str, meta_path: str | None = None) -> list[dict]:
    items = [json.loads(l) for l in open(Path(testset_dir) / "testset.jsonl", encoding="utf-8")]
    jobs = []
    for it in items:
        base = dict(text=it["text"], lang=it["lang"], ref_text=it["ref_text"], max_frames=250)
        jobs.append(dict(base, set="clone", id=it["id"], ref_wav=it["ref_wav"],
                         instruction=PLAIN[it["lang"]], cfg=1.0))
        if "noisy_ref_wav" in it:
            for name, instr, cfg in (("plain", PLAIN, 1.0), ("clean-cfg1", CLEAN, 1.0), ("clean-cfg3", CLEAN, 3.0)):
                jobs.append(dict(base, set="noisy", id=f"{it['id']}-{name}", ref_wav=it["noisy_ref_wav"],
                                 instruction=instr[it["lang"]], cfg=cfg))
    for k, t in enumerate(ZH_TEXTS):
        jobs.append(dict(set="zh", id=f"zh-{k:02d}", text=t, lang="zh", ref_wav=None, ref_text=None,
                         instruction=ZH_INSTRUCTION, cfg=1.0, max_frames=250))
    long_file = Path(testset_dir) / "long.jsonl"
    if not long_file.exists():
        meta = [json.loads(l) for l in open(meta_path, encoding="utf-8")]
        with open(long_file, "w", encoding="utf-8") as f:
            for p in long_passages(items, meta):
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    for p in (json.loads(l) for l in open(long_file, encoding="utf-8")):
        jobs.append(dict(p, set="long", instruction=PLAIN["de"], cfg=1.0, max_frames=1100))
    for j in jobs:
        j["seed"] = seed_for(j["id"])
    return jobs
