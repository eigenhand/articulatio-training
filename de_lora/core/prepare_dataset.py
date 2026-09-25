"""prepare_dataset.py — PHASE B: prepare the registered corpora for LoRA training.

Subcommands:
  process   [--limit N] [--device cuda]     parse+clean+resample+encode -> codes/, meta
  finalize  [--gold N] [--parity-device cuda]  90/5/5 splits + gold dumps + official parity check

Resumable: 'process' skips samples whose codes are already in the code store.
Usage: python de_lora/core/prepare_dataset.py process --limit 40
       (or scripts/run.sh de_lora/core/prepare_dataset.py ..., which sets up ROCm/MIOpen)
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_CORE = Path(__file__).resolve().parent
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

import common_breeze as CB
from codestore import CodeReader, CodeWriter
CB.TRAINING.mkdir(parents=True, exist_ok=True)

# Consolidated code store instead of tokens/<idx>.npz — see codestore.py.
CODES_DIR = CB.TRAINING / "codes"

_CODES_READER: CodeReader | None = None


def _codes() -> CodeReader:
    """Create the reader once; the memmap should not be rebuilt for every item."""
    global _CODES_READER
    if _CODES_READER is None:
        _CODES_READER = CodeReader(CODES_DIR)
    return _CODES_READER


META_JSONL = CB.TRAINING / "dataset_meta.jsonl"
MANIFEST_CSV = CB.TRAINING / "manifest.csv"
SUMMARY = CB.TRAINING / "summary.json"

LINE_RE = re.compile(r"^(wavs/[^\s]+\.wav)==(.*)$")


# ------------------------------------------------------------------ csv


def _speaker_map_for(corp: dict) -> dict[str, str]:
    """idx -> speaker. Priority: the corpus' speakers.jsonl; the legacy podcast
    corpus uses chunks_index.jsonl; tata has no map (default 'tata')."""
    name = corp["name"]
    root = CB.corpus_dir(corp)
    m: dict[str, str] = {}
    spk_file = root / "speakers.jsonl"
    if spk_file.exists():
        for ln in spk_file.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                m[r["idx"]] = r["speaker"]
        return m
    if name == "podcast":
        ci = CB.SCRAPING_WORK / "chunks_index.jsonl"
        if ci.exists():
            for ln in ci.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    r = json.loads(ln)
                    m[r["chunk"].removesuffix(".wav")] = f"{r['tag']}:{r['speaker']}"
    return m


def parse_csv() -> list[dict]:
    rows: list[dict] = []
    seen = Counter()
    n_bad = n_dupe = 0
    for corp in CB.load_corpora():
        name = corp["name"]
        root = CB.corpus_dir(corp)
        csv = CB.corpus_csv(corp)
        if not csv.exists():
            print(f"[parse] corpus '{name}': csv missing ({csv}) -> skipping")
            continue
        spk_map = _speaker_map_for(corp)
        raw = csv.read_text(encoding="utf-8")
        n_c = 0
        for ln_no, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            m = LINE_RE.match(line)
            if not m:
                n_bad += 1
                print(f"[parse] {name} invalid line #{ln_no}: {line[:60]!r}")
                continue
            rel, text = m.group(1), m.group(2)
            key = f"{name}:{rel}"
            seen[key] += 1
            if seen[key] > 1:
                n_dupe += 1
                continue
            text = html.unescape(text)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                n_bad += 1
                continue
            idx = Path(rel).stem
            speaker = spk_map.get(idx, "tata" if name == "tata" else idx)
            rows.append({"wav_rel": rel, "idx": idx, "text": text,
                         "corpus": name, "speaker": speaker, "root": str(root)})
            n_c += 1
        missing = [r for r in rows if r["corpus"] == name
                   and not (root / r["wav_rel"]).exists()]
        if missing:
            print(f"[parse] {name}: WARNING {len(missing)} wavs missing "
                  f"(e.g. {[r['wav_rel'] for r in missing[:3]]})")
        print(f"[parse] {name}: valid={n_c} missing={len(missing)}")
    print(f"[parse] TOTAL valid={len(rows)} dupes={n_dupe} invalid/empty={n_bad}")
    miss_keys = {(r["corpus"], r["wav_rel"]) for r in rows
                 if not (Path(r["root"]) / r["wav_rel"]).exists()}
    return [r for r in rows if (r["corpus"], r["wav_rel"]) not in miss_keys]


# ------------------------------------------------------------------ process


def process(limit: int | None, device: str) -> None:
    import librosa
    import soundfile as sf

    Qwen3TTSTokenizer = CB.import_qwen_tts()

    rows = parse_csv()
    _seen = CodeReader(CODES_DIR)
    n_done_pre = sum(1 for r in rows if r["idx"] in _seen)
    todo = [r for r in rows if r["idx"] not in _seen]
    if limit:
        todo = todo[:limit]
    print(f"[process] total_csv={len(rows)} done={n_done_pre} to_process={len(todo)}")

    CODES_DIR.mkdir(parents=True, exist_ok=True)
    writer = CodeWriter(CODES_DIR)
    meta_f = META_JSONL.open("a", encoding="utf-8")
    t_start = time.time()
    stats = Counter()
    atok = Qwen3TTSTokenizer.from_pretrained(str(CB.CKPT / "audio_tokenizer"), device_map=device)

    for k, row in enumerate(todo):
        src = Path(row["root"]) / row["wav_rel"]
        try:
            info = sf.info(src)
            dur = info.frames / info.samplerate
            if dur < CB.MIN_DUR_S:
                stats["drop_short"] += 1
                continue
            wav48, sr48 = sf.read(src, dtype="float32", always_2d=True)
            wav48 = wav48.mean(axis=1)
            text_used = row["text"]
            truncated = False
            if dur > CB.MAX_DUR_S:
                wav48, text_used = truncate_pair(wav48, sr48, row["text"], CB.MAX_DUR_S)
                truncated = True
                if len(text_used) < 20:
                    stats["drop_trunc_text"] += 1
                    continue
            wav48, _ = librosa.effects.trim(wav48, top_db=40)
            d_trim = len(wav48) / sr48
            if d_trim < CB.MIN_DUR_S:
                stats["drop_short_trim"] += 1
                continue
            if d_trim > CB.MAX_DUR_S + 0.3:
                stats["drop_long_trim"] += 1
                continue
            wav24 = librosa.resample(wav48, orig_sr=sr48, target_sr=CB.SR)
            peak = float(np.max(np.abs(wav24))) if len(wav24) else 0.0
            if peak < 0.30 or peak > 0.99:
                wav24 = wav24 * (CB.PEAK_NORM / max(peak, 1e-9))
            wav24 = wav24.clip(-1.0, 1.0).astype("float32")

            enc = atok.encode(wav24, sr=CB.SR)
            codes = enc["audio_codes"][0]
            codes = codes.detach().cpu().numpy().astype("int16") if hasattr(codes, "cpu") else codes
            assert codes.ndim == 2 and codes.shape[1] == 16, codes.shape
            assert codes.min() >= 0 and codes.max() < 2051, (codes.min(), codes.max())

            # Consolidated instead of one file per utterance — see codestore.py.
            # wavs24/ is dropped entirely: those paths only served as cache keys,
            # the files were never read (_encode_prompt_audio is patched).
            writer.append(row["idx"], codes)
            rec = {
                "idx": row["idx"],
                "wav_rel": row["wav_rel"],
                "corpus": row["corpus"],
                "speaker": row["speaker"],
                "text": text_used,
                "dur_src_s": round(dur, 3),
                "dur_proc_s": round(len(wav24) / CB.SR, 3),
                "frames": int(codes.shape[0]),
                "truncated": truncated,
            }
            meta_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            meta_f.flush()
            stats["ok_trunc" if truncated else "ok"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["error"] += 1
            print(f"[process] ERROR {row['idx']}: {type(exc).__name__}: {exc}")
        if (k + 1) % 100 == 0:
            rate = (time.time() - t_start) / (k + 1)
            eta = rate * (len(todo) - k - 1) / 60
            print(f"[process] {k+1}/{len(todo)} ok={stats['ok']} eta={eta:.1f}min")

    writer.close()   # the last buffer must be flushed, otherwise the last items are missing
    meta_f.close()
    print(f"[process] DONE {dict(stats)} in {(time.time()-t_start)/60:.1f} min")


def np_absmax(a):
    return float(np.max(np.abs(a))) if len(a) else 0.0


PUNCT_CUT_RE = re.compile(r"[.!?;:](\s|$)")


def truncate_pair(wav48, sr, text: str, cap_s: float):
    """Cuts the audio at the silence gap closest to the limit and cuts the text
    proportionally by words (read-speech corpus = uniform pace).
    Returns (truncated_wav, adjusted_text). The caller checks the minimum limits."""
    import librosa

    intervals = librosa.effects.split(wav48, top_db=40)
    cap_n = int(cap_s * sr)
    ends = [int(e) for s, e in intervals if e <= cap_n]
    cut = ends[-1] if ends else cap_n
    wav = wav48[:cut]
    dur_total = max(len(wav48) / sr, 1e-9)
    frac = min(1.0, (len(wav) / sr) / dur_total)
    words = text.split()
    kw = max(4, int(round(len(words) * frac)))
    seg = " ".join(words[:kw])
    m = list(PUNCT_CUT_RE.finditer(seg + " "))
    if m:
        seg = seg[: m[-1].start()].strip()
    elif len(m) == 0 and "," in seg and kw < len(words):
        seg = seg[: seg.rfind(",")].strip()
    return wav, seg


# ------------------------------------------------------------------ finalize


def _apply_corpus_overrides(by_idx: dict[str, dict]) -> None:
    """Applies each corpus' speakers.jsonl (speaker) and ref_map.jsonl
    (similarity-based reference) on top of the meta records."""
    for corp in CB.load_corpora():
        root = CB.corpus_dir(corp)
        name = corp["name"]
        spk_file = root / "speakers.jsonl"
        if spk_file.exists():
            for ln in spk_file.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                t = by_idx.get(r["idx"])
                if t is not None:
                    t["speaker"] = r["speaker"]
                    t.setdefault("corpus", name)
        ref_file = root / "ref_map.jsonl"
        if ref_file.exists():
            for ln in ref_file.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                t = by_idx.get(r["idx"])
                if t is not None and r.get("ref_idx"):
                    t["ref_idx"] = r["ref_idx"]


def load_meta() -> list[dict]:
    global _META_CACHE
    if _META_CACHE is not None:
        return _META_CACHE
    by_idx: dict[str, dict] = {}
    for ln in META_JSONL.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            by_idx[r["idx"]] = r  # the last occurrence wins (idempotent)
    recs = list(by_idx.values())
    spk = _chunk_speaker_map()
    for r in recs:
        if "speaker" not in r:
            r["corpus"] = "podcast" if r["idx"] in spk else "tata"
            r["speaker"] = spk.get(r["idx"], "tata")
    _apply_corpus_overrides(by_idx)
    _META_CACHE = recs
    return recs


_META_CACHE: list[dict] | None = None


def invalidate_meta_cache() -> None:
    global _META_CACHE, _REC_BY_IDX
    _META_CACHE = None
    _REC_BY_IDX = None


def _chunk_speaker_map() -> dict[str, str]:
    """podcast idx -> 'tag:SPEAKER_XX' (identities are NOT shared across episodes)."""
    global _SPK_MAP
    if _SPK_MAP is not None:
        return _SPK_MAP
    m: dict[str, str] = {}
    ci = CB.SCRAPING_WORK / "chunks_index.jsonl"
    if ci.exists():
        for ln in ci.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                m[r["chunk"].removesuffix(".wav")] = f"{r['tag']}:{r['speaker']}"
    _SPK_MAP = m
    return m


_SPK_MAP: dict[str, str] | None = None


REF_EDIT_FRAC = 0.5


def set_ref_edit_frac(frac: float) -> None:
    """Fraction of examples in ref_edit mode (with a reference). 1.0 = all of them."""
    global REF_EDIT_FRAC
    REF_EDIT_FRAC = min(1.0, max(0.0, float(frac)))


def deterministic_variant(idx: str, rec: dict | None = None) -> str:
    """ref_edit (with a reference) or tts_instruction (without), by hash of the idx.
    ref_edit_auto uses the speaker (ref_map/pool); Tata and the fallback stay self-ref."""
    h = hashlib.sha1(idx.encode()).digest()[0]
    if (h / 255.0) >= REF_EDIT_FRAC:
        return "tts_instruction"
    if rec is not None and rec.get("speaker") and rec["speaker"] != "tata":
        return "ref_edit_auto"
    return "ref_edit_tata"


def instruction_for_rec(rec: dict) -> str:
    """Deterministic instruction per recording; language chosen by corpus."""
    seed = (int(rec["frames"]) * 7 + sum(map(ord, rec["idx"]))) % 10_000
    pool = CB.instruction_pool_for(rec.get("corpus", ""))
    return pool[seed % len(pool)]


_SPK_POOL: dict[str, list[str]] | None = None


def _speaker_pool() -> dict[str, list[str]]:
    """speaker -> idxs OF THE TRAINING SPLIT (avoids leaking val/test clips as refs)."""
    global _SPK_POOL
    if _SPK_POOL is not None:
        return _SPK_POOL
    train = set(load_split_file("train"))
    pool: dict[str, list[str]] = {}
    for r in load_meta():
        if r["idx"] in train:
            pool.setdefault(r.get("speaker", "tata"), []).append(r["idx"])
    _SPK_POOL = pool
    return _SPK_POOL


def load_split_file(name: str) -> list[str]:
    p = CB.TRAINING / f"splits_{name}.txt"
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _rec_by_idx() -> dict[str, dict]:
    """idx->rec map (avoids an O(N) linear search per ref_edit item)."""
    global _REC_BY_IDX
    if _REC_BY_IDX is None:
        _REC_BY_IDX = {r["idx"]: r for r in load_meta()}
    return _REC_BY_IDX


_REC_BY_IDX: dict[str, dict] | None = None


def pick_ref_rec(rec: dict) -> dict:
    """Reference from the SAME speaker. Priority: ref_map (idx->ref_idx with a
    similarity guard); otherwise the speaker pool + hash; otherwise self-ref."""
    ridx = rec.get("ref_idx")
    if ridx:
        rr = _rec_by_idx().get(ridx)
        if rr is not None:
            return rr
    spk = rec.get("speaker", "tata")
    pool = [i for i in _speaker_pool().get(spk, []) if i != rec["idx"]]
    if not pool:
        return rec
    j = int(hashlib.sha1(("ref:" + rec["idx"]).encode()).hexdigest(), 16) % len(pool)
    return _rec_by_idx().get(pool[j], rec)


# ------------------------------------------------ noise-augmented references (run 5)
#
# For clean corpora, part of the examples get a DEGRADED version of the reference
# clip (tools/rauschvorlagen.py: noise, reverb, telephone band, codec ...), while
# the target stays clean. They also get a dedicated instruction, which a share of
# the undegraded clean examples receives as well - so the instruction reliably
# means "clean output", no matter how the reference sounds. Common Voice stays as
# it is: there reference and target are both noisy, with the usual instructions.
#
# Naming: "Rausch" = noise, "sauber" = clean, "Anteil" = share, "Korpora" =
# corpora, "Leser" = reader. rausch_wahl() returns 'rausch' | 'sauber' | 'normal'.
# The instruction texts are training data and stay as they are.
SAUBER_DE = "Saubere Studioaufnahme ohne Hintergrundgeraeusche, klar und natuerlich gesprochen."
SAUBER_EN = "Clean studio recording without background noise, spoken clearly and naturally."
_RAUSCH: dict = {"anteil": 0.0, "sauber": 0.0, "korpora": set(), "leser": None}


def set_rauschen(codes_dir, anteil: float, sauber: float, korpora) -> None:
    _RAUSCH.update(anteil=float(anteil), sauber=float(sauber), korpora=set(korpora),
                   leser=CodeReader(Path(codes_dir)))


def rausch_wahl(idx: str) -> str:
    """'rausch' (degraded reference + clean instruction), 'sauber' (instruction
    only) or 'normal' - fixed per idx, so that repetitions see the same thing."""
    if not _RAUSCH["leser"]:
        return "normal"
    u = int(hashlib.sha1(("rauschwahl:" + idx).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if u < _RAUSCH["anteil"]:
        return "rausch"
    if u < _RAUSCH["anteil"] + _RAUSCH["sauber"]:
        return "sauber"
    return "normal"


def build_item(tokenizer, rec: dict, variant: str, augment: bool = False) -> dict[str, object]:
    """Item ready for the DataLoader (1-D tensors). Uses the code cache.

    ref_edit_auto: ref = ANOTHER clip of the SAME speaker (falls back to self-ref
    if the speaker has only 1 clip in train). Template/policies = ref_edit_tata.
    """
    wav_key = str(CB.WAVS24_DIR / f"{rec['idx']}.wav")
    n_frames = CB.register_codes_array(wav_key, _codes().get(rec["idx"]))
    instruction = instruction_for_rec(rec)
    if variant == "ref_edit_auto":
        ref_rec = pick_ref_rec(rec)
        ref_key = str(CB.WAVS24_DIR / f"{ref_rec['idx']}.wav")
        ref_codes = _codes().get(ref_rec["idx"])
        if augment and rec.get("corpus") in _RAUSCH["korpora"]:
            wahl = rausch_wahl(rec["idx"])
            if wahl != "normal":
                instruction = SAUBER_EN if rec["corpus"].endswith("_en") else SAUBER_DE
            leser = _RAUSCH["leser"]
            if wahl == "rausch" and leser is not None and ref_rec["idx"] in leser:
                # separate key: the cache must not hand out the clean version
                ref_key = str(CB.WAVS24_DIR / f"{ref_rec['idx']}.rausch.wav")
                ref_codes = leser.get(ref_rec["idx"])
        n_ref = CB.register_codes_array(ref_key, ref_codes)
        req = CB.ExampleRequest(
            variant="ref_edit_tata",
            text=rec["text"],
            instruction=instruction,
            ref_text=ref_rec["text"],
            ref_audio_path=ref_key,
            target_audio_path=wav_key,
        )
    else:
        n_ref = 0
        req = CB.ExampleRequest(
            variant=variant,
            text=rec["text"],
            instruction=instruction,
            ref_text=rec["text"],
            ref_audio_path=wav_key,
        )
    ex = CB.build_example(tokenizer, req)
    ex["labels"] = CB.make_labels(ex, frame_policies=CB.POLICIES_BY_VARIANT[variant])
    # structural sanity check
    assert len(ex["input_ids"][0]) == len(ex["labels"][0])
    n_audio_placeholders = int((ex["input_ids"][0] == CB.AUDIO_TOKEN_ID).sum())
    if variant == "ref_edit_auto":
        expected_frames = n_frames + n_ref
    else:
        expected_frames = n_frames * (2 if variant == "ref_edit_tata" else 1)
    assert n_audio_placeholders == expected_frames, (n_audio_placeholders, expected_frames)
    ex["_n_frames"] = n_frames
    ex["_variant"] = variant
    ex["_instruction"] = instruction
    return ex


def finalize(num_gold: int, parity_device: str) -> None:
    from breeze_infer.templates import _prepare_one

    recs = load_meta()
    print(f"[finalize] records={len(recs)}")
    assert len(recs) >= 100, "run the complete 'process' step before finalize"

    # split STRATIFIED BY CORPUS (train/val/test within every data source)
    cuts: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    by_corpus: dict[str, list[str]] = {}
    for r in recs:
        by_corpus.setdefault(r.get("corpus", "tata"), []).append(r["idx"])
    for corpus, lst in by_corpus.items():
        idxs = sorted(lst)
        rng = __import__("random").Random(42)
        rng.shuffle(idxs)
        n = len(idxs)
        if n < 40:
            cuts["train"].extend(idxs)
            print(f"[finalize] corpus '{corpus}': {n} items -> train (small)")
            continue
        cuts["train"].extend(idxs[: int(n * 0.9)])
        cuts["val"].extend(idxs[int(n * 0.9): int(n * 0.95)])
        cuts["test"].extend(idxs[int(n * 0.95):])
        print(f"[finalize] corpus '{corpus}': train={int(n*0.9)} val/test={n - int(n*0.9)}")
    for name, lst in cuts.items():
        (CB.TRAINING / f"splits_{name}.txt").write_text("\n".join(lst), encoding="utf-8")
    print({k: len(v) for k, v in cuts.items()})

    tokenizer = CB.load_text_tokenizer()

    # ---- parity with the official code (template: without the cache, i.e. the
    # reference encodes the real wav24; gold samples from Tata = read-speech corpus
    # whose parity had already been validated in phase B).
    # In this fork wavs24/ is no longer written, so the official path is served
    # from the same code cache: the check verifies that the assembled training
    # sequence matches the official template token for token.
    rec_by_idx = {r["idx"]: r for r in recs}
    g_idx = cuts["val"][: max(1, num_gold // 2)] + cuts["test"][: max(1, num_gold - num_gold // 2)]
    # The template filtered hard for its read-speech corpus "tata" here. With any
    # other corpus name the list would be empty and the parity check would run
    # over zero examples — i.e. silently not at all. We take the gold examples
    # from whatever corpora are present.
    gold_recs = [rec_by_idx[i] for i in g_idx if i in rec_by_idx]
    CB.GOLD_DIR.mkdir(parents=True, exist_ok=True)

    parity_fail = 0
    for gi, rec in enumerate(gold_recs):
        wav_key = str(CB.WAVS24_DIR / f"{rec['idx']}.wav")
        atok_official = CB.load_audio_tokenizer(parity_device)
        for variant in ("tts_instruction", "ref_edit_tata"):
            ex = build_item(tokenizer, rec, variant)
            instruction = ex["_instruction"]

            # ---- parity: (1) plain official prefix; (2) isolated target block

            base_req = CB.ExampleRequest(
                variant=variant, text=rec["text"], instruction=instruction,
                ref_text=rec["text"], ref_audio_path=wav_key)
            prefix_official = _prepare_one(
                tokenizer, atok_official, CB.ConfigStub(16),
                CB.build_segments(base_req, include_target=False))
            target_only = _prepare_one(
                tokenizer, atok_official, CB.ConfigStub(16),
                [{"type": "audio", "append_eos": True, "drop_last_frame": False,
                  "audio_path": wav_key}])

            our_ids = ex["input_ids"][0]
            pre_len = prefix_official["input_ids"].shape[1]
            n_tgt_frames = CB.count_frames_per_block(ex)[-1]
            same_prefix = bool((ex["input_ids"][:, :pre_len] == prefix_official["input_ids"]).all())
            tgt_ids_official = target_only["input_ids"][0]
            same_suffix = bool((our_ids[pre_len:] == tgt_ids_official).all())
            our_tail = ex["input_values"][0, -n_tgt_frames:, :]
            official_tail = target_only["audio_tokens"][0]
            same_vals = bool((our_tail.long() == official_tail.long()).all())
            ok = same_prefix and same_suffix and same_vals
            parity_fail += 0 if ok else 1

            lab = ex["labels"][0]
            vals, cnts = np.unique(lab.numpy(), return_counts=True)
            dist = {str(int(v)): int(c) for v, c in zip(vals, cnts)}
            block_counts = CB.count_frames_per_block(ex)
            fname = CB.GOLD_DIR / f"gold{gi}_{variant}.txt"
            with fname.open("w", encoding="utf-8") as fh:
                fh.write(f"# GOLD {gi} variant={variant}\n")
                fh.write(f"idx={rec['idx']}  dur={rec['dur_proc_s']}s  frames={rec['frames']}\n")
                fh.write(f"instruction={instruction}\n\n")
                fh.write(f"text={rec['text']}\n\n")
                fh.write(f"seq_len={len(our_ids)} blocks_frames={block_counts} "
                         f"(official_prefix={block_counts[:-1] if len(block_counts)>1 else []}, "
                         f"target=last)\n")
                fh.write(f"label_dist={dist}\n")
                dec = tokenizer.decode(ex["input_ids"][0].tolist(), skip_special_tokens=False)
                fh.write(f"input_ids_preview={dec[:400]}\n")
                fh.write(f"parity(prefix,suffix,values) vs raw official code = "
                         f"({same_prefix},{same_suffix},{same_vals})\n")
            torch_save(ex, CB.GOLD_DIR / f"gold{gi}_{variant}.pt")
    # NOTE: the run recipes grep for "parity FAILED for 0/" - keep the wording in sync.
    print(f"[finalize] parity FAILED for {parity_fail}/{len(gold_recs)*2}")

    durations = [r["dur_proc_s"] for r in recs]
    frames_all = [r["frames"] for r in recs]
    by_corpus_stats = {}
    for r in recs:
        c = r.get("corpus", "tata")
        d = by_corpus_stats.setdefault(c, {"n": 0, "s": 0.0})
        d["n"] += 1
        d["s"] += r["dur_proc_s"]
    summary = {
        "total_kept": len(recs),
        "hours_kept": round(sum(durations) / 3600, 2),
        "avg_dur_s": round(sum(durations) / len(recs), 2),
        "max_dur_s": max(durations),
        "frames_avg": round(sum(frames_all) / len(frames_all)),
        "frames_max": max(frames_all),
        "by_corpus": {c: {"n": d["n"], "hours": round(d["s"] / 3600, 2)}
                      for c, d in by_corpus_stats.items()},
        "gold_parities_failed": parity_fail,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[finalize]", json.dumps(summary, ensure_ascii=False))

    manifest_lines = ["idx,corpus,speaker,wav_rel,text,dur_src_s,dur_proc_s,frames,variant"]
    for r in recs:
        safe = r["text"].replace('"', "'")
        manifest_lines.append(
            f"{r['idx']},{r.get('corpus','tata')},{r.get('speaker','tata')},{r['wav_rel']},"
            f"\"{safe}\",{r['dur_src_s']},{r['dur_proc_s']},{r['frames']},"
            f"{deterministic_variant(r['idx'], r)}"
        )
    MANIFEST_CSV.write_text("\n".join(manifest_lines), encoding="utf-8")


def torch_save(ex: dict, path: Path) -> None:
    slim = {
        "input_ids": ex["input_ids"], "attention_mask": ex.get("attention_mask"),
        "text_ids_mask": ex["text_ids_mask"], "text_ids_len": ex["text_ids_len"],
        "input_values": ex["input_values"].int(), "labels": ex["labels"],
        "_variant": ex["_variant"], "_n_frames": ex["_n_frames"],
    }
    torch.save(slim, path)


# ------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["process", "finalize"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gold", type=int, default=3)
    ap.add_argument("--parity-device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, str(CB.REPO))
    if args.cmd == "process":
        process(args.limit, args.device)
    else:
        finalize(args.gold, args.parity_device)


if __name__ == "__main__":
    main()
