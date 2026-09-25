"""train_lora.py — PHASE C: LoRA training of Breeze TTS 2 on a single 16 GB-class GPU.

(The pt-BR template targeted an RTX 4060 Ti 16 GB; the German runs of this fork
ran on an AMD Radeon AI PRO R9700 with a peak of ~15 GB.)

Implemented gates:
  G1: LoraConfig with an EXPLICIT EXCLUSION of the whole codec_model (exclude_modules).
  G2 (--smoke mode): N short steps that print
      - print_trainable_parameters() + a dump of the trainable module list
        (automatic assert: nothing trainable outside backbone/depth/text_encoder);
      - loss stability (mean per window, no NaN);
      - peak VRAM (< 13-14 GB required on the template's 16 GB card).

Usage:
  SMOKE : python train_lora.py --run smoke --smoke --steps 30
  FULL  : python train_lora.py --run myrun --epochs 3

Audible samples:
  training/runs/<run>/samples/checkpoint-<tag>/<NN>_<slug>.wav
  (after every epoch and at the end; plus every --sample-every-steps if set)
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
import time
from pathlib import Path

_CORE = Path(__file__).resolve().parent
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

import common_breeze as CB
import prepare_dataset as PD

CB.GOLD_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CB.REPO))

import numpy as np
import torch

DEV = "cuda"

# Probe sentences for the audible samples and the WER check. The suffix of the
# name (-de/-en/-zh) selects the ASR language in eval_wer.py. (Test data: the
# German sentences stay German.)
SAMPLE_TEXTS = [
    ("hallo-de", "Guten Tag! Dies ist eine Sprachprobe auf Deutsch."),
    ("umlaute-de", "Über den Dächern der Großstadt zogen schwere Gewitterwolken auf."),
    ("zahlen-de",
     "Meine Hausnummer ist fünfzehn null zwei, im Stadtteil Gartenfeld."),
    ("wetter-de",
     "Die Vorhersage meldet am Nachmittag kräftige Schauer, bei Temperaturen "
     "zwischen sechzehn und dreiundzwanzig Grad."),
    ("komposita-de",
     "Die Straßenbahnhaltestelle vor dem Hauptbahnhof wird derzeit umgebaut."),
    ("scharf-s-de", "Er saß draußen auf der weißen Bank und genoß die Stille."),
    ("affektiv-de", "Wie schön, deine Stimme nach all den Jahren wiederzuhören!"),
    # Regression probes: the base model speaks EN and ZH out of the box. If the
    # WER drifts away here during training, the German LoRA is eating the
    # original languages — you want to see that early, not at the end.
    ("regression-en", "The weather today is sunny with a gentle breeze from the east."),
    ("regression-zh", "今天天气晴朗，东边吹来一阵微风。"),
]
SAMPLES_PER_EVENT = len(SAMPLE_TEXTS)

TARGET_PRESETS = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "all": ["q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"],
}
# The template's pt-BR weights; corpora not listed here get 1.0 (override with --corpus-weights).
DEFAULT_CORPUS_WEIGHTS = {"tagarela": 1.0, "cetuc": 1.0, "cml_pt": 2.0,
                          "podcast": 2.0, "tata": 2.0}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower())[:40]


# ------------------------------------------------------------------ data


class TrainDataset(torch.utils.data.Dataset):
    """Items of the training split, assembled on the fly via the code cache."""

    def __init__(self, idxs: list[str], tokenizer, augment: bool = False):
        self.tokenizer = tokenizer
        self.augment = augment          # noise-augmented references only in training, never in val
        self.records = {r["idx"]: r for r in PD.load_meta()}
        self.idxs = [i for i in idxs if i in self.records]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i: int) -> dict:
        rec = self.records[self.idxs[i]]
        variant = PD.deterministic_variant(rec["idx"], rec)
        ex = PD.build_item(self.tokenizer, rec, variant, augment=self.augment)
        return {
            "input_ids": ex["input_ids"][0],
            "attention_mask": torch.ones_like(ex["input_ids"][0]),
            "text_ids_mask": ex["text_ids_mask"][0],
            "text_ids_len": ex["text_ids_len"],
            "input_values": ex["input_values"].squeeze(0),
            "labels": ex["labels"][0],
        }


def load_split(name: str) -> list[str]:
    return [
        l.strip()
        for l in open(CB.TRAINING / f"splits_{name}.txt", encoding="utf-8")
        if l.strip()
    ]


def stratified_val_picks(ds_val: TrainDataset, n_items: int) -> list[tuple[int, str]]:
    """Sample stratified by corpus (proportional to size), deterministic."""
    by: dict[str, list[int]] = {}
    for i, idx in enumerate(ds_val.idxs):
        rec = ds_val.records.get(idx, {})
        by.setdefault(rec.get("corpus", "tata"), []).append(i)
    total = max(1, len(ds_val))
    picks: list[tuple[int, str]] = []
    for corp in sorted(by):
        lst = by[corp]
        k = max(1, round(n_items * len(lst) / total))
        step = max(1, len(lst) // k)
        for i in lst[::step][:k]:
            picks.append((i, corp))
    return picks


def quick_val_loss(raw, ds_val: TrainDataset, n_items: int = 96):
    """Val loss STRATIFIED by corpus. Returns (global micro average, {corpus: loss})."""
    picks = stratified_val_picks(ds_val, n_items)
    by_corp: dict[str, list[int]] = {}
    for i, c in picks:
        by_corp.setdefault(c, []).append(i)
    per: dict[str, float] = {}
    all_l: list[float] = []
    was_training = raw.training
    raw.eval()
    with torch.no_grad():
        for corp, lst in by_corp.items():
            ls = []
            for j in range(0, len(lst), 2):
                items = [ds_val[k] for k in lst[j:j + 2]]
                batch = CB.collate(items)
                batch = {k: (v.to(DEV) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}
                ls.append(raw(**batch).loss.item())
            per[corp] = float(np.mean(ls)) if ls else float("nan")
            all_l.extend(ls)
    if was_training:
        raw.train()
    overall = float(np.mean(all_l)) if all_l else float("nan")
    return overall, per


# -------------------------------------------------------------- WAV samples


def sanitize_inference_tensors(model) -> int:
    """Neutralizes tensors created under inference_mode that may have got stuck
    in persistent module BUFFERS (e.g. rope inv_freq rebinds) and drops the
    cudagraph/fast-path caches attached to the model object.
    Avoids 'Inference tensors cannot be saved for backward' during training."""
    n_fixed = 0
    for mod in model.modules():
        for name, buf in list(mod.named_buffers(recurse=False)):
            if isinstance(buf, torch.Tensor) and buf.is_inference():
                setattr(mod, name, buf.clone())
                n_fixed += 1
    for attr in (
        "_fast_text_encoder_cudagraph",
        "_fast_text_encoder_graph_cache",
        "_backbone_graph",
        "_depth_decoder_graph",
        "_stream_runtime",
    ):
        if hasattr(model, attr):
            try:
                delattr(model, attr)
            except Exception:
                pass
    return n_fixed


def generate_samples(raw, tokenizer, out_dir, tag: str, seed0: int = 1000):
    """EAGER sampling via BreezeForConditionalGeneration.generate(output_audio=True).

    Does NOT use FastBreezeStreamingRuntime/iter_audio_chunks: that path runs under
    @torch.inference_mode() and builds cudagraphs ON THE TRAINING OBJECT, leaving
    inference tensors stuck in the modules -> backward crash in the next epoch.
    """
    import soundfile as sf
    from breeze_infer.runtime import set_all_seeds, update_generation_config_for_breeze
    from breeze_infer.templates import get_template, prepare_inputs

    audio_tok = CB.load_audio_tokenizer(DEV)
    sr_out = audio_tok.get_output_sample_rate()
    update_generation_config_for_breeze(raw)

    out_dir.mkdir(parents=True, exist_ok=True)
    was_training = raw.training
    raw.eval()
    template = get_template("tts_instruction")
    try:
        with torch.no_grad():  # no_grad only: does not contaminate the model
            for k, (name, text) in enumerate(SAMPLE_TEXTS[:SAMPLES_PER_EVENT]):
                request = {
                    "id": f"eval-{tag}-{k}",
                    "text": text,
                    # Instruction in the language of the probe, as in training:
                    # "-en" gets the English one, everything else the German one.
                    # A German instruction in front of English text would make
                    # the regression probe look worse than the model really is.
                    "instruction": ("Speak clearly and naturally."
                                    if name.endswith("-en")
                                    else "Sprich klar und natuerlich."),
                    "speaker": "S0",
                }
                set_all_seeds(seed0 + k)
                inputs = prepare_inputs(
                    tokenizer,
                    audio_tok,
                    raw,
                    [request],
                    template,
                    guidance_scale=1.0,
                    guidance_scale_ref=None,
                    guidance_scale_ins=None,
                )
                inputs.pop("input_values", None)  # purely textual prompt
                inputs.pop("cfg_scale", None)
                path = out_dir / f"{k:02d}_{slug(name)}.wav"
                try:
                    res = raw.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        text_ids_mask=inputs["text_ids_mask"],
                        text_ids_len=inputs["text_ids_len"],
                        output_audio=True,
                        audio_tokenizer=audio_tok,
                        max_new_tokens=400,
                        do_sample=True,
                        temperature=0.9,
                        top_k=50,
                        top_p=1.0,
                    )
                    if isinstance(res, (list, tuple)):
                        wav_t = res[0]
                    elif hasattr(res, "audio"):
                        wav_t = res.audio[0] if res.audio else None
                    else:
                        wav_t = res
                    if wav_t is None:
                        print(f"    [samples] '{name}': no audio returned")
                        continue
                    while wav_t.dim() > 1:
                        wav_t = wav_t[0]
                    wav = wav_t.detach().float().cpu().numpy()
                    sf.write(str(path), np.clip(wav, -1.0, 1.0), sr_out, subtype="PCM_16")
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"    [samples] ERROR '{name}': {type(exc).__name__}: {str(exc)[:140]}"
                    )
    finally:
        n_fix = sanitize_inference_tensors(raw)
        del audio_tok
        torch.cuda.empty_cache()
        if was_training:
            raw.train()
        if n_fix:
            print(f"    [sanitize] {n_fix} buffers restored inference->normal")


# ------------------------------------------------------------------ main



def wort_ausgleich(ds, weights, liste, ziel, deckel, min_zipf=4.0):
    """Rare-word re-weighting: each clip is weighted by its rarest word.

    ("Wortausgleich" = word balancing; liste = word list JSON from
    tools/analyse_seltene_woerter.py, ziel = target count, deckel = cap.)

    A word that occurs c times in the training split gives the factor
    ziel / c, at most `deckel`, at least 1. That is duplication in expectation
    - the WeightedRandomSampler draws with replacement - without duplicate
    files and without any duplicate being able to end up in val/test.

    Counts are taken in the training split itself, not in the original list:
    what Thorsten, CML and Common Voice contributed later is counted too.
    """
    import json as _json
    import re as _re
    wort = _re.compile(r"[A-Za-zÄÖÜäöüß]+")
    # Only words that are frequent in present-day German. Without this threshold
    # (all 14,500 rare words) 37 % of the clips contained at least one of them -
    # then 78 % of the draws went to "rare" clips, and the normalization ate up
    # the gain for the important words again.
    selten = {r["wort"].casefold() for r in _json.load(open(liste, encoding="utf-8"))
              if r["zipf"] >= min_zipf}
    texte = [ds.records.get(i, {}).get("text", "") for i in ds.idxs]
    woerter = [{w.casefold() for w in wort.findall(t)} & selten for t in texte]
    anzahl = collections.Counter(w for ws in woerter for w in ws)
    neu = []
    for g, ws in zip(weights, woerter):
        f = max(1.0, min(deckel, ziel / min(anzahl[w] for w in ws))) if ws else 1.0
        neu.append(g * f)

    n = len(neu); summe = sum(neu)
    hoch = [x for x in neu if x > 1.0001]
    # Expected draws per epoch: length n, probability w/sum
    normal = n / summe
    print(f"[reweight] {len(anzahl)} rare words in the training split, {len(hoch)} clips up-weighted "
          f"(mean factor {sum(hoch)/max(1,len(hoch)):.2f})")
    print(f"[reweight] share of draws on up-weighted clips: "
          f"{100*sum(hoch)/summe:.1f} %  |  a normal clip is drawn "
          f"{normal:.2f}x per epoch (before: 1.00x)")
    def ausgesetzt(w):
        return sum(n * x / summe for x, ws in zip(neu, woerter) if w in ws)
    for w in ("temperaturen", "telefon", "informationen", "projekt", "online", "internet"):
        if w in anzahl:
            print(f"[reweight]   {w:<14} {anzahl[w]:>4} clips  ->  ~{ausgesetzt(w):.0f} draws per epoch")
    return neu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--grad-acc", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--targets", choices=sorted(TARGET_PRESETS), default="attn",
                    help="attn = q/k/v/o | all = + gate/up/down (MLPs)")
    ap.add_argument("--use-rslora", action="store_true")
    # Noise-augmented references (run 5; "Rausch" = noise). --rausch-anteil is the
    # share of clean-corpus examples that get a degraded reference plus the
    # clean-studio instruction; --rausch-korpora lists the clean corpora.
    ap.add_argument("--rausch-codes", default=None,
                    help="code store of the degraded references (tools/rauschvorlagen.py kodieren)")
    ap.add_argument("--rausch-anteil", type=float, default=0.35)
    ap.add_argument("--rausch-sauber", type=float, default=0.25,
                    help="share of clean examples that only get the clean-studio instruction")
    ap.add_argument("--rausch-korpora", default="hui_de,thorsten_rare_de,hifitts2_en")
    # Rare-word re-weighting per clip (instead of per corpus). Path to the list
    # from tools/analyse_seltene_woerter.py; empty = off. ("wa" = Wortausgleich:
    # --wa-ziel = target, --wa-max = cap per clip, --wa-min-zipf = frequency floor.)
    ap.add_argument("--wort-ausgleich", default="")
    ap.add_argument("--wa-ziel", type=float, default=50.0,
                    help="target number of training examples per rare word")
    ap.add_argument("--wa-max", type=float, default=8.0,
                    help="cap per clip - higher values make the model memorize single recordings")
    ap.add_argument("--wa-min-zipf", type=float, default=4.0,
                    help="only words that are at least this frequent today (Zipf scale)")
    ap.add_argument("--ref-edit-frac", type=float, default=0.9,
                    help="fraction of examples in ref_edit mode (with a speaker reference)")
    ap.add_argument("--val-items", type=int, default=96)
    ap.add_argument("--corpus-weights", type=str, default=None,
                    help="JSON {corpus: weight}; default = the template's pt-BR weights (24 kHz+ corpora 2x), others 1.0")
    ap.add_argument("--no-tensorboard", action="store_true")
    ap.add_argument("--no-wer", action="store_true",
                    help="disable the automatic WER/CER per checkpoint (faster-whisper on CPU)")
    ap.add_argument("--sample-every-steps", type=int, default=500)
    ap.add_argument(
        "--resume-adapter",
        type=str,
        default=None,
        help="path to an adapter folder to continue training from",
    )
    args = ap.parse_args()

    batch_size = args.batch or (2 if args.smoke else 4)
    grad_acc = args.grad_acc or (2 if args.smoke else 8)

    run_dir = CB.TRAINING / "runs" / args.run
    ckpt_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "samples"
    for d in (run_dir, ckpt_dir, samples_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "log.csv"
    log_f = log_path.open("a", encoding="utf-8")
    if log_f.tell() == 0:
        log_f.write(
            "step,epoch,opt_step,total_loss,backbone_loss,depth_loss,"
            "val_loss,lr,vram_win_gb,vram_peak_gb,s_per_step\n"
        )

    print(
        f"[cfg] run={args.run} smoke={args.smoke} batch={batch_size} acc={grad_acc} "
        f"lr={args.lr} r={args.rank} a={args.alpha}"
    )

    # ---------------------------------------------------------- base model
    t0 = time.time()
    raw = CB.load_breeze_model(DEV)
    raw.gradient_checkpointing_enable()
    raw.enable_input_require_grads()
    print(f"[load] ready in {time.time() - t0:.0f}s | gradient checkpointing ON")

    from peft import LoraConfig, get_peft_model

    if args.resume_adapter:
        from peft import PeftModel

        # Note: when resuming, the LoRA shape (rank, alpha, targets, rsLoRA)
        # comes from the adapter; --rank/--alpha/--targets/--use-rslora are ignored.
        print(f"[resume] loading existing adapter from: {args.resume_adapter}")
        pm = PeftModel.from_pretrained(raw, args.resume_adapter, is_trainable=True)
    else:
        from peft import LoraConfig, get_peft_model

        lconf = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=TARGET_PRESETS[args.targets],
            use_rslora=args.use_rslora,
            exclude_modules=r".*codec_model.*",
        )
        pm = get_peft_model(raw, lconf)
    pm.print_trainable_parameters()

    # ------------------------- GATE G1: auditable + scope asserts
    PREFIX = "base_model.model."  # PeftModel -> LoraModel -> raw
    trainable_lines, stats, bad = (
        [],
        {"backbone_model": 0, "depth_decoder": 0, "text_encoder": 0},
        [],
    )
    for name, p in pm.named_parameters():
        if not p.requires_grad:
            continue
        trainable_lines.append(name)
        core = name.removeprefix(PREFIX)
        hit = next((s for s in stats if core.startswith(s + ".")), None)
        if hit is None:
            bad.append((name, tuple(p.shape)))
        else:
            stats[hit] += p.numel()
    assert not bad, f"GATE G1 VIOLATED - trainable parameters outside the stacks: {bad[:10]}"
    assert not any(".codec_model." in n for n in trainable_lines), "codec among the targets!"
    n_train = sum(stats.values())
    print(f"[G1] trainable per stack: {stats} | total={n_train:,}")
    (run_dir / "trainable_modules.txt").write_text(
        "\n".join(sorted(trainable_lines)), encoding="utf-8"
    )

    # ---------------------------------------------------------- data
    tokenizer = CB.load_text_tokenizer()
    PD.set_ref_edit_frac(args.ref_edit_frac)
    if args.rausch_codes:
        PD.set_rauschen(args.rausch_codes, args.rausch_anteil, args.rausch_sauber,
                        [k for k in args.rausch_korpora.split(",") if k])
    ds = TrainDataset(load_split("train"), tokenizer, augment=bool(args.rausch_codes))
    if args.rausch_codes:
        from collections import Counter as _C
        _w = _C(PD.rausch_wahl(i) for i in ds.idxs
                if ds.records.get(i, {}).get("corpus") in PD._RAUSCH["korpora"])
        print(f"[noise] clean corpora: {dict(_w)} | degraded codes: "
              f"{sum(1 for i in ds.idxs if i in PD._RAUSCH['leser'])}", flush=True)
    ds_val = TrainDataset(load_split("val"), tokenizer)
    from collections import Counter as _Counter
    _vdist = _Counter(PD.deterministic_variant(i, ds.records.get(i))
                      for i in ds.idxs[::max(1, len(ds.idxs) // 4000)])
    print(f"[data] ref_edit_frac={args.ref_edit_frac} | variants (sample)={dict(_vdist)}")
    cw = dict(DEFAULT_CORPUS_WEIGHTS)
    if args.corpus_weights:
        cw.update(json.loads(args.corpus_weights))
    weights = [float(cw.get(ds.records.get(i, {}).get("corpus", "tata"), 1.0))
               for i in ds.idxs]
    from collections import Counter as _Counter
    if args.wort_ausgleich:
        weights = wort_ausgleich(ds, weights, args.wort_ausgleich, args.wa_ziel, args.wa_max,
                                 args.wa_min_zipf)
    print(f"[data] corpus weights={cw} | "
          f"dist={dict(_Counter(ds.records.get(i, {}).get('corpus', 'tata') for i in ds.idxs))}")
    steps_per_epoch = math.ceil(len(ds) / (batch_size * grad_acc))
    opt_steps_total = args.steps if args.smoke else steps_per_epoch * args.epochs
    print(
        f"[data] items={len(ds)} val={len(ds_val)} "
        f"steps/epoch={steps_per_epoch} total_target={opt_steps_total}"
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "batch_size": batch_size,
                "grad_acc": grad_acc,
                "n_train_items": len(ds),
                "opt_steps_total": opt_steps_total,
                "trainable_params": n_train,
                "per_stack": stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    opt = torch.optim.AdamW(
        (p for p in pm.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=0.01,
        eps=1e-8,
    )

    def lr_lambda(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, opt_steps_total - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    trainable_params = [p for p in pm.parameters() if p.requires_grad]

    def save_checkpoint(tag_s: str):
        d = ckpt_dir / tag_s
        pm.save_pretrained(str(d), safe_serialization=True)
        print(f"[ckpt] saved: {d}")
        return d

    tb = None
    if not args.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb = SummaryWriter(log_dir=str(run_dir / "tb"))
            print(f"[tb] TensorBoard -> {run_dir / 'tb'}")
        except Exception as exc:  # noqa: BLE001
            print(f"[tb] unavailable: {type(exc).__name__}: {exc}")

    def run_wer(sample_dir: Path):
        if args.no_wer:
            return
        try:
            # de_lora/eval is not on sys.path — without this the import fails
            # and the WER evaluation silently drops out.
            _EVAL_DIR = str(Path(__file__).resolve().parents[1] / "eval")
            if _EVAL_DIR not in sys.path:
                sys.path.insert(0, _EVAL_DIR)
            import eval_wer

            res = eval_wer.evaluate_dir(sample_dir, SAMPLE_TEXTS)
            if not res:
                return
            wer = float(np.mean([r["wer"] for r in res]))
            cer = float(np.mean([r["cer"] for r in res]))
            print(f"[wer] {sample_dir.name}: WER={wer:.3f} CER={cer:.3f} "
                  f"({len(res)} samples)")
            if tb is not None:
                tb.add_scalar("wer", wer, global_step)
                tb.add_scalar("cer", cer, global_step)
                tb.flush()
            (sample_dir / "wer.json").write_text(
                json.dumps({"wer": wer, "cer": cer, "per_sample": res},
                           ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            print(f"[wer] failed: {type(exc).__name__}: {str(exc)[:120]}")

    global_step = 0
    t_run = time.time()
    stop = False
    torch.cuda.reset_peak_memory_stats()

    epochs_to_run = 1 if args.smoke else args.epochs
    for epoch in range(epochs_to_run):
        if stop:
            break
        g = torch.Generator()
        g.manual_seed(42 + epoch)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double), len(ds),
            replacement=True, generator=g)
        order = list(sampler)
        micro: list[dict] = []
        win = {"n_micro": 0, "loss": 0.0, "b": 0.0, "d": 0.0}
        win_peak0 = torch.cuda.max_memory_allocated()
        t_win = time.time()
        steps_win = 0
        raw.train()
        for i in order:
            micro.append(ds[i])
            if len(micro) < batch_size:
                continue
            batch = CB.collate(micro)
            batch = {
                k: (v.to(DEV) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            micro.clear()
            out = raw(**batch)
            (out.loss / grad_acc).backward()
            win["n_micro"] += 1
            win["loss"] += out.loss.item()
            win["b"] += out.backbone_loss.item()
            win["d"] += out.depth_decoder_loss.item()
            if win["n_micro"] % grad_acc != 0:
                continue

            gnorm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1
            steps_win += 1

            mean_l = win["loss"] / win["n_micro"]
            if global_step <= 3 or global_step % 5 == 0:
                # t_win is only reset when logging, i.e. every 5 steps. The
                # template divided the window duration by grad_acc instead — and
                # the output below divided once more, so in total by grad_acc^2
                # instead of by the number of steps. The displayed value was
                # therefore 5*grad_acc times the truth.
                sp = (time.time() - t_win) / max(steps_win, 1)
                win_peak = torch.cuda.max_memory_allocated()
                mean_b = win["b"] / win["n_micro"]
                mean_d = win["d"] / win["n_micro"]
                line = (
                    f"{global_step},{epoch},{global_step},{mean_l:.4f},"
                    f"{mean_b:.4f},{mean_d:.4f},nan,"
                    f"{sched.get_last_lr()[0]:.2e},"
                    f"{(win_peak - win_peak0) / 2**30:.2f},{win_peak / 2**30:.2f},{sp:.3f}\n"
                )
                log_f.write(line)
                log_f.flush()
                print(
                    f"[{global_step}/{opt_steps_total}] ep{epoch} loss={mean_l:.3f} "
                    f"(b{mean_b:.3f}/d{mean_d:.3f}) "
                    f"g={float(gnorm):.2f} vrwin={(win_peak - win_peak0) / 2**30:.2f}GB "
                    f"pk={win_peak / 2**30:.2f}GB {sp * 1000:.0f}ms/st"
                )
                if tb is not None:
                    tb.add_scalar("loss/total", mean_l, global_step)
                    tb.add_scalar("loss/backbone", mean_b, global_step)
                    tb.add_scalar("loss/depth", mean_d, global_step)
                    tb.add_scalar("lr", sched.get_last_lr()[0], global_step)
                    tb.add_scalar("grad_norm", float(gnorm), global_step)
                    tb.add_scalar("vram_peak_gb", win_peak / 2**30, global_step)
                    tb.flush()
                win = {"n_micro": 0, "loss": 0.0, "b": 0.0, "d": 0.0}
                win_peak0 = torch.cuda.max_memory_allocated()
                t_win = time.time()
                steps_win = 0

            if args.smoke and global_step >= args.steps:
                stop = True
                break
            if (
                not args.smoke
                and global_step > 0
                and global_step % args.sample_every_steps == 0
            ):
                save_checkpoint(f"step{global_step}")
                generate_samples(
                    raw,
                    tokenizer,
                    samples_dir / f"checkpoint-{global_step}",
                    f"s{global_step}",
                    seed0=1000 + global_step,
                )

        if not args.smoke:
            vl, per_corpus = quick_val_loss(raw, ds_val, n_items=args.val_items)
            log_f.write(
                f"{global_step},{epoch},{global_step},nan,nan,nan,{vl:.4f},"
                f"nan,nan,{torch.cuda.max_memory_allocated() / 2**30:.2f},nan\n"
            )
            log_f.flush()
            msg = " ".join(f"{c}={v:.3f}" for c, v in sorted(per_corpus.items()))
            print(f"[epoch {epoch}] val_loss={vl:.4f} | {msg}")
            if tb is not None:
                tb.add_scalar("val/loss", vl, global_step)
                for c, v in per_corpus.items():
                    tb.add_scalar(f"val/{c}", v, global_step)
                tb.flush()
            save_checkpoint(f"epoch{epoch}_val{vl:.3f}".replace(".", "_"))
            ep_samples = samples_dir / f"checkpoint-epoch{epoch}"
            generate_samples(raw, tokenizer, ep_samples, f"ep{epoch}", seed0=2000 + epoch)
            run_wer(ep_samples)

    # Known issue: after the final checkpoint the process can hang in the WER
    # evaluation below (faster-whisper on CPU); the adapter is already written by
    # then. recipes/pipeline_v5.sh waits up to 20 min and then stops the trainer.
    final_d = save_checkpoint("final")
    final_samples = samples_dir / "final"
    generate_samples(raw, tokenizer, final_samples, "final", seed0=9000)
    run_wer(final_samples)
    if tb is not None:
        tb.close()
    dt_min = (time.time() - t_run) / 60
    peak_all = torch.cuda.max_memory_allocated() / 2**30
    print(f"[DONE] steps={global_step} time={dt_min:.1f}min vram_peak={peak_all:.2f}GB")
    print(f"[OUTPUT] final adapter: {final_d}")
    print(f"[LISTEN] wavs in: {samples_dir}")


if __name__ == "__main__":
    main()
