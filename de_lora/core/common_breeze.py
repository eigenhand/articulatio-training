"""common_breeze.py — shared module for phases B/C of the LoRA training.

Single source of truth for:
- project paths;
- the instruction pools (German + English in this fork);
- building training examples in the EXACT format of Breeze's forward()
  (reuses breeze_infer.templates._prepare_one verbatim, with a cache patch for
  pre-extracted codes so audio is not re-encoded on every access);
- label assembly (markers 262144 / -101 / 262145 / -100);
- a collate with left padding that replicates the official _collate_inputs (+ labels).

Does not modify any file of the breeze-tts engine.
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class _DropMessage(logging.Filter):
    """Drops one specific log message (by substring)."""

    def __init__(self, needle: str) -> None:
        super().__init__()
        self._needle = needle

    def filter(self, record: logging.LogRecord) -> bool:
        return self._needle not in record.getMessage()


def _silence_known_warnings() -> None:
    # `transformers` warns about an "incorrect regex pattern" and suggests
    # fix_mistral_regex=True, but that flag breaks with the installed version
    # (TypeError in tokenizers). The model was trained with this tokenizer, so we
    # keep the default behaviour and silence only this specific message.
    logging.getLogger("transformers.tokenization_utils_base").addFilter(
        _DropMessage("incorrect regex pattern")
    )


_silence_known_warnings()


@contextlib.contextmanager
def _silence_stdout():
    """Silences library prints during an import (e.g. the qwen_tts banner)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def import_qwen_tts():
    """Imports Qwen3TTSTokenizer while suppressing qwen_tts' flash-attn banner.

    On import, `qwen_tts` prints a warning when flash-attn is not installed (the
    audio tokenizer then falls back to the plain PyTorch path — functional, just
    slower). Installing flash-attn is not practical on Windows (where the template
    was developed), so the banner is suppressed.
    """
    with _silence_stdout():
        from qwen_tts import Qwen3TTSTokenizer

    return Qwen3TTSTokenizer


import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------------------------------------------ paths
# Paths come from de_lora/core/paths.py (driven by PTBR_ARTIFACTS / .env).
# The engine (breeze-tts) sits at the root of this fork; artifacts stay outside git.
ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = Path(__file__).resolve().parent
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import paths  # noqa: E402

REPO = paths.BREEZE_REPO
CKPT = paths.CKPT
DATASET = paths.DATASET
TRAINING = paths.TRAINING
TOKENS_DIR = paths.TOKENS_DIR
WAVS24_DIR = paths.WAVS24_DIR
GOLD_DIR = paths.GOLD_DIR

TEXTS_CSV = paths.TEXTS_CSV

# ------------------------------------------------------------- multi-corpus
# Corpus registry (datasets/corpora.json). Backwards compatible: without the file
# it falls back to the legacy pair (tata + podcast) driven by BREEZE_DATASET_DIR.
DATASETS_ROOT = paths.DATASETS_ROOT
CORPORA_JSON = paths.CORPORA_JSON
SCRAPING_WORK = paths.SCRAPING_WORK

_LEGACY_CORPORA = [
    {"name": "tata", "root": "TTS-Portuguese-Corpus", "csv": "texts.csv"},
    {"name": "podcast", "root": "podcast", "csv": "texts.csv"},
]


def load_corpora() -> list[dict]:
    """List of active corpora. Each item: {name, root, csv[, speakers]}."""
    import json

    if CORPORA_JSON.exists():
        data = json.loads(CORPORA_JSON.read_text(encoding="utf-8"))
        return [c for c in data if c.get("enabled", True)]
    return list(_LEGACY_CORPORA)


def corpus_dir(corp: dict) -> Path:
    return DATASETS_ROOT / corp["root"]


def corpus_csv(corp: dict) -> Path:
    return corpus_dir(corp) / corp.get("csv", "texts.csv")
SR = 24_000                    # target sample rate of the model
MAX_DUR_S = 10.2               # drop clips longer than this (text/audio integrity)
MIN_DUR_S = 0.98
PEAK_NORM = 0.95

sys.path.insert(0, str(REPO))

AUDIO_TOKEN_ID = 262144        # <|AUDIO|>       (frame marker)
AUDIO_EOS_TOKEN_ID = 262145    # <|audio_eos|>
BACKBONE_EOS_CLASS = None      # resolved via the config (vocab_size = 2051)
IGNORE_IDX = -100              # ignored by the backbone and the depth decoder
DEPTH_IGNORE = -101            # frame used ONLY by the backbone (codebook 0)

# German, matching the target corpus (audiobook readings). The template used
# Portuguese sentences here; we follow the same pattern, because in training the
# instruction and the content are in the same language anyway, and that is how
# the pt-BR run worked.
# (Training data: keep these strings as they are, including the ASCII
# transliterations of the umlauts.)
INSTRUCTION_POOL = [
    "Sprich klar und natuerlich.",
    "Lies den Text mit deutlicher Aussprache und ruhigem Tempo.",
    "Erzaehle mit neutraler Stimme und natuerlichem Tonfall.",
    "Lies laut vor, mit sorgfaeltiger Aussprache.",
    "Sprich mit ruhigem, gemaessigtem Ton.",
    "Lies den Text im Tonfall einer sachlichen Lesung.",
    "Erzeuge eine saubere, ausdrucksvolle Lesung.",
    "Sprich natuerlich, wie ein deutscher Erzaehler.",
]

# English counterparts. In mixed training a German instruction in front of
# English text would be contradictory: the model is supposed to learn which
# language to speak, and the instruction is the clearest signal for that.
INSTRUCTION_POOL_EN = [
    "Speak clearly and naturally.",
    "Read the text with clear diction and a calm pace.",
    "Narrate in a neutral voice with a natural delivery.",
    "Read aloud with careful pronunciation.",
    "Speak in a calm, measured tone.",
    "Read the text in the tone of an informative reading.",
    "Produce a clean, expressive reading.",
    "Speak naturally, like an English narrator.",
]


def instruction_pool_for(corpus: str) -> list:
    """Pool by corpus language: suffix _en -> English, anything else -> German."""
    return INSTRUCTION_POOL_EN if str(corpus).endswith("_en") else INSTRUCTION_POOL

# ------------------------------------------------------------------ loaders


def load_text_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(CKPT)


def load_audio_tokenizer(device: str = "cuda"):
    Qwen3TTSTokenizer = import_qwen_tts()

    return Qwen3TTSTokenizer.from_pretrained(str(CKPT / "audio_tokenizer"), device_map=device)


def load_breeze_model(device: str = "cuda", attn: str = "eager"):
    from models.breeze import BreezeForConditionalGeneration

    model = BreezeForConditionalGeneration.from_pretrained(
        CKPT, dtype=torch.bfloat16, attn_implementation=attn
    )
    return model.to(device).eval()


# --------------------------------------------------------- code cache patch
#
# templates._prepare_one -> _resolve_segment_audio_codes -> _encode_prompt_audio.
# The patch replaces ONLY the final encode function inside the templates
# namespace; when the path is in the cache we return the tensor saved in phase B,
# otherwise we fall back to the original encode (used by the parity check).

_cache_store: dict[str, torch.Tensor] = {}
_orig_encode_prompt_audio = None


def enable_code_cache() -> None:
    """Enables the global cache (manual registration via register_codes())."""
    global _orig_encode_prompt_audio
    import breeze_infer.templates as T

    if _orig_encode_prompt_audio is not None:
        return

    def _patched(audio_tokenizer, audio_path):
        key = str(audio_path)
        hit = _cache_store.get(key)
        if hit is not None:
            return hit.clone()
        # Since the switch to the consolidated code store there are no files
        # under wavs24/ any more — the paths are pure cache keys. A miss is
        # therefore always a programming error (a forgotten register_codes*),
        # not a missing file. Without this check the fallback would end in an
        # uninformative FileNotFoundError.
        if str(WAVS24_DIR) in key:
            raise KeyError(
                f"Codes for {key} are not registered. register_codes_array() must be "
                f"called before an example is assembled.")
        return _orig_encode_prompt_audio(audio_tokenizer, audio_path)

    _orig_encode_prompt_audio = T._encode_prompt_audio
    T._encode_prompt_audio = _patched


def disable_code_cache() -> None:
    global _orig_encode_prompt_audio
    if _orig_encode_prompt_audio is None:
        return
    import breeze_infer.templates as T

    T._encode_prompt_audio = _orig_encode_prompt_audio
    _orig_encode_prompt_audio = None


def register_codes(wav_path: str | Path, codes_npz_path: str | Path) -> int:
    """Loads codes (.npz) into the cache under the key wav_path. Returns n_frames."""
    enable_code_cache()
    arr = np.load(codes_npz_path)["codes"]
    assert arr.ndim == 2 and arr.shape[1] == 16 and arr.dtype == np.int16
    _cache_store[str(Path(wav_path))] = torch.from_numpy(np.ascontiguousarray(arr))
    return int(arr.shape[0])


def register_codes_array(wav_path: str | Path, arr: np.ndarray) -> int:
    """Like register_codes, but from an array that is already loaded.

    Counterpart of the consolidated code store (de_lora/core/codestore.py):
    there the codes come from a memmap slice instead of an .npz file.
    """
    enable_code_cache()
    assert arr.ndim == 2 and arr.shape[1] == 16 and arr.dtype == np.int16, \
        (arr.shape, arr.dtype)
    _cache_store[str(Path(wav_path))] = torch.from_numpy(np.ascontiguousarray(arr))
    return int(arr.shape[0])


class ConfigStub:
    """Lightweight stand-in for model.config for _prepare_one (uses only num_codebooks)."""

    def __init__(self, num_codebooks: int = 16):
        self.num_codebooks = num_codebooks


@dataclass
class ExampleRequest:
    variant: str                 # 'tts_instruction' | 'ref_edit_tata'
    text: str                    # target text (transcript)
    instruction: str
    ref_text: str = ""           # same as text in the clone variant (transcript of the ref)
    ref_audio_path: str = ""     # processed 24 kHz wav (cache key)
    target_audio_path: str = ""  # wav of the SUPERVISED speech (the same clip here)


def build_segments(req: ExampleRequest, include_target: bool = True) -> list[dict]:
    """Training segments = official template + an appended TARGET AUDIO BLOCK.

    At inference the official templates end with the text (the audio is GENERATED).
    In training (CSM style) the sequence has to carry the supervised audio:
      tts_instruction -> [text]               + [target audio]
      ref_edit_tata   -> [ref text, ref audio,
                          text]               + [target audio]
    include_target=False returns the plain template (used by the parity check).
    """
    from breeze_infer import templates as T

    r = {
        "speaker": "S0",
        "text": req.text,
        "instruction": req.instruction,
    }
    if req.variant == "tts_instruction":
        segments = T._tts_instruction_segments(r)
    elif req.variant == "ref_edit_tata":
        r["ref_audio_path"] = req.ref_audio_path
        r["ref_text"] = req.ref_text
        segments = T._ref_edit_tata_segments(r)
    else:
        raise ValueError(f"unknown variant: {req.variant}")

    if include_target:
        seg_path = req.target_audio_path or req.ref_audio_path
        if not seg_path:
            raise ValueError("the target block requires target_audio_path/ref_audio_path")
        segments.append({"type": "audio", "append_eos": True, "drop_last_frame": False,
                         "audio_path": seg_path})
    return segments


def build_example(tokenizer, req: ExampleRequest) -> dict[str, torch.Tensor]:
    """Builds input_ids/masks/input_values exactly like the official prepare_inputs.

    Requires register_codes() for ref_audio_path when variant=ref_edit_tata.
    Returns single-sample tensors (batch dim 1).
    """
    from breeze_infer.templates import _prepare_one

    segments = build_segments(req)
    out = _prepare_one(tokenizer, None, ConfigStub(16), segments)
    out["input_values"] = out["audio_tokens"]          # clearer alias
    return out


# ------------------------------------------------------------------ labels


def make_labels(example: dict, *, frame_policies: list[str]) -> torch.Tensor:
    """Builds the labels (1,S) for LoRA training.

    Grammar VALIDATED EMPIRICALLY in the smoke test (smoke_forward_b.py):
    - The backbone lm_head has ONLY 2052 classes (codes 0..2050 + EOS class 2051);
      text ids (262158) are INVALID as targets -> ALL text stays -100 and serves
      only as CONTEXT (the usual codec-LM/CSM pattern).
    - 'train' frames         -> 262144 (the internal expansion uses the real codes in all 16 slots);
    - 'backbone_only' frames -> -101   (codebook 0 trains the backbone; depth ignores 1..15);
    - <|audio_eos|> position -> 262145 (slot 0 becomes EOS class 2051, depth ignored).
    """
    input_ids = example["input_ids"][0]
    labels = torch.full_like(input_ids, IGNORE_IDX)

    frame_positions = (input_ids == AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]
    eos_positions = (input_ids == AUDIO_EOS_TOKEN_ID).nonzero(as_tuple=True)[0]

    counts = count_frames_per_block(example)
    assert len(counts) == len(frame_policies), (
        f"audio blocks={len(counts)} != policies={len(frame_policies)}"
    )
    assert sum(counts) == len(frame_positions), (
        f"frame misalignment: ids={len(frame_positions)} vs blocks={sum(counts)}"
    )
    idx = 0
    for cnt, pol in zip(counts, frame_policies):
        assert pol in ("train", "backbone_only"), pol
        value = AUDIO_TOKEN_ID if pol == "train" else DEPTH_IGNORE
        for _ in range(cnt):
            labels[frame_positions[idx]] = value
            idx += 1
    for p in eos_positions:
        labels[p] = AUDIO_EOS_TOKEN_ID
    return labels.unsqueeze(0)


def count_frames_per_block(example: dict) -> list[int]:
    """Counts consecutive '<|AUDIO|>' per block (blocks are separated by <|audio_eos|>)."""
    ids = example["input_ids"][0].tolist()
    blocks: list[int] = []
    cur = 0
    for t in ids:
        if t == AUDIO_TOKEN_ID:
            cur += 1
        elif t == AUDIO_EOS_TOKEN_ID:
            blocks.append(cur)
            cur = 0
    return blocks


TTS_INSTRUCTION_POLICIES = ["train"]
REF_EDIT_TATA_POLICIES = ["backbone_only", "train"]

POLICIES_BY_VARIANT = {
    "tts_instruction": TTS_INSTRUCTION_POLICIES,
    "ref_edit_tata": REF_EDIT_TATA_POLICIES,
    # ref_edit_auto uses the SAME template/policies as ref_edit (ref = another clip
    # of the SAME speaker; internally it maps to the ref_edit_tata template)
    "ref_edit_auto": REF_EDIT_TATA_POLICIES,
}


# ------------------------------------------------------------------ collate


def pad_left(t: torch.Tensor, pad_len: int, value) -> torch.Tensor:
    return F.pad(t, (pad_len, 0), value=value) if pad_len > 0 else t


def collate(items: list[dict], pad_token_id: int = 0):
    """Replicates the official _collate_inputs (left pad) + labels (-100 on padding).

    items: list of single-sample dicts coming from the Dataset:
      input_ids (L,), attention_mask (L,), text_ids_mask (L,), text_ids_len (nseg,),
      input_values (F,16), labels (L,)
    """
    max_len = max(it["input_ids"].shape[-1] for it in items)

    ids_l, att_l, mask_l, lab_l, tl_all = [], [], [], [], []
    for it in items:
        L = it["input_ids"].shape[-1]
        pad_len = max_len - L
        ids_l.append(pad_left(it["input_ids"].view(1, -1), pad_len, pad_token_id))
        att_l.append(pad_left(it["attention_mask"].view(1, -1), pad_len, 0))
        mask_l.append(pad_left(it["text_ids_mask"].view(1, -1), pad_len, False))
        lab_l.append(pad_left(it["labels"].view(1, -1), pad_len, IGNORE_IDX))
        tl_all.append(it["text_ids_len"])

    _ivs = []
    for it in items:
        iv = it["input_values"]
        if iv.dim() == 2:
            iv = iv.unsqueeze(0)
        _ivs.append(iv)
    input_values = torch.cat(_ivs, dim=1)  # (1, F_total, 16) without padding

    batch = {
        "input_ids": torch.cat(ids_l, dim=0),
        "attention_mask": torch.cat(att_l, dim=0),
        "text_ids_mask": torch.cat(mask_l, dim=0),
        "labels": torch.cat(lab_l, dim=0),
        "text_ids_len": torch.cat(tl_all, dim=0),
        "input_values": input_values.long(),
    }
    return batch


# ------------------------------------------------------------------ dataset item


def instruction_for(idx: int) -> str:
    return INSTRUCTION_POOL[idx % len(INSTRUCTION_POOL)]
