"""merge_lora.py — merge a LoRA into the base weights and save the result as an HF folder.

Runs on the CPU on purpose: the GPU (an R9700 in our setup) is busy with
training, and a second 3B model next to it would compete for both memory and
compute. Without a GPU it takes a few minutes instead of seconds — worth it.

The target folder additionally gets copies of audio_tokenizer/ and the tokenizer
files, because convert_hf_to_gguf.py (companion C++ engine) expects everything in
the same directory.

  python scripts/merge_lora.py <checkpoint-dir> <target-dir>
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "de_lora" / "core"))
import common_breeze as CB                                    # noqa: E402

sys.path.insert(0, str(CB.REPO))
from models.breeze import BreezeForConditionalGeneration      # noqa: E402
from peft import PeftModel                                    # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ckpt, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if not (ckpt / "adapter_model.safetensors").is_file():
        sys.exit(f"no adapter in {ckpt}")
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[merge] loading the base model from {CB.CKPT} (CPU, float32 for clean arithmetic)")
    # float32 while merging: the LoRA contribution is small compared to the base
    # weights; in bfloat16 part of it falls below the resolution.
    base = BreezeForConditionalGeneration.from_pretrained(
        CB.CKPT, dtype=torch.float32, attn_implementation="eager")
    print(f"[merge] applying adapter: {ckpt}")
    pm = PeftModel.from_pretrained(base, str(ckpt))
    print("[merge] merging ...")
    merged = pm.merge_and_unload()
    merged = merged.to(torch.float16)      # the GGUF converter wants f16
    print(f"[merge] writing to {dst}")
    merged.save_pretrained(dst, safe_serialization=True, max_shard_size="4GB")

    # The converter reads config.json, the shards, tokenizer.json and
    # audio_tokenizer/ from one directory. save_pretrained only creates the
    # first two.
    for name in ("tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "generation_config.json"):
        src = CB.CKPT / name
        if src.is_file():
            shutil.copy2(src, dst / name)
            print(f"[merge] copied: {name}")
    at_src, at_dst = CB.CKPT / "audio_tokenizer", dst / "audio_tokenizer"
    if at_src.is_dir() and not at_dst.exists():
        shutil.copytree(at_src, at_dst)
        print(f"[merge] copied: audio_tokenizer/ ({sum(f.stat().st_size for f in at_dst.rglob('*') if f.is_file())/1e6:.0f} MB)")
    print("[merge] done")


if __name__ == "__main__":
    main()
