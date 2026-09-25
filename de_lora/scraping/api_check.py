"""Health check of the X.ai transcription API (api.x.ai/v1/stt).

Sends a 0.5 s clip (the first available chunk) and prints the result.
Usage: python api_check.py  ->  HTTP 200 + text = API working; 403/429 = blocked/quota.
Note: the paths below use Windows separators (as in the template).
"""
import glob
import sys
from pathlib import Path

import requests
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
import paths  # noqa: E402

ROOT = str(paths.ARTIFACTS)
key = [l.split("=", 1)[1].strip() for l in open(ROOT + r"\.env") if l.startswith("XAI_TRANSCRIBE_KEY")][0]
f = sorted(glob.glob(ROOT + r"\datasets\podcast\wavs\*.wav"))[0]
info = sf.info(f)
wav, sr = sf.read(f, frames=info.samplerate // 2, dtype="float32")
sf.write(ROOT + r"\dataScrapping\work\api_check_clip.wav", wav, sr, subtype="PCM_16")

r = requests.post("https://api.x.ai/v1/stt",
                  headers={"Authorization": f"Bearer {key}"},
                  files={"file": ("clip.wav", open(ROOT + r"\dataScrapping\work\api_check_clip.wav", "rb"), "audio/wav")},
                  data={"format": "true", "language": "pt"}, timeout=60)
print("HTTP", r.status_code)
print(r.text[:300])
