"""paths.py — single, environment-driven resolution of all project paths.

Rule: the CODE lives in this repository (the engine fork + `de_lora/`), while
the ARTIFACTS (datasets/, training/, models/) live OUTSIDE the git tree.

Set the artifact root in one of two ways:
  - environment variable  PTBR_ARTIFACTS=/path/to/artifacts
  - a .env file at the repository root (same KEY=VALUE format; see .env.example)
Variables already set in the environment take precedence over the .env file.

Accepted variables (all optional; the defaults are repository-local):
  PTBR_ARTIFACTS       root of datasets/, training/, models/ (name inherited from the pt-BR template)
  BREEZE_TTS_REPO      where the engine lives (default: the root of this fork)
  BREEZE_CKPT          base checkpoint folder (default: <ARTIFACTS>/models/Breeze-TTS-2)
  BREEZE_DATASETS_DIR  datasets folder (default: <ARTIFACTS>/datasets)
  BREEZE_TRAINING_DIR  training folder (default: <ARTIFACTS>/training)
  BREEZE_DATASET_DIR   single legacy corpus (default: <datasets>/TTS-Portuguese-Corpus)
  BREEZE_PY            Python interpreter for subprocesses (default: sys.executable)

German pipeline (added in this fork): raw source corpora and tool outputs.
Raw downloads follow the "_raw" convention of data/build_corpus.py:
  CML_DE_ROOT          extracted CML-TTS German (default: <datasets>/_raw/cml-tts-german/cml_tts_dataset_german_v0.1)
  HUI_GLOB             HUI-Audio-Corpus-German parquet files (default: <datasets>/_raw/opendata-iisys-hui/data/train-*.parquet)
  HIFITTS2_ROOT        HiFiTTS-2 subset (default: <datasets>/_raw/hifitts2)
  THORSTEN_DIR         Thorsten-Voice parquet download (default: <datasets>/_raw/thorsten-tv44)
  CV_DIR               Common Voice 27.0 German (default: <datasets>/_raw/common-voice-de)
  BREEZE_WORDS_DIR     rare-word lists, capitalization table, coverage reports (default: <ARTIFACTS>)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# de_lora/core/paths.py -> parents[2] = repository root (engine + de_lora)
REPO = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from <repo>/.env without depending on python-dotenv."""
    env = REPO / ".env"
    if not env.is_file():
        return
    for raw in env.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()


def _p(env: str, default: Path) -> Path:
    val = os.environ.get(env)
    return Path(val).expanduser().resolve() if val else default


ARTIFACTS = _p("PTBR_ARTIFACTS", REPO / "artifacts")

# The engine sits at the root of this fork by default; it can point at another clone.
BREEZE_REPO = _p("BREEZE_TTS_REPO", REPO)

CKPT = _p("BREEZE_CKPT", ARTIFACTS / "models" / "Breeze-TTS-2")
DATASETS_ROOT = _p("BREEZE_DATASETS_DIR", ARTIFACTS / "datasets")
TRAINING = _p("BREEZE_TRAINING_DIR", ARTIFACTS / "training")
TOKENS_DIR = TRAINING / "tokens"
WAVS24_DIR = TRAINING / "wavs24"
GOLD_DIR = TRAINING / "gold_samples"

DATASET = _p("BREEZE_DATASET_DIR", DATASETS_ROOT / "TTS-Portuguese-Corpus")
TEXTS_CSV = DATASET / "texts.csv"
CORPORA_JSON = DATASETS_ROOT / "corpora.json"

# Outputs of the podcast scraping pipeline (the code lives in de_lora/scraping).
SCRAPING = ARTIFACTS / "dataScrapping"
SCRAPING_WORK = SCRAPING / "work"

# Interpreter for subprocesses (auto_train); default = the current Python.
BREEZE_PY = os.environ.get("BREEZE_PY") or sys.executable

# ------------------------------------------------ German pipeline (this fork)
# Raw source corpora, read by de_lora/data/build_corpus_*.py and tools/.
RAW_DIR = DATASETS_ROOT / "_raw"
CML_DE_ROOT = _p("CML_DE_ROOT", RAW_DIR / "cml-tts-german" / "cml_tts_dataset_german_v0.1")
HUI_GLOB = os.environ.get("HUI_GLOB") or str(RAW_DIR / "opendata-iisys-hui" / "data" / "train-*.parquet")
HIFITTS2_ROOT = _p("HIFITTS2_ROOT", RAW_DIR / "hifitts2")
THORSTEN_DIR = _p("THORSTEN_DIR", RAW_DIR / "thorsten-tv44")
CV_DIR = _p("CV_DIR", RAW_DIR / "common-voice-de")
# Word lists and reports written by tools/ (seltene_woerter.json, grossschreibung.json, ...).
WORDS_DIR = _p("BREEZE_WORDS_DIR", ARTIFACTS)
