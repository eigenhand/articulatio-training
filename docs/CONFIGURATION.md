# Configuration and artifacts

The code lives in the repository; the **artifacts** (`datasets/`, `training/`,
`models/`) stay **outside git**. All path resolution happens in
`de_lora/core/paths.py`, driven by environment variables or by a `.env` file at
the repository root (variables already set in the environment win). The shell
scripts (`scripts/*.sh`, `recipes/*.sh`) ask `paths.py` for their locations, so
Python and shell always agree.

## Minimal configuration

Copy `.env.example` to `.env` and edit:

```ini
PTBR_ARTIFACTS=/path/to/artifacts
```

`PTBR_ARTIFACTS` is the root that contains `datasets/`, `training/` and
`models/` (the name is inherited from the pt-BR template).

## Variables read by the Python code (`paths.py`, also from `.env`)

| Variable | Default | Description |
|---|---|---|
| `PTBR_ARTIFACTS` | `<repo>/artifacts` | root of the artifacts |
| `BREEZE_TTS_REPO` | repository root | where the engine lives |
| `BREEZE_CKPT` | `<ARTIFACTS>/models/Breeze-TTS-2` | base checkpoint |
| `BREEZE_DATASETS_DIR` | `<ARTIFACTS>/datasets` | processed corpora and `corpora.json` |
| `BREEZE_TRAINING_DIR` | `<ARTIFACTS>/training` | code store, splits, manifests, runs (the recipes set it per run) |
| `BREEZE_DATASET_DIR` | `<datasets>/TTS-Portuguese-Corpus` | single legacy pt-BR corpus |
| `BREEZE_PY` | `sys.executable` | interpreter used by `auto_train.py` (and by `scripts/run.sh`) |
| `CML_DE_ROOT` | `<datasets>/_raw/cml-tts-german/cml_tts_dataset_german_v0.1` | extracted CML-TTS German |
| `HUI_GLOB` | `<datasets>/_raw/opendata-iisys-hui/data/train-*.parquet` | HUI-Audio-Corpus-German parquet files |
| `HIFITTS2_ROOT` | `<datasets>/_raw/hifitts2` | HiFiTTS-2 subset (`selected_chapters.jsonl`, `chapters/`, `texte.tsv`) |
| `THORSTEN_DIR` | `<datasets>/_raw/thorsten-tv44` | Thorsten-Voice parquet (`TV-2021.02-Neutral/`, `TV-2022.10-Neutral/`) |
| `CV_DIR` | `<datasets>/_raw/common-voice-de` | Common Voice 27.0 German (`tsv/`, `teil.0`…`teil.3`) |
| `BREEZE_WORDS_DIR` | `<ARTIFACTS>` | rare-word lists, capitalization table, coverage reports |

## Variables read by the shell scripts (environment only)

| Variable | Default | Used by |
|---|---|---|
| `BREEZE_ROOT` | repository root | all scripts |
| `LOG_DIR` | `<ARTIFACTS>/logs` | recipes (they coordinate through log files) |
| `CPU_PIN` | empty (no pinning) | `taskset` core list for side jobs, e.g. `10-13` |
| `MIOPEN_USER_DB_PATH` | `<repo>/.miopen` | `scripts/run.sh` |
| `ADAPTER_BYTES` | `593168056` | `pipeline_v5.sh` waits for an adapter of this size (r=64, all targets) |
| `BREEZE_CPP_DIR` | — (required for GGUF) | `scripts/deploy_checkpoint.sh` |
| `RUN_DIR`, `TAG`, `GGUF_DIR`, `ACTIVE_NAME` | see script header | `scripts/deploy_checkpoint.sh` |
| `RESTART_CMD`, `HEALTH_URL` | empty | optional server restart after `--activate` |
| `TTS_URL`, `TEST_VOICES`, `LISTEN_DIR` | empty, `thorsten`, `<ARTIFACTS>/listening` | optional listening test in `pipeline_v5.sh` |

## Artifact tree

```
<PTBR_ARTIFACTS>/
├─ models/Breeze-TTS-2/          # base checkpoint (official download)
├─ datasets/
│  ├─ corpora.json               # registry of corpora (list; "enabled" per corpus)
│  ├─ cml_de/ hui_de/ hifitts2_en/ thorsten_rare_de/ cml_rare_de/ cv_rare_de/ cv_breit_de/
│  │                             # wavs/ + texts.csv + speakers.jsonl (+ selection/quality files)
│  ├─ rauschvorlagen/            # degraded reference WAVs + manifest.jsonl (run 5)
│  └─ _raw/                      # raw downloads (see the variables above)
├─ training[-v2|-v3]/            # one directory per data generation, see the recipes
│  ├─ codes/codes.i16            # consolidated codec codes (int16, frames x 16)
│  ├─ codes/index.jsonl          # idx -> (offset, frames)
│  ├─ codes_rauschen/            # codes of the degraded references (run 5)
│  ├─ gold_samples/              # parity-check examples
│  ├─ dataset_meta.jsonl         # manifest per item (idx, text, dur, corpus, speaker)
│  ├─ manifest.csv  summary.json
│  ├─ splits_{train,val,test}.txt
│  └─ runs/<run>/
│     ├─ config.json  log.csv  trainable_modules.txt
│     ├─ checkpoints/<tag>/      # LoRA adapter (adapter_config.json + .safetensors)
│     ├─ samples/                # validation WAVs + wer.json
│     └─ tb/                     # TensorBoard
├─ seltene_woerter.json/.tsv, grossschreibung.json, rare_de_*.json   # BREEZE_WORDS_DIR
├─ logs/                         # recipe logs (LOG_DIR)
└─ gguf/                         # merged + quantized models (GGUF_DIR)
```

## Registering corpora

`datasets/corpora.json` is a list of corpora; only entries with
`"enabled": true` (default) are read by `prepare_dataset.py`:

```json
[
  {"name": "hui_de",      "root": "hui_de",      "csv": "texts.csv", "speakers": "speakers.jsonl", "enabled": true},
  {"name": "hifitts2_en", "root": "hifitts2_en", "csv": "texts.csv", "speakers": "speakers.jsonl", "enabled": true},
  {"name": "cml_de",      "root": "cml_de",      "csv": "texts.csv", "speakers": "speakers.jsonl", "enabled": false}
]
```

The builders register their corpus themselves; the rare-word and Common Voice
tools register theirs **disabled**, and the recipes switch the set of enabled
corpora per run. **Corpus names must end in `_de` or `_en`**: the language of the
number expansion and of the instructions is chosen from that suffix.

Without `corpora.json` the pipeline falls back to the template's legacy pair
(`tata` + `podcast`) driven by `BREEZE_DATASET_DIR`.
