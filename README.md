
# Welcome to Civilization

**Multi-label toxicity classification on the Jigsaw Toxic Comment dataset.**

Labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

Later stages will train four models (TF-IDF+LR, BiGRU+attention, DistilRoBERTa, RoBERTa)
through one shared pipeline built on top of this data foundation.

---

## Project Structure

```
welcome-to-civilization/
  configs/config.yaml          ← seed, paths, label names, split ratios
  data/raw/                    ← untouched downloaded data
  data/processed/              ← cleaned data (parquet + csv)
  data/splits/                 ← train/val/test index files + split_manifest.json
  src/wtc/                     ← Python package (config, utils, data, preprocess, eda)
  scripts/
    01_download_data.py        ← Download & save raw data
    02_preprocess.py           ← Clean, feature-engineer, split
    03_eda.py                  ← EDA analyses a–l
    04_build_report.py         ← Generate EDA conclusions + report
  notebooks/01_eda.ipynb       ← Thin EDA wrapper notebook
  reports/eda/                 ← Figures, tables, findings JSON, reports
  tests/test_data.py           ← pytest data-integrity tests
  requirements.txt
  Makefile
```

---

## Local Setup

### Prerequisites
- Python 3.10+
- `make` (or use `run_all.sh` on Windows without make)

### 1. Clone and navigate
```bash
git clone <repo-url>
cd welcome-to-civilization
```

### 2. Set up the environment
```bash
make setup
# OR (Windows PowerShell without make):
python -m venv .venv
.venv\Scripts\pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
```

### 3. Obtain the data

**Option A — Kaggle CLI (recommended)**:
```bash
# Set up Kaggle credentials: place kaggle.json in ~/.kaggle/
make download
```

**Option B — Manual download**:
1. Go to https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
2. Accept the rules and download `train.csv`, `test.csv`, `test_labels.csv`
3. Place all three in `data/raw/kaggle_manual/`
4. Run: `make download`

### 4. Run the full pipeline
```bash
make all
# OR on Windows PowerShell:
.\run_all.ps1
# OR on Bash without make:
bash run_all.sh
# OR step by step:
make download
make preprocess
make eda
make report
make test
```

---

## Google Colab Setup

```python
# 1. Mount Drive and navigate to project root
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/MyDrive/welcome-to-civilization')

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Run the pipeline
!python scripts/01_download_data.py
!python scripts/02_preprocess.py
!python scripts/03_eda.py
!python scripts/04_build_report.py

# 4. Test
!pytest tests/ -v
```

> **Note**: All paths are config-relative (no absolute paths). The config loader
> auto-detects the project root from the location of `configs/config.yaml`.

---

## Run Order

| Step | Command | Output |
|---|---|---|
| 1. Setup | `make setup` | `.venv/` with pinned deps |
| 2. Download | `make download` | `data/raw/*.parquet`, `raw_manifest.json` |
| 3. Preprocess | `make preprocess` | `data/processed/processed.parquet`, `data/splits/` |
| 4. EDA | `make eda` | `reports/eda/figures/`, `tables/`, `eda_findings.json` |
| 5. Report | `make report` | `eda_conclusions.md`, `eda_report.md` |
| 6. Test | `make test` | pytest results |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| No lowercasing/punct stripping in saved data | Toxicity signals in casing/obfuscation |
| Keep duplicates, add `dup_group` | Avoids data loss; leakage prevented by split constraint |
| iterstrat MultilabelStratifiedShuffleSplit | Better label balance than random stratification |
| Keep official test label=-1 rows (flagged) | Required by spec; useful for later analysis |
| Global seed=42 | Reproducibility across all stages |
| All paths from `configs/config.yaml` | Works locally and in Colab |

---

## Configuration

Edit `configs/config.yaml` to change:
- `seed`: global random seed
- `splits`: train/val/test ratios
- `labels`: label column names
- `paths`: all data and output paths
- `tokenizer_name`: HuggingFace tokenizer for EDA step (e)
- `max_len_options`: token length options to analyze

---

## Tests

```bash
make test
# OR:
.venv/Scripts/pytest tests/ -v
```

12 tests covering:
- No index overlap between train/val, train/test, and val/test splits (3 tests)
- All processed rows covered exactly once across splits
- No duplicate-group leakage across splits
- Label columns strictly binary (0/1)
- Split sizes within ±2% tolerance of 80/10/10 configuration
- Row counts strictly match split manifest
- No empty `text_clean` rows
- All 21 required feature and label columns present
- `n_labels` consistent with sum of label columns
- `is_clean` consistent with inverse of positive label sum
