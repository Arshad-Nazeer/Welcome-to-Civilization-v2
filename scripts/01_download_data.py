
#!/usr/bin/env python
"""
scripts/01_download_data.py
---------------------------
Download the Jigsaw Toxicity dataset and save raw parquet files.
Tries strategies in order:
  (a) Local CSVs in data/raw/kaggle_manual/ (train.csv, test.csv, test_labels.csv)
  (b) Kaggle CLI download
  (c) HuggingFace load_dataset("google/jigsaw_toxicity_pred")
  (d) STOP and print manual instructions

For manual CSV placement, see README.md section "Data Download".
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from wtc.config import CFG, PROJECT_ROOT
from wtc.utils import get_logger, set_seed, sha256_file, Timer

log = get_logger("01_download", PROJECT_ROOT / CFG["paths"]["logs"])
set_seed(CFG["seed"])

RAW_DIR = PROJECT_ROOT / CFG["paths"]["data_raw"]
KAGGLE_DIR = PROJECT_ROOT / CFG["paths"]["data_raw_kaggle"]
HF_DATASET = CFG["hf_dataset_name"]
KAGGLE_COMP = CFG["kaggle_competition"]

RAW_DIR.mkdir(parents=True, exist_ok=True)
KAGGLE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def get_lib_versions() -> dict:
    import datasets
    import pandas
    import transformers
    return {
        "datasets": datasets.__version__,
        "pandas": pandas.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
    }


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)
    log.info("Saved %s  (%d rows, %d cols)", path.name, len(df), df.shape[1])


def write_manifest(source: str, train_df: pd.DataFrame,
                   test_df: pd.DataFrame | None,
                   official_test_df: pd.DataFrame | None) -> None:
    files = {}
    for name, path in [
        ("train.parquet", RAW_DIR / "train.parquet"),
        ("test.parquet",  RAW_DIR / "test.parquet"),
        ("official_test.parquet", RAW_DIR / "official_test.parquet"),
    ]:
        if path.exists():
            files[name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }

    manifest = {
        "source": source,
        "download_date_utc": datetime.now(timezone.utc).isoformat(),
        "library_versions": get_lib_versions(),
        "files": files,
        "train_rows": int(len(train_df)) if train_df is not None else None,
        "train_columns": list(train_df.columns) if train_df is not None else None,
        "test_rows": int(len(test_df)) if test_df is not None else None,
        "official_test_rows": int(len(official_test_df)) if official_test_df is not None else None,
        "notes": {
            "test_labels_minus1": (
                "In the official test labels, -1 means the row was not used "
                "for scoring. These rows are kept in raw data but flagged."
            )
        },
    }
    manifest_path = RAW_DIR / "raw_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("Manifest written → %s", manifest_path)


# ------------------------------------------------------------------ #
#  Strategy (a): HuggingFace direct                                    #
# ------------------------------------------------------------------ #

def try_hf_direct() -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    """Returns (train_df, test_df, source_tag) or (None, None, '')."""
    try:
        from datasets import load_dataset
        log.info("Strategy (a): load_dataset('%s')", HF_DATASET)
        ds = load_dataset(HF_DATASET)
        train_df = ds["train"].to_pandas()
        test_df  = ds["test"].to_pandas() if "test" in ds else None
        log.info("Strategy (a) SUCCESS — train=%d rows", len(train_df))
        return train_df, test_df, "huggingface_direct"
    except Exception as exc:
        log.warning("Strategy (a) FAILED: %s", exc)
        return None, None, ""


# ------------------------------------------------------------------ #
#  Strategy (b): HuggingFace with local data_dir                      #
# ------------------------------------------------------------------ #

def try_hf_manual() -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    """Use CSV files in data/raw/kaggle_manual/ via load_dataset(..., data_dir=...)."""
    train_csv  = KAGGLE_DIR / "train.csv"
    test_csv   = KAGGLE_DIR / "test.csv"
    labels_csv = KAGGLE_DIR / "test_labels.csv"

    missing = [f for f in [train_csv, test_csv] if not f.exists()]
    if missing:
        log.warning(
            "Strategy (b) skipped — missing files: %s",
            [str(f) for f in missing],
        )
        return None, None, ""

    try:
        from datasets import load_dataset
        log.info("Strategy (b): load_dataset with data_dir='%s'", KAGGLE_DIR)
        ds = load_dataset(HF_DATASET, data_dir=str(KAGGLE_DIR),
                          trust_remote_code=True)
        train_df = ds["train"].to_pandas()
        test_df  = ds["test"].to_pandas() if "test" in ds else None
        if "id" in train_df.columns and (test_df is None or len(test_df) >= 150000):
            log.info("Strategy (b) HF+data_dir SUCCESS")
            return train_df, test_df, "huggingface_manual_dir"
        log.warning(
            "Strategy (b) HF loader stripped 'id' or unscored test rows. Falling back to raw CSV to preserve required fields."
        )
    except Exception as exc:
        log.warning(
            "Strategy (b) HF loader failed (%s). Falling back to raw CSV.", exc
        )

    # Plain CSV fallback (still counts as strategy b)
    try:
        log.info("Strategy (b-csv): reading CSVs directly")
        train_df = pd.read_csv(train_csv)
        test_df  = pd.read_csv(test_csv)
        if labels_csv.exists():
            labels_df = pd.read_csv(labels_csv)
            test_df = test_df.merge(labels_df, on="id", how="left")
        log.info("Strategy (b-csv) SUCCESS")
        return train_df, test_df, "kaggle_manual_csv"
    except Exception as exc2:
        log.warning("Strategy (b-csv) FAILED: %s", exc2)
        return None, None, ""


# ------------------------------------------------------------------ #
#  Strategy (c): Kaggle CLI                                            #
# ------------------------------------------------------------------ #

def try_kaggle_cli() -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    """Download via Kaggle API CLI, then read CSVs."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        log.warning("Strategy (c) skipped — kaggle package not installed")
        return None, None, ""

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        log.warning("Strategy (c) skipped — %s not found", kaggle_json)
        return None, None, ""

    try:
        log.info("Strategy (c): Kaggle CLI download")
        cmd = [
            sys.executable, "-m", "kaggle", "competitions", "download",
            "-c", KAGGLE_COMP, "-p", str(RAW_DIR), "--unzip",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            log.warning("Kaggle CLI failed: %s", result.stderr)
            return None, None, ""

        # Move files to kaggle_manual if not already there
        for fname in ["train.csv", "test.csv", "test_labels.csv"]:
            src = RAW_DIR / fname
            if src.exists():
                dst = KAGGLE_DIR / fname
                src.rename(dst)

        train_df, test_df, _ = try_hf_manual()
        if train_df is not None:
            log.info("Strategy (c) SUCCESS")
            return train_df, test_df, "kaggle_cli"

    except Exception as exc:
        log.warning("Strategy (c) FAILED: %s", exc)

    return None, None, ""


# ------------------------------------------------------------------ #
#  Strategy (d): STOP                                                  #
# ------------------------------------------------------------------ #

def stop_and_instruct() -> None:
    msg = """
╔══════════════════════════════════════════════════════════════════════╗
║  MANUAL STEP REQUIRED — Cannot auto-download the Jigsaw dataset     ║
╠══════════════════════════════════════════════════════════════════════╣
║  1. Go to: https://www.kaggle.com/c/jigsaw-toxic-comment-           ║
║            classification-challenge/data                            ║
║  2. Accept the competition rules and download these three files:    ║
║       • train.csv.zip                                               ║
║       • test.csv.zip                                                ║
║       • test_labels.csv.zip                                         ║
║  3. Unzip them and place the CSV files here:                        ║
║       data/raw/kaggle_manual/train.csv                              ║
║       data/raw/kaggle_manual/test.csv                               ║
║       data/raw/kaggle_manual/test_labels.csv                        ║
║  4. Re-run:  python scripts/01_download_data.py                     ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    log.error("All download strategies failed.")
    print(msg)
    sys.exit(1)


# ------------------------------------------------------------------ #
#  Process and save                                                    #
# ------------------------------------------------------------------ #

LABELS = CFG["labels"]


def process_and_save(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    source: str,
) -> None:
    """Validate columns, flag test rows, save parquet + manifest."""
    # --- Train ---
    log.info("Train columns: %s", list(train_df.columns))
    missing_labels = [l for l in LABELS if l not in train_df.columns]
    if missing_labels:
        log.warning("Missing label columns in train: %s", missing_labels)

    save_parquet(train_df, RAW_DIR / "train.parquet")

    # --- Test (official, may have labels) ---
    official_test_df: pd.DataFrame | None = None
    if test_df is not None:
        log.info("Test columns: %s", list(test_df.columns))
        # Flag rows where labels are -1 (not used in scoring)
        if all(l in test_df.columns for l in LABELS):
            test_df["label_available"] = (test_df[LABELS] != -1).all(axis=1)
            official_test_df = test_df[test_df["label_available"]].copy()
            log.info(
                "Official test: %d total rows, %d with valid labels (label != -1)",
                len(test_df), len(official_test_df),
            )
        save_parquet(test_df, RAW_DIR / "test.parquet")
        if official_test_df is not None and len(official_test_df) > 0:
            save_parquet(official_test_df, RAW_DIR / "official_test.parquet")

    write_manifest(source, train_df, test_df, official_test_df)


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    log.info("=== 01_download_data.py START ===")
    with Timer() as t:
        # Order: local CSVs first, then Kaggle CLI, then HuggingFace
        for strategy_fn in [try_hf_manual, try_kaggle_cli, try_hf_direct]:
            train_df, test_df, source = strategy_fn()
            if train_df is not None:
                process_and_save(train_df, test_df, source)
                log.info("=== Download complete in %s ===", t)
                return
        stop_and_instruct()


if __name__ == "__main__":
    main()
