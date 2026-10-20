
#!/usr/bin/env python
"""
scripts/02_preprocess.py
------------------------
Clean the raw Jigsaw data, engineer features, run QA checks,
build multilabel-stratified splits, and save processed files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from wtc.config import CFG, LABELS, PROJECT_ROOT, TEXT_COL
from wtc.data import get_processed_dir, get_raw_dir, get_splits_dir
from wtc.preprocess import assign_dup_groups, clean_text, engineer_features
from wtc.utils import Timer, get_logger, set_seed

log = get_logger("02_preprocess", PROJECT_ROOT / CFG["paths"]["logs"])
set_seed(CFG["seed"])

RAW_DIR = get_raw_dir()
PROC_DIR = get_processed_dir()
SPLITS_DIR = get_splits_dir()
PROC_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

SEED = CFG["seed"]


# ------------------------------------------------------------------ #
#  Load raw                                                            #
# ------------------------------------------------------------------ #

def load_raw() -> pd.DataFrame:
    p = RAW_DIR / "train.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        log.info("Loaded raw train: %d rows, columns: %s", len(df), list(df.columns))
        return df
    raise FileNotFoundError(
        "data/raw/train.parquet not found. Run scripts/01_download_data.py first."
    )


# ------------------------------------------------------------------ #
#  QA checks                                                           #
# ------------------------------------------------------------------ #

def run_qa_checks(df: pd.DataFrame) -> dict:
    log.info("--- QA Checks ---")
    qa = {}

    # Nulls
    null_counts = df.isnull().sum().to_dict()
    qa["null_counts"] = {k: int(v) for k, v in null_counts.items()}
    log.info("Null counts: %s", qa["null_counts"])

    # Empty text after cleaning
    empty_mask = df["text_clean"].str.len() == 0
    qa["empty_text_count"] = int(empty_mask.sum())
    log.info("Empty text_clean rows: %d", qa["empty_text_count"])

    # Exact duplicates (on cleaned text)
    dup_mask = df.duplicated(subset=["text_clean"])
    qa["exact_duplicate_count"] = int(dup_mask.sum())
    log.info("Exact duplicates (text_clean): %d", qa["exact_duplicate_count"])

    # Duplicates with conflicting labels
    conflict_count = 0
    dup_groups = df.groupby("dup_group")
    for _, grp in dup_groups:
        if len(grp) > 1:
            label_vals = grp[LABELS].drop_duplicates()
            if len(label_vals) > 1:
                conflict_count += 1
    qa["conflicting_label_duplicates"] = conflict_count
    log.info("Duplicate groups with conflicting labels: %d", conflict_count)

    return qa


# ------------------------------------------------------------------ #
#  Split                                                               #
# ------------------------------------------------------------------ #

def build_splits(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multilabel-stratified 80/10/10 split.
    Groups with duplicates (dup_group) are kept together — we split on the
    unique dup_group IDs first, then expand back to row indices.
    """
    log.info("Building multilabel-stratified splits (seed=%d)...", SEED)
    split_cfg = CFG["splits"]
    val_frac = split_cfg["val"]
    test_frac = split_cfg["test"]

    # Build group-level DataFrame
    # For each dup_group, aggregate labels by max (if any row in group is toxic, group is toxic)
    group_df = (
        df.groupby("dup_group")[LABELS]
        .max()
        .reset_index()
    )
    X_groups = group_df["dup_group"].values.reshape(-1, 1)
    Y_groups = group_df[LABELS].values

    # First split: train vs temp (val+test)
    temp_frac = val_frac + test_frac
    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=temp_frac, random_state=SEED
    )
    train_g_idx, temp_g_idx = next(msss1.split(X_groups, Y_groups))

    # Second split: val vs test within temp
    val_frac_of_temp = val_frac / temp_frac
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=(1 - val_frac_of_temp), random_state=SEED
    )
    X_temp = X_groups[temp_g_idx]
    Y_temp = Y_groups[temp_g_idx]
    val_local_idx, test_local_idx = next(msss2.split(X_temp, Y_temp))

    train_groups = set(group_df.iloc[train_g_idx]["dup_group"])
    val_groups   = set(group_df.iloc[temp_g_idx[val_local_idx]]["dup_group"])
    test_groups  = set(group_df.iloc[temp_g_idx[test_local_idx]]["dup_group"])

    assert train_groups.isdisjoint(val_groups), "Train/val overlap in dup_groups!"
    assert train_groups.isdisjoint(test_groups), "Train/test overlap in dup_groups!"
    assert val_groups.isdisjoint(test_groups), "Val/test overlap in dup_groups!"

    train_idx = df[df["dup_group"].isin(train_groups)].index.tolist()
    val_idx   = df[df["dup_group"].isin(val_groups)].index.tolist()
    test_idx  = df[df["dup_group"].isin(test_groups)].index.tolist()

    log.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_idx), len(val_idx), len(test_idx),
    )
    return (
        np.array(train_idx),
        np.array(val_idx),
        np.array(test_idx),
    )


# ------------------------------------------------------------------ #
#  Save splits                                                         #
# ------------------------------------------------------------------ #

def save_splits(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        with open(SPLITS_DIR / f"{name}_idx.json", "w") as fh:
            # Store positional indices (iloc positions in the processed df)
            # Map from original df.index to iloc position
            pos_map = {orig: pos for pos, orig in enumerate(df.index)}
            iloc_positions = [pos_map[i] for i in idx]
            json.dump(iloc_positions, fh)

    # Leakage check
    train_groups = set(df.loc[train_idx, "dup_group"])
    val_groups   = set(df.loc[val_idx,   "dup_group"])
    test_groups  = set(df.loc[test_idx,  "dup_group"])
    assert train_groups.isdisjoint(val_groups),  "LEAKAGE: train/val"
    assert train_groups.isdisjoint(test_groups), "LEAKAGE: train/test"
    assert val_groups.isdisjoint(test_groups),   "LEAKAGE: val/test"
    log.info("Leakage check PASSED — no dup_group crosses splits.")

    # Manifest
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "split_ratios": CFG["splits"],
        "splits": {},
    }
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        sdf = df.loc[idx]
        manifest["splits"][name] = {
            "n_rows": int(len(idx)),
            "label_positive_counts": {
                lbl: int(sdf[lbl].sum()) for lbl in LABELS
            },
            "label_positive_rates": {
                lbl: float(round(sdf[lbl].mean(), 6)) for lbl in LABELS
            },
        }

    with open(SPLITS_DIR / "split_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("split_manifest.json written.")
    return manifest


# ------------------------------------------------------------------ #
#  Data dictionary                                                     #
# ------------------------------------------------------------------ #

DATA_DICT_MD = """\
# Data Dictionary — Welcome to Civilization

## Source
Jigsaw Toxic Comment Classification Challenge  
(Originally: Kaggle / google/jigsaw_toxicity_pred on HuggingFace)

## File: `data/processed/processed.parquet` / `processed.csv`

| Column | Type | Description |
|---|---|---|
| id | str | Original comment ID |
| comment_text | str | Raw, untouched original comment text |
| toxic | int8 | 1 if comment is toxic (any toxicity), else 0 |
| severe_toxic | int8 | 1 if severely toxic, else 0 |
| obscene | int8 | 1 if obscene content, else 0 |
| threat | int8 | 1 if contains a threat, else 0 |
| insult | int8 | 1 if insulting, else 0 |
| identity_hate | int8 | 1 if identity-based hate speech, else 0 |
| text_clean | str | Cleaned text: NFKC normalized, control chars removed, whitespace collapsed, HTML entities decoded. Case and punctuation preserved. |
| has_url | bool | True if text_clean contains an HTTP/HTTPS URL or www. link |
| has_ip | bool | True if text_clean contains an IPv4 address pattern |
| has_mention | bool | True if text_clean contains an @username mention |
| n_chars | int | Character count of text_clean |
| n_words | int | Word count (split on whitespace) of text_clean |
| n_labels | int | Sum of the six label columns (0–6) |
| is_clean | int | 1 if n_labels == 0 (no toxic label), else 0 |
| upper_ratio | float | Fraction of alphabetic characters that are uppercase |
| non_ascii_ratio | float | Fraction of characters with ord > 127 |
| punct_ratio | float | Fraction of characters in Unicode punctuation categories |
| max_repeated_char_run | int | Length of the longest run of consecutive identical characters |
| dup_group | int64 | ID of the duplicate group (rows with identical lowercased text_clean share a group) |

## Notes
- Label columns are binary int8.  
- `comment_text` is never modified.  
- Model-specific normalization (lowercasing, special tokens, etc.) happens inside each model adapter, NOT here.  
- Duplicates are kept but tracked via `dup_group`. Split construction ensures no dup_group crosses splits.  
- Rows with empty `text_clean` are dropped.  

## File: `data/splits/`

| File | Description |
|---|---|
| `train_idx.json` | Positional (iloc) indices into processed.parquet for the train split |
| `val_idx.json` | Positional (iloc) indices for the validation split |
| `test_idx.json` | Positional (iloc) indices for the test split |
| `split_manifest.json` | Row counts, label positive counts and rates per split |

## File: `data/raw/official_test.parquet`
Official Kaggle test set rows with valid labels (label != -1).  
These are kept for optional external evaluation but are NOT used in the train/val/test splits above.
"""


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    log.info("=== 02_preprocess.py START ===")
    with Timer() as t:

        # Load
        df = load_raw()

        # Engineer features
        log.info("Engineering features...")
        df = engineer_features(df, text_col=TEXT_COL, labels=LABELS)

        # Assign dup groups
        log.info("Assigning duplicate groups...")
        df["dup_group"] = assign_dup_groups(df, text_col="text_clean")
        log.info("Unique dup groups: %d", df["dup_group"].nunique())

        # QA checks
        qa = run_qa_checks(df)

        # Drop empty-text rows (documented decision)
        empty_mask = df["text_clean"].str.len() == 0
        if empty_mask.sum() > 0:
            log.warning(
                "Dropping %d rows with empty text_clean (decision: keep data quality)",
                empty_mask.sum(),
            )
            df = df[~empty_mask].copy()

        # Reset index after drop
        df = df.reset_index(drop=True)

        # Save processed
        df.to_parquet(PROC_DIR / "processed.parquet", index=False)
        df.to_csv(PROC_DIR / "processed.csv", index=False)
        log.info("Processed data saved: %d rows", len(df))

        # Save data dictionary
        with open(PROC_DIR / "DATA_DICTIONARY.md", "w", encoding="utf-8") as fh:
            fh.write(DATA_DICT_MD)

        # Build splits
        train_idx, val_idx, test_idx = build_splits(df)

        # Save splits + manifest
        manifest = save_splits(df, train_idx, val_idx, test_idx)

        # QA summary
        qa_path = PROC_DIR / "qa_report.json"
        with open(qa_path, "w") as fh:
            json.dump({
                "qa_checks": qa,
                "final_rows": int(len(df)),
                "split_manifest": manifest,
            }, fh, indent=2)

        log.info("=== 02_preprocess.py DONE in %s ===", t)
        log.info(
            "Split summary — train: %d | val: %d | test: %d",
            len(train_idx), len(val_idx), len(test_idx),
        )


if __name__ == "__main__":
    main()
