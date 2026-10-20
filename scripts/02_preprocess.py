
#!/usr/bin/env python
"""
scripts/02_preprocess.py
------------------------
Clean the raw Jigsaw data, engineer features, run QA checks,
build multilabel-stratified splits on dup_group_v2 (near-duplicate safe),
and save processed files.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from wtc.config import CFG, LABELS, PROJECT_ROOT, TEXT_COL
from wtc.data import get_processed_dir, get_raw_dir, get_splits_dir
from wtc.preprocess import (
    assign_dup_groups,
    assign_near_dup_groups,
    clean_text,
    engineer_features,
)
from wtc.utils import Timer, get_logger, set_seed

log = get_logger("02_preprocess", PROJECT_ROOT / CFG["paths"]["logs"])
set_seed(CFG["seed"])

RAW_DIR = get_raw_dir()
PROC_DIR = get_processed_dir()
SPLITS_DIR = get_splits_dir()
PROC_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

SEED = CFG["seed"]
NEAR_DUP_CFG = CFG.get("near_dup", {})


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_lib_versions() -> dict:
    import datasketch
    import datasets
    import sklearn
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "datasketch": datasketch.__version__,
        "scikit_learn": sklearn.__version__,
    }


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

    null_counts = df.isnull().sum().to_dict()
    qa["null_counts"] = {k: int(v) for k, v in null_counts.items()}
    log.info("Null counts: %s", qa["null_counts"])

    empty_mask = df["text_clean"].str.len() == 0
    qa["empty_text_count"] = int(empty_mask.sum())
    log.info("Empty text_clean rows: %d", qa["empty_text_count"])

    dup_mask = df.duplicated(subset=["text_clean"])
    qa["exact_duplicate_count"] = int(dup_mask.sum())
    log.info("Exact duplicates (text_clean): %d", qa["exact_duplicate_count"])

    conflict_count = 0
    for _, grp in df.groupby("dup_group"):
        if len(grp) > 1:
            if len(grp[LABELS].drop_duplicates()) > 1:
                conflict_count += 1
    qa["conflicting_label_duplicates"] = conflict_count
    log.info("Duplicate groups with conflicting labels: %d", conflict_count)

    return qa


# ------------------------------------------------------------------ #
#  Near-dup summary                                                    #
# ------------------------------------------------------------------ #

def compute_near_dup_summary(df: pd.DataFrame, train_idx, val_idx, test_idx) -> dict:
    """Compute near_dup_summary.json stats after splits are known."""
    grp_col = "dup_group_v2"
    grp_sizes = df[grp_col].value_counts()
    clusters_ge2 = int((grp_sizes >= 2).sum())
    rows_in_clusters_ge2 = int(grp_sizes[grp_sizes >= 2].sum())
    largest_cluster = int(grp_sizes.max())

    # cross-split check
    train_set = set(df.loc[train_idx, grp_col].unique())
    val_set   = set(df.loc[val_idx,   grp_col].unique())
    test_set  = set(df.loc[test_idx,  grp_col].unique())
    cross_split = len(train_set & val_set) + len(train_set & test_set) + len(val_set & test_set)

    # conflicting labels within near-dup clusters
    conflict_count = 0
    for _, grp in df.groupby(grp_col):
        if len(grp) > 1 and len(grp[LABELS].drop_duplicates()) > 1:
            conflict_count += 1

    total_clusters = df[grp_col].nunique()

    return {
        "total_clusters": int(total_clusters),
        "clusters_ge2": clusters_ge2,
        "rows_in_clusters_ge2": rows_in_clusters_ge2,
        "largest_cluster_size": largest_cluster,
        "cross_split_clusters": int(cross_split),
        "clusters_with_conflicting_labels": conflict_count,
        "near_dup_config": NEAR_DUP_CFG,
    }


# ------------------------------------------------------------------ #
#  Split                                                               #
# ------------------------------------------------------------------ #

def build_splits(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multilabel-stratified 80/10/10 split on dup_group_v2.
    Groups are kept together; group label = max over members.
    """
    log.info("Building multilabel-stratified splits on dup_group_v2 (seed=%d)...", SEED)
    split_cfg = CFG["splits"]
    val_frac  = split_cfg["val"]
    test_frac = split_cfg["test"]

    group_df = (
        df.groupby("dup_group_v2")[LABELS]
        .max()
        .reset_index()
    )
    X_groups = group_df["dup_group_v2"].values.reshape(-1, 1)
    Y_groups = group_df[LABELS].values

    temp_frac = val_frac + test_frac
    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=temp_frac, random_state=SEED
    )
    train_g_idx, temp_g_idx = next(msss1.split(X_groups, Y_groups))

    val_frac_of_temp = val_frac / temp_frac
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=(1 - val_frac_of_temp), random_state=SEED
    )
    X_temp = X_groups[temp_g_idx]
    Y_temp = Y_groups[temp_g_idx]
    val_local_idx, test_local_idx = next(msss2.split(X_temp, Y_temp))

    train_groups = set(group_df.iloc[train_g_idx]["dup_group_v2"])
    val_groups   = set(group_df.iloc[temp_g_idx[val_local_idx]]["dup_group_v2"])
    test_groups  = set(group_df.iloc[temp_g_idx[test_local_idx]]["dup_group_v2"])

    assert train_groups.isdisjoint(val_groups),  "Train/val overlap in dup_group_v2!"
    assert train_groups.isdisjoint(test_groups), "Train/test overlap in dup_group_v2!"
    assert val_groups.isdisjoint(test_groups),   "Val/test overlap in dup_group_v2!"

    train_idx = df[df["dup_group_v2"].isin(train_groups)].index.tolist()
    val_idx   = df[df["dup_group_v2"].isin(val_groups)].index.tolist()
    test_idx  = df[df["dup_group_v2"].isin(test_groups)].index.tolist()

    log.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_idx), len(val_idx), len(test_idx),
    )
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


# ------------------------------------------------------------------ #
#  Save splits                                                         #
# ------------------------------------------------------------------ #

def save_splits(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    pos_map = {orig: pos for pos, orig in enumerate(df.index)}
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        iloc_positions = [pos_map[i] for i in idx]
        with open(SPLITS_DIR / f"{name}_idx.json", "w") as fh:
            json.dump(iloc_positions, fh)

    # Leakage check on dup_group_v2
    train_grps = set(df.loc[train_idx, "dup_group_v2"])
    val_grps   = set(df.loc[val_idx,   "dup_group_v2"])
    test_grps  = set(df.loc[test_idx,  "dup_group_v2"])
    assert train_grps.isdisjoint(val_grps),   "LEAKAGE: train/val (dup_group_v2)"
    assert train_grps.isdisjoint(test_grps),  "LEAKAGE: train/test (dup_group_v2)"
    assert val_grps.isdisjoint(test_grps),    "LEAKAGE: val/test (dup_group_v2)"
    log.info("Leakage check PASSED — no dup_group_v2 crosses splits.")

    proc_parquet = PROC_DIR / "processed.parquet"
    manifest = {
        "split_version": "v2_near_dup",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "near_dup_config": NEAR_DUP_CFG,
        "processed_parquet_sha256": sha256_path(proc_parquet) if proc_parquet.exists() else None,
        "split_ratios": CFG["splits"],
        "library_versions": get_lib_versions(),
        "splits": {},
    }
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        sdf = df.loc[idx]
        manifest["splits"][name] = {
            "n_rows": int(len(idx)),
            "label_positive_counts": {lbl: int(sdf[lbl].sum()) for lbl in LABELS},
            "label_positive_rates":  {lbl: float(round(sdf[lbl].mean(), 6)) for lbl in LABELS},
        }

    with open(SPLITS_DIR / "split_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("split_manifest.json written.")
    return manifest


# ------------------------------------------------------------------ #
#  Validation gate                                                     #
# ------------------------------------------------------------------ #

def validation_gate(df: pd.DataFrame, train_idx, val_idx, test_idx) -> bool:
    """
    Returns True if all gates pass (and prints compact table).
    Returns False (and stops) if any gate fails.
    """
    total = len(df)
    n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)
    tol_pp = 0.005   # 0.5 percentage points
    rel_tol = 0.15   # 15% relative
    min_rare = 40

    # Gate 1: cross-split dup_group_v2
    train_grps = set(df.loc[train_idx, "dup_group_v2"])
    val_grps   = set(df.loc[val_idx,   "dup_group_v2"])
    test_grps  = set(df.loc[test_idx,  "dup_group_v2"])
    cross = len(train_grps & val_grps) + len(train_grps & test_grps) + len(val_grps & test_grps)
    gate1_ok = cross == 0

    # Gate 2: split sizes within 0.5 pp
    train_r = n_train / total
    val_r   = n_val   / total
    test_r  = n_test  / total
    gate2_ok = (
        abs(train_r - 0.80) <= tol_pp and
        abs(val_r   - 0.10) <= tol_pp and
        abs(test_r  - 0.10) <= tol_pp
    )

    # Gate 3: label rates within 15% relative
    train_df = df.loc[train_idx]
    val_df   = df.loc[val_idx]
    test_df  = df.loc[test_idx]
    max_rel_diff = 0.0
    gate3_ok = True
    for lbl in LABELS:
        tr_rate = train_df[lbl].mean()
        if tr_rate == 0:
            continue
        for sdf in [val_df, test_df]:
            rel = abs(sdf[lbl].mean() - tr_rate) / tr_rate
            max_rel_diff = max(max_rel_diff, rel)
            if rel > rel_tol:
                gate3_ok = False

    # Gate 4: threat and identity_hate >= 40 in val and test
    gate4_ok = True
    rare_labels = ["threat", "identity_hate"]
    gate4_counts = {}
    for lbl in rare_labels:
        for split_name, sdf in [("val", val_df), ("test", test_df)]:
            cnt = int(sdf[lbl].sum())
            gate4_counts[f"{lbl}/{split_name}"] = cnt
            if cnt < min_rare:
                gate4_ok = False

    all_pass = gate1_ok and gate2_ok and gate3_ok and gate4_ok

    if not all_pass:
        log.error("=== VALIDATION GATE FAILED ===")
        log.error("Gate1 (cross-split clusters==0): %s  cross=%d", gate1_ok, cross)
        log.error("Gate2 (split sizes ±0.5pp): %s  train=%.4f val=%.4f test=%.4f",
                  gate2_ok, train_r, val_r, test_r)
        log.error("Gate3 (label rates ±15%% rel): %s  max_rel=%.4f", gate3_ok, max_rel_diff)
        log.error("Gate4 (rare labels >=40): %s  %s", gate4_ok, gate4_counts)
        return False

    # Print compact table
    header = f"{'label':<15} {'train_pos':>9} {'val_pos':>7} {'test_pos':>8} {'maxRelDiff':>10}"
    sep = "-" * len(header)
    log.info("=== VALIDATION GATE PASSED ===")
    log.info(sep)
    log.info(header)
    log.info(sep)
    for lbl in LABELS:
        tp = int(train_df[lbl].sum())
        vp = int(val_df[lbl].sum())
        ep = int(test_df[lbl].sum())
        tr_rate = train_df[lbl].mean()
        rd = 0.0
        if tr_rate > 0:
            rd = max(
                abs(val_df[lbl].mean()  - tr_rate) / tr_rate,
                abs(test_df[lbl].mean() - tr_rate) / tr_rate,
            )
        log.info(f"{lbl:<15} {tp:>9} {vp:>7} {ep:>8} {rd:>10.3%}")
    log.info(sep)
    log.info(
        f"{'TOTAL':<15} {n_train:>9} {n_val:>7} {n_test:>8}  "
        f"({train_r:.2%}/{val_r:.2%}/{test_r:.2%})"
    )
    log.info(sep)
    return True


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
| dup_group | int64 | Exact-duplicate group ID (rows with identical lowercased text_clean) |
| dup_group_v2 | int64 | Near-duplicate group ID (Union-Find over exact dups + MinHash-LSH char-5gram near-dups at Jaccard>=0.8). Component id = smallest positional row index. Used for split construction. |

## Notes
- Label columns are binary int8.
- `comment_text` is never modified.
- Model-specific normalization (lowercasing, special tokens, etc.) happens inside each model adapter, NOT here.
- Duplicates are kept but tracked via `dup_group` and `dup_group_v2`. Split construction uses `dup_group_v2`.
- Rows with empty `text_clean` are dropped.

## File: `data/splits/`

| File | Description |
|---|---|
| `train_idx.json` | Positional (iloc) indices into processed.parquet for the train split |
| `val_idx.json` | Positional (iloc) indices for the validation split |
| `test_idx.json` | Positional (iloc) indices for the test split |
| `split_manifest.json` | Row counts, label positive counts and rates per split, split_version=v2_near_dup |
| `v1_exact_dup/` | Previous split (exact-dup only, for reference) |

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

        # Assign exact dup groups
        log.info("Assigning exact duplicate groups...")
        df["dup_group"] = assign_dup_groups(df, text_col="text_clean")
        log.info("Unique exact dup groups: %d", df["dup_group"].nunique())

        # Drop empty-text rows
        empty_mask = df["text_clean"].str.len() == 0
        if empty_mask.sum() > 0:
            log.warning(
                "Dropping %d rows with empty text_clean", empty_mask.sum()
            )
            df = df[~empty_mask].copy()

        df = df.reset_index(drop=True)

        # QA checks
        qa = run_qa_checks(df)

        # Assign near-dup groups (MinHash-LSH)
        log.info("Assigning near-duplicate groups (MinHash-LSH, threshold=%.2f)...", NEAR_DUP_CFG.get("threshold", 0.8))
        df["dup_group_v2"] = assign_near_dup_groups(df, NEAR_DUP_CFG, text_col="text_clean")
        log.info("Unique near-dup groups: %d", df["dup_group_v2"].nunique())

        # Save processed
        df.to_parquet(PROC_DIR / "processed.parquet", index=False)
        df.to_csv(PROC_DIR / "processed.csv", index=False)
        log.info("Processed data saved: %d rows", len(df))

        # Save data dictionary
        with open(PROC_DIR / "DATA_DICTIONARY.md", "w", encoding="utf-8") as fh:
            fh.write(DATA_DICT_MD)

        # Copy v1 split from current splits dir (already backed up)
        v1_dir = SPLITS_DIR / "v1_exact_dup"
        v1_dir.mkdir(exist_ok=True)
        bk_splits = sorted(PROJECT_ROOT.glob("backup_*/splits"))
        if bk_splits:
            bk = bk_splits[-1]
            for fname in ["train_idx.json", "val_idx.json", "test_idx.json", "split_manifest.json"]:
                src = bk / fname
                if src.exists():
                    shutil.copy2(src, v1_dir / fname)
            log.info("v1_exact_dup splits copied from %s", bk)
        else:
            log.warning("No backup splits found; v1_exact_dup directory is empty.")

        # Build near-dup-safe splits
        train_idx, val_idx, test_idx = build_splits(df)

        # Save splits + manifest
        manifest = save_splits(df, train_idx, val_idx, test_idx)

        # Near-dup summary
        summary = compute_near_dup_summary(df, train_idx, val_idx, test_idx)
        dedup_dir = PROJECT_ROOT / "reports" / "dedup"
        dedup_dir.mkdir(parents=True, exist_ok=True)
        with open(dedup_dir / "near_dup_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        log.info("near_dup_summary.json written.")

        # Validation gate
        passed = validation_gate(df, train_idx, val_idx, test_idx)
        if not passed:
            log.error("Aborting due to validation gate failure.")
            sys.exit(1)

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
