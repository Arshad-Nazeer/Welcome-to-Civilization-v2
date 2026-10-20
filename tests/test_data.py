
"""
tests/test_data.py
------------------
Data integrity tests for the Welcome to Civilization pipeline.
Run with: pytest tests/ -v
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wtc.config import CFG, LABELS, PROJECT_ROOT
from wtc.data import (
    get_processed_dir,
    get_splits_dir,
    load_processed,
    load_split,
    load_split_manifest,
)

SPLITS_DIR = get_splits_dir()
PROC_DIR = get_processed_dir()


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def processed() -> pd.DataFrame:
    return load_processed()


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    return load_split("train")


@pytest.fixture(scope="module")
def val_df() -> pd.DataFrame:
    return load_split("val")


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    return load_split("test")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return load_split_manifest()


@pytest.fixture(scope="module")
def train_idx() -> list:
    with open(SPLITS_DIR / "train_idx.json") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def val_idx() -> list:
    with open(SPLITS_DIR / "val_idx.json") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def test_idx() -> list:
    with open(SPLITS_DIR / "test_idx.json") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ #
#  T1: No split overlap (row indices)                                  #
# ------------------------------------------------------------------ #

def test_no_index_overlap_train_val(train_idx, val_idx):
    overlap = set(train_idx) & set(val_idx)
    assert len(overlap) == 0, f"Train/val overlap: {len(overlap)} shared indices"


def test_no_index_overlap_train_test(train_idx, test_idx):
    overlap = set(train_idx) & set(test_idx)
    assert len(overlap) == 0, f"Train/test overlap: {len(overlap)} shared indices"


def test_no_index_overlap_val_test(val_idx, test_idx):
    overlap = set(val_idx) & set(test_idx)
    assert len(overlap) == 0, f"Val/test overlap: {len(overlap)} shared indices"


# ------------------------------------------------------------------ #
#  T2: All indices covered                                             #
# ------------------------------------------------------------------ #

def test_all_rows_covered(processed, train_idx, val_idx, test_idx):
    all_idx = sorted(train_idx + val_idx + test_idx)
    assert all_idx == list(range(len(processed))), \
        "Split indices do not cover all processed rows exactly once"


# ------------------------------------------------------------------ #
#  T3: No leakage across duplicate groups                              #
# ------------------------------------------------------------------ #

def test_no_dup_group_leakage(train_df, val_df, test_df):
    train_groups = set(train_df["dup_group"].unique())
    val_groups   = set(val_df["dup_group"].unique())
    test_groups  = set(test_df["dup_group"].unique())

    tv_overlap = train_groups & val_groups
    tt_overlap = train_groups & test_groups
    vt_overlap = val_groups & test_groups

    assert len(tv_overlap) == 0, \
        f"Dup_group leakage train/val: {len(tv_overlap)} shared groups"
    assert len(tt_overlap) == 0, \
        f"Dup_group leakage train/test: {len(tt_overlap)} shared groups"
    assert len(vt_overlap) == 0, \
        f"Dup_group leakage val/test: {len(vt_overlap)} shared groups"


# ------------------------------------------------------------------ #
#  T4: Label columns are binary (0 or 1 only)                         #
# ------------------------------------------------------------------ #

def test_labels_binary(processed):
    for lbl in LABELS:
        unique_vals = set(processed[lbl].unique())
        assert unique_vals.issubset({0, 1}), \
            f"Label '{lbl}' has non-binary values: {unique_vals}"


# ------------------------------------------------------------------ #
#  T5: Split sizes within tolerance of config ratios                   #
# ------------------------------------------------------------------ #

def test_split_sizes_within_tolerance(
    processed, train_idx, val_idx, test_idx, manifest
):
    total = len(processed)
    expected_train = CFG["splits"]["train"]
    expected_val   = CFG["splits"]["val"]
    expected_test  = CFG["splits"]["test"]
    tol = 0.02  # ±2 percentage points

    actual_train = len(train_idx) / total
    actual_val   = len(val_idx)   / total
    actual_test  = len(test_idx)  / total

    assert abs(actual_train - expected_train) <= tol, \
        f"Train rate {actual_train:.3f} deviates from {expected_train} by >{tol}"
    assert abs(actual_val - expected_val) <= tol, \
        f"Val rate {actual_val:.3f} deviates from {expected_val} by >{tol}"
    assert abs(actual_test - expected_test) <= tol, \
        f"Test rate {actual_test:.3f} deviates from {expected_test} by >{tol}"


# ------------------------------------------------------------------ #
#  T6: Row counts match manifest                                       #
# ------------------------------------------------------------------ #

def test_row_counts_match_manifest(
    train_df, val_df, test_df, train_idx, val_idx, test_idx, manifest
):
    manifest_splits = manifest.get("splits", {})
    for split_name, df, idx in [
        ("train", train_df, train_idx),
        ("val",   val_df,   val_idx),
        ("test",  test_df,  test_idx),
    ]:
        expected = manifest_splits.get(split_name, {}).get("n_rows", None)
        if expected is not None:
            assert len(df) == expected, \
                f"{split_name}: manifest says {expected} rows, got {len(df)}"
            assert len(idx) == expected, \
                f"{split_name}_idx.json has {len(idx)} entries, manifest says {expected}"


# ------------------------------------------------------------------ #
#  T7: No empty text_clean                                             #
# ------------------------------------------------------------------ #

def test_no_empty_text_clean(processed):
    empty = (processed["text_clean"].str.len() == 0).sum()
    assert empty == 0, f"{empty} rows have empty text_clean"


# ------------------------------------------------------------------ #
#  T8: Required columns present                                        #
# ------------------------------------------------------------------ #

REQUIRED_COLS = [
    "id", "comment_text", "text_clean", "dup_group",
    "n_chars", "n_words", "n_labels", "is_clean",
    "upper_ratio", "non_ascii_ratio", "punct_ratio", "max_repeated_char_run",
    "has_url", "has_ip", "has_mention",
] + LABELS

def test_required_columns_present(processed):
    missing = [c for c in REQUIRED_COLS if c not in processed.columns]
    assert len(missing) == 0, f"Missing columns: {missing}"


# ------------------------------------------------------------------ #
#  T9: n_labels == sum of label columns                                #
# ------------------------------------------------------------------ #

def test_n_labels_consistent(processed):
    computed = processed[LABELS].sum(axis=1)
    assert (computed == processed["n_labels"]).all(), \
        "n_labels does not equal sum of label columns"


# ------------------------------------------------------------------ #
#  T10: is_clean is inverse of any label                               #
# ------------------------------------------------------------------ #

def test_is_clean_consistent(processed):
    expected_clean = (processed[LABELS].sum(axis=1) == 0).astype(int)
    assert (expected_clean == processed["is_clean"]).all(), \
        "is_clean is not consistent with label columns"


# ------------------------------------------------------------------ #
#  T11: No dup_group_v2 cluster crosses splits (near-dup leakage)     #
# ------------------------------------------------------------------ #

def test_no_near_dup_group_v2_leakage(train_df, val_df, test_df):
    """Verify that no near-duplicate cluster (dup_group_v2) spans multiple splits."""
    if "dup_group_v2" not in train_df.columns:
        pytest.skip(
            "dup_group_v2 column absent — run 02_preprocess.py with near-dup support first."
        )

    train_groups = set(train_df["dup_group_v2"].unique())
    val_groups   = set(val_df["dup_group_v2"].unique())
    test_groups  = set(test_df["dup_group_v2"].unique())

    tv = train_groups & val_groups
    tt = train_groups & test_groups
    vt = val_groups   & test_groups

    msgs = []
    if tv:
        msgs.append(f"dup_group_v2 leakage train/val: {len(tv)} shared clusters")
    if tt:
        msgs.append(f"dup_group_v2 leakage train/test: {len(tt)} shared clusters")
    if vt:
        msgs.append(f"dup_group_v2 leakage val/test: {len(vt)} shared clusters")

    assert not msgs, "\n".join(msgs)

