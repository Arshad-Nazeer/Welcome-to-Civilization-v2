
"""
src/wtc/data.py
---------------
Data loading helpers: read raw parquet/csv, load splits, build DataFrames.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import CFG, LABELS, PROJECT_ROOT


def get_raw_dir() -> Path:
    return PROJECT_ROOT / CFG["paths"]["data_raw"]


def get_processed_dir() -> Path:
    return PROJECT_ROOT / CFG["paths"]["data_processed"]


def get_splits_dir() -> Path:
    return PROJECT_ROOT / CFG["paths"]["data_splits"]


# ------------------------------------------------------------------ #
#  Raw data                                                            #
# ------------------------------------------------------------------ #

def load_raw_train() -> pd.DataFrame:
    p = get_raw_dir() / "train.parquet"
    if p.exists():
        return pd.read_parquet(p)
    p = get_raw_dir() / "train.csv"
    return pd.read_csv(p)


def load_raw_test() -> pd.DataFrame:
    p = get_raw_dir() / "test.parquet"
    if p.exists():
        return pd.read_parquet(p)
    p = get_raw_dir() / "test.csv"
    return pd.read_csv(p)


# ------------------------------------------------------------------ #
#  Processed data                                                      #
# ------------------------------------------------------------------ #

def load_processed() -> pd.DataFrame:
    return pd.read_parquet(get_processed_dir() / "processed.parquet")


# ------------------------------------------------------------------ #
#  Splits                                                              #
# ------------------------------------------------------------------ #

def load_split_manifest() -> dict:
    p = get_splits_dir() / "split_manifest.json"
    with open(p) as fh:
        return json.load(fh)


def load_split(name: str) -> pd.DataFrame:
    """Load a named split (train / val / test) as a DataFrame."""
    processed = load_processed()
    idx_file = get_splits_dir() / f"{name}_idx.json"
    with open(idx_file) as fh:
        indices = json.load(fh)
    return processed.iloc[indices].reset_index(drop=True)


def label_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the six label columns."""
    return df[LABELS].copy()
