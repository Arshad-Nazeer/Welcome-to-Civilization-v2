
"""
src/wtc/preprocess.py
---------------------
Text cleaning and feature engineering functions.
Philosophy: minimal, non-destructive. Toxicity signals live in casing,
punctuation, obfuscation, and repeated characters — do NOT lowercase,
strip punctuation, or remove stopwords here.
Model-specific normalization happens inside each model adapter.
"""

from __future__ import annotations

import html
import re
import unicodedata

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
#  Cleaning                                                            #
# ------------------------------------------------------------------ #

_CTRL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"     # C0 controls (keep \t \n \r)
    r"\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]"  # zero-width / BOM
)
_WS_RE = re.compile(r"[ \t\r\n]+")
_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+",
    re.IGNORECASE,
)
_IP_RE = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)
_MENTION_RE = re.compile(r"@\w+")


def clean_text(text: str) -> str:
    """
    Apply NFKC normalisation, remove control/zero-width chars,
    decode HTML entities, collapse whitespace. Do NOT change case,
    remove punctuation, or alter URLs/IPs/usernames.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    # 1. HTML entity decode
    text = html.unescape(text)
    # 2. Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)
    # 3. Remove control / zero-width characters
    text = _CTRL_RE.sub("", text)
    # 4. Collapse whitespace (preserve single spaces/newlines)
    text = _WS_RE.sub(" ", text)
    # 5. Strip leading/trailing whitespace
    text = text.strip()
    return text


# ------------------------------------------------------------------ #
#  Boolean flags                                                        #
# ------------------------------------------------------------------ #

def has_url(text: str) -> bool:
    return bool(_URL_RE.search(text))


def has_ip(text: str) -> bool:
    return bool(_IP_RE.search(text))


def has_mention(text: str) -> bool:
    return bool(_MENTION_RE.search(text))


# ------------------------------------------------------------------ #
#  Derived numeric features                                            #
# ------------------------------------------------------------------ #

def upper_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are uppercase."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) > 127) / len(text)


def punct_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if unicodedata.category(c).startswith("P")) / len(text)


def max_repeated_char_run(text: str) -> int:
    """Length of the longest run of the same character."""
    if not text:
        return 0
    max_run = 1
    cur_run = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    return max_run


# ------------------------------------------------------------------ #
#  Duplicate detection                                                 #
# ------------------------------------------------------------------ #

def assign_dup_groups(df: pd.DataFrame, text_col: str = "text_clean") -> pd.Series:
    """
    Assign an integer duplicate-group ID to every row.
    Unique texts get their own group (group id == their positional index in
    the deduplicated series). Returns a Series aligned to df.index.
    """
    normalized = df[text_col].str.strip().str.lower()
    # Map each unique text to its first occurrence index in df
    first_seen = {}
    group_ids = []
    for i, txt in enumerate(normalized):
        if txt not in first_seen:
            first_seen[txt] = i
        group_ids.append(first_seen[txt])
    return pd.Series(group_ids, index=df.index, name="dup_group", dtype="int64")


# ------------------------------------------------------------------ #
#  Near-duplicate detection (MinHash + LSH + Union-Find)              #
# ------------------------------------------------------------------ #

class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            # Always keep the smaller index as root
            if rx < ry:
                self.parent[ry] = rx
            else:
                self.parent[rx] = ry


def assign_near_dup_groups(df: pd.DataFrame, cfg: dict, text_col: str = "text_clean") -> pd.Series:
    """
    Union-Find over:
      (a) exact-duplicate dup_group pairs
      (b) MinHash-LSH near-duplicate candidates with Jaccard >= threshold (char 5-grams).
    Texts with len < min_chars are skipped for LSH (but merged via dup_group).
    Returns dup_group_v2 Series (int; component id = smallest positional index).
    cfg keys: num_perm, threshold, min_chars, seed  (from config.yaml near_dup block).
    """
    from datasketch import MinHash, MinHashLSH

    num_perm = int(cfg.get("num_perm", 128))
    threshold = float(cfg.get("threshold", 0.8))
    min_chars = int(cfg.get("min_chars", 20))
    seed = int(cfg.get("seed", 42))

    n = len(df)
    uf = _UnionFind(n)

    # (a) Merge exact dup_group pairs
    dup_group = assign_dup_groups(df, text_col=text_col)
    pos_map = {orig: pos for pos, orig in enumerate(df.index)}
    group_to_positions: dict = {}
    for idx, grp in zip(df.index, dup_group):
        group_to_positions.setdefault(int(grp), []).append(pos_map[idx])
    for positions in group_to_positions.values():
        root = positions[0]
        for p in positions[1:]:
            uf.union(root, p)

    # (b) MinHash-LSH on lowercase text_clean
    texts_lower = df[text_col].str.strip().str.lower().tolist()

    def make_minhash(text: str) -> MinHash:
        m = MinHash(num_perm=num_perm, seed=seed)
        shingles = {text[i:i+5].encode("utf8") for i in range(len(text) - 4)}
        for s in shingles:
            m.update(s)
        return m

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: dict[int, MinHash] = {}

    for pos, text in enumerate(texts_lower):
        if len(text) < min_chars:
            continue
        m = make_minhash(text)
        minhashes[pos] = m
        lsh.insert(str(pos), m)

    # Query and verify Jaccard >= threshold, then union
    for pos, m in minhashes.items():
        candidates = lsh.query(m)
        for c_str in candidates:
            c = int(c_str)
            if c == pos:
                continue
            j = m.jaccard(minhashes[c])
            if j >= threshold:
                uf.union(pos, c)

    # Build output: component id = smallest positional index in component
    raw_comp = [uf.find(i) for i in range(n)]
    # Map root -> min position in component
    comp_min: dict[int, int] = {}
    for i, r in enumerate(raw_comp):
        if r not in comp_min or i < comp_min[r]:
            comp_min[r] = i
    group_ids = [comp_min[raw_comp[i]] for i in range(n)]

    return pd.Series(group_ids, index=df.index, name="dup_group_v2", dtype="int64")


# ------------------------------------------------------------------ #
#  Full feature-engineering pass                                       #
# ------------------------------------------------------------------ #

def engineer_features(df: pd.DataFrame, text_col: str = "comment_text",
                      labels: list[str] | None = None) -> pd.DataFrame:
    """
    Add all derived columns in-place (operates on a copy).
    Expects df to have `text_col` and optionally `labels` columns.
    """
    df = df.copy()

    # --- Cleaned text ---
    df["text_clean"] = df[text_col].apply(clean_text)

    # --- Boolean flags ---
    df["has_url"]     = df["text_clean"].apply(has_url)
    df["has_ip"]      = df["text_clean"].apply(has_ip)
    df["has_mention"] = df["text_clean"].apply(has_mention)

    # --- Length features ---
    df["n_chars"] = df["text_clean"].str.len()
    df["n_words"] = df["text_clean"].str.split().str.len()

    # --- Label features (only if label cols present) ---
    if labels is not None and all(c in df.columns for c in labels):
        df["n_labels"]  = df[labels].sum(axis=1)
        df["is_clean"]  = (df["n_labels"] == 0).astype(int)

    # --- Style features ---
    df["upper_ratio"]           = df["text_clean"].apply(upper_ratio)
    df["non_ascii_ratio"]       = df["text_clean"].apply(non_ascii_ratio)
    df["punct_ratio"]           = df["text_clean"].apply(punct_ratio)
    df["max_repeated_char_run"] = df["text_clean"].apply(max_repeated_char_run)

    return df
