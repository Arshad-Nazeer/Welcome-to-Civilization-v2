
"""
src/wtc/eda.py
--------------
Reusable EDA helper functions called by scripts/03_eda.py.
Every plotting function receives an `ax` or saves directly to a path.
No plt.show() is ever called.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

# ------------------------------------------------------------------ #
#  Style                                                               #
# ------------------------------------------------------------------ #

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
PALETTE = sns.color_palette("muted")
DPI = 200


def savefig(fig: plt.Figure, path: Path, dpi: int = DPI) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ #
#  a) Label distribution                                               #
# ------------------------------------------------------------------ #

def label_distribution(
    df: pd.DataFrame,
    labels: list[str],
    out_dir: Path,
) -> dict:
    counts = df[labels].sum().astype(int)
    total = len(df)
    rates = (counts / total).round(6)
    pos_weight = ((total - counts) / counts.clip(lower=1)).round(4)
    imbalance = (total - counts) / counts.clip(lower=1)

    table = pd.DataFrame({
        "label": labels,
        "positive_count": counts.values,
        "positive_rate": rates.values,
        "pos_weight": pos_weight.values,
        "imbalance_ratio": imbalance.values,
    })
    table.to_csv(out_dir / "a_label_distribution.csv", index=False)

    # Bar chart (linear)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, counts.values, color=PALETTE[:len(labels)])
    ax.set_title("Label distribution (positive counts)", fontsize=14)
    ax.set_ylabel("Count")
    ax.set_xlabel("Label")
    for bar, cnt in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{cnt:,}", ha="center", va="bottom", fontsize=9)
    savefig(fig, out_dir / "a_label_distribution.png")

    # Bar chart (log scale)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, counts.values, color=PALETTE[:len(labels)])
    ax.set_yscale("log")
    ax.set_title("Label distribution (log scale)", fontsize=14)
    ax.set_ylabel("Count (log)")
    ax.set_xlabel("Label")
    savefig(fig, out_dir / "a_label_distribution_log.png")

    return {
        "counts": counts.to_dict(),
        "rates": rates.to_dict(),
        "pos_weight": pos_weight.to_dict(),
        "imbalance_ratio": imbalance.to_dict(),
        "total": int(total),
    }


# ------------------------------------------------------------------ #
#  b) Labels per comment                                               #
# ------------------------------------------------------------------ #

def labels_per_comment(
    df: pd.DataFrame,
    labels: list[str],
    out_dir: Path,
) -> dict:
    n_labels_col = df[labels].sum(axis=1)
    vc = n_labels_col.value_counts().sort_index()
    frac_clean = (n_labels_col == 0).mean()

    table = pd.DataFrame({"n_labels": vc.index, "count": vc.values})
    table.to_csv(out_dir / "b_labels_per_comment.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(vc.index.astype(str), vc.values, color=PALETTE[0])
    ax.set_title("Number of labels per comment", fontsize=14)
    ax.set_xlabel("Number of positive labels")
    ax.set_ylabel("Count")
    savefig(fig, out_dir / "b_labels_per_comment.png")

    return {
        "label_counts_distribution": vc.to_dict(),
        "fraction_clean": float(round(frac_clean, 6)),
    }


# ------------------------------------------------------------------ #
#  c) Co-occurrence                                                    #
# ------------------------------------------------------------------ #

def label_cooccurrence(
    df: pd.DataFrame,
    labels: list[str],
    out_dir: Path,
) -> dict:
    L = df[labels].values.astype(int)
    count_matrix = (L.T @ L)
    count_df = pd.DataFrame(count_matrix, index=labels, columns=labels)
    count_df.to_csv(out_dir / "c_cooccurrence_counts.csv")

    # P(B | A) = count(A & B) / count(A)
    label_counts = np.diag(count_matrix)
    cond_prob = count_matrix / label_counts[:, None].clip(min=1)
    cond_df = pd.DataFrame(cond_prob, index=labels, columns=labels)
    cond_df.to_csv(out_dir / "c_cooccurrence_conditional.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(count_df, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Co-occurrence counts")
    sns.heatmap(cond_df, annot=True, fmt=".2f", cmap="Oranges", vmin=0, vmax=1,
                ax=axes[1])
    axes[1].set_title("P(col | row)")
    fig.suptitle("Label co-occurrence", fontsize=14)
    savefig(fig, out_dir / "c_cooccurrence.png")

    return {
        "cooccurrence_counts": count_df.to_dict(),
        "cooccurrence_conditional": cond_df.round(4).to_dict(),
    }


# ------------------------------------------------------------------ #
#  d) Text length                                                      #
# ------------------------------------------------------------------ #

def text_length_analysis(
    df: pd.DataFrame,
    labels: list[str],
    out_dir: Path,
) -> dict:
    results = {}
    for col in ("n_chars", "n_words"):
        desc = df[col].describe(percentiles=[.25, .5, .75, .90, .95, .99]).round(2)
        results[col] = desc.to_dict()

        # Histogram
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(df[col], bins=100, color=PALETTE[1], edgecolor="none")
        ax.set_title(f"Distribution of {col}", fontsize=14)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        savefig(fig, out_dir / f"d_{col}_hist.png")

        # Boxplot toxic vs clean
        toxic_mask = df[labels].sum(axis=1) > 0
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.boxplot(
            [df.loc[~toxic_mask, col].values, df.loc[toxic_mask, col].values],
            labels=["clean", "toxic"],
            patch_artist=True,
            boxprops=dict(facecolor=PALETTE[2]),
        )
        ax.set_title(f"{col} by toxicity", fontsize=14)
        ax.set_ylabel(col)
        savefig(fig, out_dir / f"d_{col}_boxplot.png")

    # Percentile table
    pct_table = df[["n_chars", "n_words"]].describe(
        percentiles=[.5, .75, .90, .95, .99]
    ).T
    pct_table.to_csv(out_dir / "d_text_length_percentiles.csv")

    return results


# ------------------------------------------------------------------ #
#  e) Tokenized lengths (FIXED: true token positions via offsets)      #
# ------------------------------------------------------------------ #

# Word-boundary inflected profanity lexicon (avoids substring false-positives like "peacock")
_PROFANITY_WB_PATTERNS = [
    r"\bfuck\w*",
    r"\bshit\w*",
    r"\bbitch\w*",
    r"\bass+hole\w*",
    r"\bcunt\w*",
    r"\bcock\w*\b",     # cock but not peacock
    r"\bdick\w*\b",
    r"\bnigger\w*",
    r"\bnigga\w*",
    r"\bfaggot\w*",
    r"\bretard\w*",
    r"\bbastard\w*",
    r"\bwhore\w*",
    r"\bslut\w*",
]
_PROFANITY_WB_RE = re.compile("|".join(_PROFANITY_WB_PATTERNS), re.IGNORECASE)


def _find_profanity_token_position(text: str, encoding) -> int:
    """
    Return the token index of the first profanity match using offset mapping.
    Returns -1 if no match found or no offset map.
    """
    if not hasattr(encoding, "offset_mapping") or encoding.offset_mapping is None:
        return -1
    offsets = encoding.offset_mapping  # list of (char_start, char_end)
    m = _PROFANITY_WB_RE.search(text)
    if m is None:
        return -1
    char_pos = m.start()
    for tok_idx, (cs, ce) in enumerate(offsets):
        if cs <= char_pos < ce:
            return tok_idx
    # char_pos after last token
    return len(offsets)


def tokenized_length_analysis(
    df: pd.DataFrame,
    labels: list[str],
    tokenizer,
    max_len_options: list[int],
    profanity_terms: list[str],   # kept for signature compat; now use _PROFANITY_WB_RE
    out_dir: Path,
) -> dict:
    """
    Compute token length distribution and truncation fractions.
    Uses return_offsets_mapping for true token-position profanity probe.
    tokenizer: a HuggingFace tokenizer (roberta-base).
    """
    from tqdm import tqdm

    supports_offsets = getattr(tokenizer, "is_fast", False)

    texts = df["text_clean"].tolist()
    tok_lengths = []
    # Store offset mappings only for fast tokenizers (needed for profanity probe)
    offset_maps: list | None = [] if supports_offsets else None

    batch_size = 256
    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing"):
        batch = texts[i: i + batch_size]
        if supports_offsets:
            enc = tokenizer(batch, truncation=False, padding=False,
                            add_special_tokens=True,
                            return_offsets_mapping=True)
            tok_lengths.extend(len(ids) for ids in enc["input_ids"])
            offset_maps.extend(enc["offset_mapping"])
        else:
            enc = tokenizer(batch, truncation=False, padding=False,
                            add_special_tokens=True)
            tok_lengths.extend(len(ids) for ids in enc["input_ids"])

    tok_lengths = np.array(tok_lengths)
    df = df.copy()
    df["n_tokens"] = tok_lengths

    pcts = {
        str(p): float(np.percentile(tok_lengths, p))
        for p in [25, 50, 75, 90, 95, 99, 100]
    }

    # Truncation fractions per max_len — overall and per label
    trunc_stats: dict = {"overall": {}, "by_label": {}}
    for ml in max_len_options:
        trunc_mask = tok_lengths > ml
        trunc_stats["overall"][str(ml)] = float(round(trunc_mask.mean(), 6))
        trunc_stats["by_label"][str(ml)] = {}
        for lbl in labels:
            pos_mask = df[lbl].values == 1
            if pos_mask.sum() > 0:
                trunc_stats["by_label"][str(ml)][lbl] = float(
                    round(trunc_mask[pos_mask].mean(), 6)
                )

    # Profanity position probe: toxic comments exceeding 128 tokens
    long_toxic_mask = (tok_lengths > 128) & (df[labels].sum(axis=1) > 0).values
    long_toxic_positions = np.where(long_toxic_mask)[0]
    n_long_toxic = int(long_toxic_mask.sum())
    n_all_toxic = int((df[labels].sum(axis=1) > 0).sum())

    profanity_after_128 = 0
    has_profanity_count = 0

    if supports_offsets and offset_maps is not None:
        for pos in long_toxic_positions:
            text = texts[pos]
            om_obj = offset_maps[pos]
            # Build a simple object with offset_mapping attribute
            class _EncWrapper:
                offset_mapping = om_obj
            tok_pos = _find_profanity_token_position(text, _EncWrapper())
            if tok_pos >= 0:
                has_profanity_count += 1
                if tok_pos > 128:
                    profanity_after_128 += 1
    else:
        # Fallback: character-ratio approximation with word-boundary regex
        for pos in long_toxic_positions:
            text = texts[pos]
            m = _PROFANITY_WB_RE.search(text)
            if m is not None:
                has_profanity_count += 1
                n_chars = len(text)
                n_toks = int(tok_lengths[pos])
                approx_tok_pos = m.start() / max(n_chars, 1) * n_toks
                if approx_tok_pos > 128:
                    profanity_after_128 += 1

    # Rates: share of ALL toxic and share of LONG toxic
    rate_of_all_toxic = profanity_after_128 / max(n_all_toxic, 1)
    rate_of_long_toxic = profanity_after_128 / max(n_long_toxic, 1) if n_long_toxic > 0 else 0.0

    profanity_probe = {
        "n_all_toxic": int(n_all_toxic),
        "n_long_toxic_gt128": int(n_long_toxic),
        "has_profanity_in_long_toxic": int(has_profanity_count),
        "profanity_after_token_128": int(profanity_after_128),
        "rate_of_all_toxic": float(round(rate_of_all_toxic, 6)),
        "rate_of_long_toxic": float(round(rate_of_long_toxic, 6)),
        "used_offset_mapping": bool(supports_offsets),
        "lexicon": "word_boundary_inflected",
        # Legacy field for backward compat
        "profanity_after_128_fraction": float(round(rate_of_long_toxic, 6)),
    }

    # Histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(tok_lengths, bins=100, color=PALETTE[3], edgecolor="none")
    for ml in max_len_options:
        ax.axvline(ml, linestyle="--", linewidth=1.2, label=f"max={ml}")
    ax.set_title("Token length distribution (roberta-base)", fontsize=14)
    ax.set_xlabel("Token count (incl. special tokens)")
    ax.set_ylabel("Count")
    ax.legend()
    savefig(fig, out_dir / "e_token_length_hist.png")

    # Truncation rate bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ml_keys = [str(ml) for ml in max_len_options]
    ax.bar(ml_keys, [trunc_stats["overall"][k] for k in ml_keys], color=PALETTE[4])
    ax.set_title("Fraction of comments truncated by max_len", fontsize=14)
    ax.set_xlabel("max_len")
    ax.set_ylabel("Truncation rate")
    savefig(fig, out_dir / "e_truncation_rate.png")

    return {
        "percentiles": pcts,
        "truncation": trunc_stats,
        "profanity_probe": profanity_probe,
    }


# ------------------------------------------------------------------ #
#  f) Style / obfuscation features by label (FIXED)                   #
# ------------------------------------------------------------------ #

# Exclude emails; match leet-speak including digit substitutions (sh1t, @ss)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_LEET_RE  = re.compile(r"[a-z][*@#$!][a-z]|[a-z][0-9][a-z]", re.IGNORECASE)  # digit leetspeak


def _has_obfuscation(text: str) -> bool:
    """True if text has leet-speak patterns after removing emails."""
    clean = _EMAIL_RE.sub("", text)
    return bool(_LEET_RE.search(clean))


def style_features_by_label(
    df: pd.DataFrame,
    labels: list[str],
    out_dir: Path,
) -> dict:
    df = df.copy()
    df["has_obfuscation"] = df["text_clean"].apply(_has_obfuscation)

    feature_cols = [
        "upper_ratio", "punct_ratio", "non_ascii_ratio",
        "max_repeated_char_run", "has_obfuscation",
    ]
    results = {}
    directions = {}  # feature -> label -> "higher"/"lower"

    for feat in feature_cols:
        means = {}
        feat_directions = {}
        for lbl in labels:
            pos_mask = df[lbl] == 1
            neg_mask = df[lbl] == 0
            toxic_mean = float(round(df.loc[pos_mask, feat].mean(), 6))
            clean_mean = float(round(df.loc[neg_mask, feat].mean(), 6))
            means[lbl] = {"toxic_mean": toxic_mean, "clean_mean": clean_mean}
            feat_directions[lbl] = "higher" if toxic_mean >= clean_mean else "lower"
        results[feat] = means
        directions[feat] = feat_directions

    results["_computed_directions"] = directions

    # Heatmap: mean of each feature for each label (toxic subset)
    heat_data = pd.DataFrame(
        {lbl: {f: df.loc[df[lbl] == 1, f].mean() for f in feature_cols}
         for lbl in labels}
    )
    heat_data.to_csv(out_dir / "f_style_features_by_label.csv")

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
    ax.set_title("Style feature means (positive examples only)", fontsize=14)
    savefig(fig, out_dir / "f_style_features_heatmap.png")

    return results


# ------------------------------------------------------------------ #
#  g) Log-odds lexical analysis                                        #
# ------------------------------------------------------------------ #

def _tokenize_simple(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def log_odds_with_prior(
    pos_texts: list[str],
    neg_texts: list[str],
    alpha: float = 1.0,
    top_n: int = 30,
    ngram: int = 1,
) -> pd.DataFrame:
    """
    Monroe et al. log-odds with informative Dirichlet prior.
    Returns DataFrame with term, log_odds, sorted descending.
    """
    def ngram_counts(texts: list[str], n: int) -> Counter:
        c: Counter = Counter()
        for t in texts:
            tokens = _tokenize_simple(t)
            if n == 1:
                c.update(tokens)
            else:
                c.update(zip(*[tokens[i:] for i in range(n)]))
        return c

    pos_c = ngram_counts(pos_texts, ngram)
    neg_c = ngram_counts(neg_texts, ngram)
    vocab = set(pos_c) | set(neg_c)

    n_pos = sum(pos_c.values())
    n_neg = sum(neg_c.values())

    rows = []
    for term in vocab:
        p = (pos_c[term] + alpha) / (n_pos + alpha * len(vocab))
        q = (neg_c[term] + alpha) / (n_neg + alpha * len(vocab))
        lo = np.log(p / q)
        rows.append({"term": term if ngram == 1 else " ".join(term),
                     "log_odds": lo,
                     "pos_count": pos_c[term],
                     "neg_count": neg_c[term]})
    result = pd.DataFrame(rows).sort_values("log_odds", ascending=False)
    return result.head(top_n)


def lexical_analysis(
    df: pd.DataFrame,
    labels: list[str],
    out_dir: Path,
    top_n: int = 20,
) -> dict:
    results = {}
    clean_mask = df[labels].sum(axis=1) == 0

    for lbl in labels:
        pos_texts = df.loc[df[lbl] == 1, "text_clean"].tolist()
        neg_texts = df.loc[clean_mask, "text_clean"].tolist()

        for ng, ng_name in [(1, "unigram"), (2, "bigram")]:
            lo_df = log_odds_with_prior(pos_texts, neg_texts, top_n=top_n, ngram=ng)
            fname = f"g_{lbl}_{ng_name}_logodds"
            lo_df.to_csv(out_dir / f"{fname}.csv", index=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(lo_df["term"][::-1], lo_df["log_odds"][::-1], color=PALETTE[1])
            ax.set_title(f"Top {top_n} {ng_name}s for '{lbl}' (log-odds)", fontsize=13)
            ax.set_xlabel("Log-odds (vs clean comments)")
            savefig(fig, out_dir / f"{fname}.png")

        results[lbl] = lo_df["term"].tolist()

    return results


# ------------------------------------------------------------------ #
#  h) Identity term analysis (FIXED: compare to overall any-toxic rate)#
# ------------------------------------------------------------------ #

def identity_term_analysis(
    df: pd.DataFrame,
    labels: list[str],
    identity_terms: list[str],
    out_dir: Path,
    high_toxic_threshold: float = 0.3,
) -> dict:
    overall_toxic_rate = float((df[labels].sum(axis=1) > 0).mean())
    rows = []
    for term in identity_terms:
        mask = df["text_clean"].str.contains(
            rf"\b{re.escape(term)}\b", case=False, na=False, regex=True
        )
        count = int(mask.sum())
        toxic_rate = float(round(df.loc[mask, labels].any(axis=1).mean(), 6)) if count > 0 else 0.0
        # Compare per-term rate to overall baseline
        direction = "above_baseline" if toxic_rate > overall_toxic_rate else "below_or_equal"
        rows.append({
            "term": term,
            "frequency": count,
            "toxic_rate": toxic_rate,
            "overall_toxic_rate": round(overall_toxic_rate, 6),
            "relative_to_baseline": round(toxic_rate - overall_toxic_rate, 6),
            "direction": direction,
            "flagged": toxic_rate >= high_toxic_threshold and count > 10,
        })

    table = pd.DataFrame(rows).sort_values("toxic_rate", ascending=False)
    table.to_csv(out_dir / "h_identity_terms.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE[3] if r else PALETTE[0] for r in table["flagged"]]
    ax.barh(table["term"], table["toxic_rate"], color=colors)
    ax.axvline(high_toxic_threshold, linestyle="--", color="red", label=f"threshold={high_toxic_threshold}")
    ax.axvline(overall_toxic_rate, linestyle=":", color="blue", label=f"overall_baseline={overall_toxic_rate:.3f}")
    ax.set_title("Toxic rate for comments containing identity terms", fontsize=13)
    ax.set_xlabel("Toxic rate")
    ax.legend()
    savefig(fig, out_dir / "h_identity_terms.png")

    return {
        "overall_toxic_rate": round(overall_toxic_rate, 6),
        "identity_term_stats": table.to_dict(orient="records"),
        "high_threshold": high_toxic_threshold,
        "flagged_terms": table.loc[table["flagged"], "term"].tolist(),
    }


# ------------------------------------------------------------------ #
#  i) Lexicon-coverage probe (renamed from label_noise_probes)        #
# ------------------------------------------------------------------ #

# Word-boundary inflected profanity regex (same as truncation probe)
_PROFANITY_PROBE_RE = re.compile(
    r"\b(fuck\w*|shit\w*|bitch\w*|ass+hole\w*|cunt\w*|cock\b|dick\b|nigger\w*|nigga\w*|"
    r"faggot\w*|retard\w*|bastard\w*|whore\w*|slut\w*)",
    re.IGNORECASE,
)


def label_noise_probes(
    df: pd.DataFrame,
    labels: list[str],
    out_dir: Path,
    n_sample: int = 30,
    seed: int = 42,
) -> dict:
    """
    Lexicon-coverage probe (not a noise measurement).
    Samples comments where lexicon presence/absence conflicts with labels —
    requires manual review before any noise claim can be made.
    """
    toxic_mask = df[labels].sum(axis=1) > 0
    has_profanity = df["text_clean"].apply(
        lambda t: bool(_PROFANITY_PROBE_RE.search(t))
    )

    # Non-toxic with profanity (lexicon hit, label=0)
    false_neg_pool = df[~toxic_mask & has_profanity]
    false_neg = false_neg_pool.sample(
        n=min(n_sample, len(false_neg_pool)), random_state=seed
    )[["comment_text", "text_clean"] + labels]
    false_neg.to_csv(out_dir / "i_noise_probe_nontoxic_with_profanity.csv", index=False)

    # Toxic without profanity (label=1, no lexicon hit)
    false_pos_pool = df[toxic_mask & ~has_profanity]
    false_pos = false_pos_pool.sample(
        n=min(n_sample, len(false_pos_pool)), random_state=seed
    )[["comment_text", "text_clean"] + labels]
    false_pos.to_csv(out_dir / "i_noise_probe_toxic_no_profanity.csv", index=False)

    return {
        "lexicon": "word_boundary_inflected",
        "note": (
            "This is a lexicon-coverage probe, NOT a noise measurement. "
            "The 30+30 samples require manual review before any noise claim."
        ),
        "nontoxic_with_profanity_pool": int(len(false_neg_pool)),
        "toxic_without_profanity_pool": int(len(false_pos_pool)),
        "sampled_each": n_sample,
    }


# ------------------------------------------------------------------ #
#  j) Rare-label viability                                             #
# ------------------------------------------------------------------ #

def rare_label_viability(
    split_dfs: dict[str, pd.DataFrame],
    labels: list[str],
    out_dir: Path,
    min_positives: int = 100,
) -> dict:
    rows = []
    warnings = []
    for split_name, sdf in split_dfs.items():
        for lbl in labels:
            cnt = int(sdf[lbl].sum())
            rows.append({"split": split_name, "label": lbl, "positive_count": cnt})
            if split_name in ("val", "test") and cnt < min_positives:
                warnings.append(
                    f"WARNING: {lbl} has only {cnt} positives in {split_name} split "
                    f"(threshold={min_positives})"
                )

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "j_rare_label_viability.csv", index=False)

    pivot = table.pivot(index="label", columns="split", values="positive_count")
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, colormap="Set2")
    ax.axhline(min_positives, linestyle="--", color="red", label=f"min={min_positives}")
    ax.set_title("Positive counts per label per split", fontsize=13)
    ax.set_ylabel("Positive count")
    ax.set_xlabel("Label")
    ax.legend()
    savefig(fig, out_dir / "j_rare_label_viability.png")

    return {"viability_table": rows, "warnings": warnings}


# ------------------------------------------------------------------ #
#  k) Split sanity                                                     #
# ------------------------------------------------------------------ #

def split_sanity(
    split_dfs: dict[str, pd.DataFrame],
    labels: list[str],
    out_dir: Path,
) -> dict:
    rates = {}
    for split_name, sdf in split_dfs.items():
        rates[split_name] = {lbl: float(round(sdf[lbl].mean(), 6)) for lbl in labels}

    table = pd.DataFrame(rates).T
    table.to_csv(out_dir / "k_split_label_rates.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    table.T.plot(kind="bar", ax=ax, colormap="tab10")
    ax.set_title("Label positive rates across splits", fontsize=13)
    ax.set_ylabel("Positive rate")
    ax.set_xlabel("Label")
    ax.legend(title="Split")
    savefig(fig, out_dir / "k_split_sanity.png")

    return {"split_label_rates": rates}


# ------------------------------------------------------------------ #
#  l) Vocabulary stats (FIXED: type-level and token-level OOV +       #
#     coverage at 10k/20k/50k/80k)                                    #
# ------------------------------------------------------------------ #

def vocabulary_stats(
    df: pd.DataFrame,
    out_dir: Path,
) -> dict:
    """
    Tokenize with \\w+|[^\\w\\s]; report type-level and token-level OOV
    and token coverage at vocab sizes 10k, 20k, 50k, 80k.
    """
    _TOK_RE = re.compile(r"\w+|[^\w\s]")

    token_counter: Counter = Counter()
    total_tokens = 0
    for text in df["text_clean"]:
        tokens = _TOK_RE.findall(text)
        token_counter.update(t.lower() for t in tokens)
        total_tokens += len(tokens)

    vocab_size = len(token_counter)
    hapax = sum(1 for v in token_counter.values() if v == 1)
    hapax_ratio = hapax / max(vocab_size, 1)

    # Type-level OOV: types appearing only once
    min_freq = 5
    in_vocab_types = sum(1 for v in token_counter.values() if v >= min_freq)
    oov_type_rate = 1 - in_vocab_types / max(vocab_size, 1)

    # Token-level OOV: tokens belonging to out-of-vocab types
    oov_token_count = sum(v for v in token_counter.values() if v < min_freq)
    oov_token_rate = oov_token_count / max(total_tokens, 1)

    # Coverage at different vocab sizes (sorted by frequency)
    sorted_by_freq = sorted(token_counter.values(), reverse=True)
    cumulative = np.cumsum(sorted_by_freq)
    coverage_at: dict[str, float] = {}
    for k in [10000, 20000, 50000, 80000]:
        if k <= len(cumulative):
            coverage_at[str(k)] = float(round(cumulative[k - 1] / max(total_tokens, 1), 6))
        else:
            coverage_at[str(k)] = float(round(cumulative[-1] / max(total_tokens, 1), 6))

    results = {
        "vocab_size": vocab_size,
        "total_tokens": int(total_tokens),
        "hapax_count": hapax,
        "hapax_ratio": float(round(hapax_ratio, 6)),
        "in_vocab_types_min_freq_5": in_vocab_types,
        "oov_type_rate_vs_freq5_vocab": float(round(oov_type_rate, 6)),
        "oov_token_rate_vs_freq5_vocab": float(round(oov_token_rate, 6)),
        "token_coverage_at_vocab_sizes": coverage_at,
        # Legacy alias
        "oov_rate_vs_freq5_vocab": float(round(oov_type_rate, 6)),
    }

    # Save token frequency table (top 1000)
    freq_df = pd.DataFrame(
        token_counter.most_common(1000), columns=["token", "frequency"]
    )
    freq_df.to_csv(out_dir / "l_vocab_top1000.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    top50 = freq_df.head(50)
    ax.bar(top50["token"], top50["frequency"], color=PALETTE[0])
    ax.set_title("Top 50 tokens by frequency", fontsize=13)
    ax.set_xlabel("Token")
    ax.set_ylabel("Frequency")
    plt.xticks(rotation=90)
    savefig(fig, out_dir / "l_vocab_top50.png")

    return results
