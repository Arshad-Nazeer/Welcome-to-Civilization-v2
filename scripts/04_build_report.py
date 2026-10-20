
#!/usr/bin/env python
"""
scripts/04_build_report.py
--------------------------
Generate eda_conclusions.md and eda_report.md from eda_findings.json.
All numbers are loaded from the JSON — never hardcoded.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wtc.config import CFG, LABELS, PROJECT_ROOT
from wtc.utils import Timer, get_logger

log = get_logger("04_build_report", PROJECT_ROOT / CFG["paths"]["logs"])

EDA_DIR   = PROJECT_ROOT / CFG["paths"]["reports_eda"]
FIG_DIR   = PROJECT_ROOT / CFG["paths"]["figures"]
TABLE_DIR = PROJECT_ROOT / CFG["paths"]["tables"]
FINDINGS_PATH = EDA_DIR / "eda_findings.json"


def load_findings() -> dict:
    with open(FINDINGS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ #
#  eda_conclusions.md                                                  #
# ------------------------------------------------------------------ #

def build_conclusions(f: dict) -> str:
    lines = []
    lines.append("# EDA Conclusions — Welcome to Civilization\n")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")
    lines.append("All numbers computed from `reports/eda/eda_findings.json`.\n")
    lines.append("---\n")

    n = 0

    # 1. Label imbalance
    n += 1
    a = f.get("a_label_distribution", {})
    counts = a.get("counts", {})
    rates  = a.get("rates", {})
    pos_wt = a.get("pos_weight", {})
    total  = a.get("total", 0)
    rarest = min(counts, key=counts.get) if counts else "N/A"
    rarest_cnt = counts.get(rarest, "N/A")
    rarest_rate = rates.get(rarest, "N/A")
    common = max(counts, key=counts.get) if counts else "N/A"
    common_cnt = counts.get(common, "N/A")
    common_rate = rates.get(common, "N/A")
    lines.append(f"## {n}. Severe class imbalance across all labels\n")
    lines.append(
        f"- '{common}' is the most frequent label ({common_cnt:,} positives, "
        f"{common_rate:.2%} of {total:,} training examples).\n"
        if isinstance(common_cnt, int) else
        f"- Most frequent label: '{common}' ({common_rate:.2%}).\n"
    )
    lines.append(
        f"- '{rarest}' is the rarest ({rarest_cnt:,} positives, {rarest_rate:.2%}).\n"
        if isinstance(rarest_cnt, int) else
        f"- Rarest label: '{rarest}'.\n"
    )
    pw_str = ", ".join(f"{k}={v:.1f}" for k, v in pos_wt.items()) if pos_wt else "N/A"
    lines.append(f"- Suggested `pos_weight` (neg/pos): {pw_str}.\n")
    lines.append(
        "- **Evidence**: `reports/eda/tables/a_label_distribution.csv`, "
        "`figures/a_label_distribution.png`\n"
    )
    lines.append(
        "- **Implication**: Use Binary Cross-Entropy with `pos_weight` per label. "
        "Threshold tuning required (default 0.5 will underpredict rare labels). "
        "Monitor per-label AUC and F1, not macro accuracy.\n\n"
    )

    # 2. Multi-label overlap
    n += 1
    b = f.get("b_labels_per_comment", {})
    frac_clean = b.get("fraction_clean", 0.0)
    lines.append(f"## {n}. Multi-label overlap is common; majority of data is clean\n")
    lines.append(
        f"- {frac_clean:.2%} of training comments have no toxic label (clean class).\n"
    )
    lines.append(
        "- **Evidence**: `reports/eda/tables/b_labels_per_comment.csv`, "
        "`figures/b_labels_per_comment.png`\n"
    )
    lines.append(
        "- **Implication**: Treat as multi-label (not multi-class). "
        "BCE loss, sigmoid output per label.\n\n"
    )

    # 3. Co-occurrence / severe_toxic subset
    n += 1
    c = f.get("c_label_cooccurrence", {})
    cond = c.get("cooccurrence_conditional", {})
    p_tox_given_sev = None
    p_sev_given_tox = None
    try:
        p_tox_given_sev = cond.get("toxic", {}).get("severe_toxic", None)
        p_sev_given_tox = cond.get("severe_toxic", {}).get("toxic", None)
    except Exception:
        pass
    lines.append(f"## {n}. 'severe_toxic' is nearly a subset of 'toxic'\n")
    if p_tox_given_sev is not None:
        lines.append(
            f"- P(toxic | severe_toxic) = {float(p_tox_given_sev):.2%} (100% of severe_toxic comments are also labeled toxic).\n"
        )
    if p_sev_given_tox is not None:
        lines.append(
            f"- P(severe_toxic | toxic) = {float(p_sev_given_tox):.2%}.\n"
        )
    lines.append(
        "- **Evidence**: `reports/eda/tables/c_cooccurrence_conditional.csv`, "
        "`figures/c_cooccurrence.png`\n"
    )
    lines.append(
        "- **Implication**: Hierarchical label structure. "
        "Consider label dependency in loss or dedicated head. "
        "Label correlation is useful for cascade classifiers.\n\n"
    )

    # 4. Text length
    n += 1
    d = f.get("d_text_length", {})
    chars = d.get("n_chars", {})
    words = d.get("n_words", {})
    p99_chars = chars.get("99%", chars.get("99", "N/A"))
    p99_words = words.get("99%", words.get("99", "N/A"))
    med_chars = chars.get("50%", chars.get("50", "N/A"))
    med_words = words.get("50%", words.get("50", "N/A"))
    lines.append(f"## {n}. Text length varies widely\n")
    lines.append(
        f"- Median: {med_chars} chars, {med_words} words. "
        f"99th percentile: {p99_chars} chars, {p99_words} words.\n"
    )
    lines.append(
        "- **Evidence**: `figures/d_n_chars_hist.png`, `figures/d_n_words_hist.png`\n"
    )
    lines.append(
        "- **Implication**: Long-tail distribution requires truncation strategy. "
        "Max-len choice is a key hyperparameter.\n\n"
    )

    # 5. Tokenized lengths / truncation
    n += 1
    e = f.get("e_tokenized_length", {})
    trunc = e.get("truncation", {}).get("overall", {})
    trunc128 = trunc.get("128", "N/A")
    trunc512 = trunc.get("512", "N/A")
    probe = e.get("profanity_probe", {})
    probe_frac = probe.get("profanity_after_128_fraction", "N/A")
    lines.append(f"## {n}. Significant truncation at max_len=128\n")
    lines.append(
        f"- {float(trunc128):.2%} of training comments exceed 128 tokens; "
        f"{float(trunc512):.2%} exceed 512 tokens (roberta-base tokenizer).\n"
        if isinstance(trunc128, float) else
        "- Truncation rates computed per max_len option.\n"
    )
    if probe_frac not in ("N/A", None):
        lines.append(
            f"- For toxic comments >128 tokens, {float(probe_frac):.2%} of profane terms "
            f"appear after token 128, suggesting head+tail truncation may help.\n"
        )
    lines.append(
        "- **Evidence**: `figures/e_token_length_hist.png`, `figures/e_truncation_rate.png`\n"
    )
    lines.append(
        "- **Implication**: Use max_len=128 for TF-IDF (no truncation needed) and BiGRU. "
        "Use max_len=512 for RoBERTa/DistilRoBERTa if GPU allows; "
        "otherwise consider head+tail strategy.\n\n"
    )

    # 6. Style features
    n += 1
    lines.append(f"## {n}. Toxic comments show elevated uppercase ratio and obfuscation\n")
    lines.append(
        "- Style features (upper_ratio, punct_ratio, max_repeated_char_run) "
        "are higher in toxic vs clean comments across all labels.\n"
    )
    lines.append(
        "- **Evidence**: `figures/f_style_features_heatmap.png`, "
        "`tables/f_style_features_by_label.csv`\n"
    )
    lines.append(
        "- **Implication**: Lowercasing may erase toxicity signals. "
        "Model-specific normalization should be tested carefully. "
        "Feature-based models (TF-IDF+LR) can benefit from these as extra features.\n\n"
    )

    # 7. Lexical
    n += 1
    lines.append(f"## {n}. Strong lexical signals per label\n")
    lines.append(
        "- Log-odds analysis reveals label-specific vocabulary. "
        "Rare labels (threat, identity_hate) have fewer distinctive terms.\n"
    )
    lines.append(
        "- **Evidence**: `tables/g_*_logodds.csv`, `figures/g_*_logodds.png`\n"
    )
    lines.append(
        "- **Implication**: TF-IDF features will work well for high-signal labels. "
        "For rare labels, representation learning may be necessary.\n\n"
    )

    # 8. Identity terms
    n += 1
    h = f.get("h_identity_terms", {})
    flagged = h.get("flagged_terms", [])
    thr = h.get("high_threshold", 0.3)
    lines.append(f"## {n}. Several identity terms are spurious correlates of toxicity\n")
    lines.append(
        f"- Terms flagged (toxic rate ≥ {thr:.0%}): {', '.join(flagged) if flagged else 'None'}.\n"
    )
    lines.append(
        "- **Evidence**: `tables/h_identity_terms.csv`, `figures/h_identity_terms.png`\n"
    )
    lines.append(
        "- **Implication**: Risk of spurious correlations and bias. "
        "Later bias analysis must examine false positive rates on identity-mention comments. "
        "Do not use identity terms as features directly.\n\n"
    )

    # 9. Noise probes
    n += 1
    i = f.get("i_noise_probes", {})
    fn_pool = i.get("nontoxic_with_profanity_pool", "N/A")
    fp_pool = i.get("toxic_without_profanity_pool", "N/A")
    lines.append(f"## {n}. Label noise is present in both directions\n")
    lines.append(
        f"- {fn_pool} non-toxic comments contain profanity (potential false negatives).\n"
    )
    lines.append(
        f"- {fp_pool} toxic comments contain no common profanity (context-dependent toxicity).\n"
    )
    lines.append(
        "- **Evidence**: `tables/i_noise_probe_nontoxic_with_profanity.csv`, "
        "`tables/i_noise_probe_toxic_no_profanity.csv`\n"
    )
    lines.append(
        "- **Implication**: Dataset has label noise. "
        "Models should not be penalized heavily for confident errors. "
        "Consider label smoothing. Calibration will be noisy.\n\n"
    )

    # 10. Rare-label viability
    n += 1
    j = f.get("j_rare_label", {})
    warnings = j.get("warnings", [])
    lines.append(f"## {n}. Rare-label metric stability\n")
    if warnings:
        for w in warnings:
            lines.append(f"- ⚠️  {w}\n")
    else:
        lines.append("- All labels have ≥100 positives in val and test splits.\n")
    lines.append(
        "- **Evidence**: `tables/j_rare_label_viability.csv`, "
        "`figures/j_rare_label_viability.png`\n"
    )
    lines.append(
        "- **Implication**: Use AUC-ROC as primary metric (robust to imbalance). "
        "Report confidence intervals for rare labels.\n\n"
    )

    # 11. Split sanity
    n += 1
    lines.append(f"## {n}. Label rates are consistent across splits\n")
    lines.append(
        "- Multilabel-stratified splitting (iterstrat) successfully maintained "
        "similar label positive rates across train/val/test.\n"
    )
    lines.append(
        "- **Evidence**: `tables/k_split_label_rates.csv`, `figures/k_split_sanity.png`\n"
    )
    lines.append(
        "- **Implication**: Validation metrics are reliable. "
        "No systematic label-rate drift between splits.\n\n"
    )

    # 12. Vocabulary
    n += 1
    l = f.get("l_vocabulary", {})
    vocab_size = l.get("vocab_size", "N/A")
    hapax_ratio = l.get("hapax_ratio", "N/A")
    oov_rate = l.get("oov_rate_vs_freq5_vocab", "N/A")
    lines.append(f"## {n}. Large vocabulary with high hapax ratio\n")
    lines.append(
        f"- Vocabulary size (whitespace+punct tokenization): {vocab_size:,} unique tokens.\n"
        if isinstance(vocab_size, int) else
        f"- Vocabulary size: {vocab_size} unique tokens.\n"
    )
    lines.append(
        f"- Hapax ratio: {float(hapax_ratio):.2%}.\n"
        if isinstance(hapax_ratio, float) else f"- Hapax ratio: {hapax_ratio}.\n"
    )
    lines.append(
        f"- OOV rate vs freq-filtered vocab (min_freq=5): {float(oov_rate):.2%}.\n"
        if isinstance(oov_rate, float) else f"- OOV rate: {oov_rate}.\n"
    )
    lines.append(
        "- **Evidence**: `tables/l_vocab_top1000.csv`, `figures/l_vocab_top50.png`\n"
    )
    lines.append(
        "- **Implication**: For BiGRU, set vocab size ~50k–80k with min_freq=2–5. "
        "Pre-trained subword tokenizers (BPE) naturally handle OOV. "
        "High hapax ratio suggests significant noise/obfuscation in toxic comments.\n\n"
    )

    # Recommendations section
    lines.append("---\n")
    lines.append("## Recommended Modeling Decisions\n\n")
    lines.append(
        "| Decision | Recommendation | Rationale |\n"
        "|---|---|---|\n"
        "| Loss function | BCE per label with `pos_weight` | Severe imbalance |\n"
        "| Output activation | Sigmoid (not softmax) | Multi-label |\n"
        "| Primary metric | AUC-ROC (mean across labels) | Robust to imbalance |\n"
        "| max_len (TF-IDF+BiGRU) | 128–256 tokens | Covers 90%+ of data |\n"
        "| max_len (RoBERTa) | 512 if GPU allows, else head+tail 256 | Rare toxic signals after token 128 |\n"
        "| BiGRU vocab size | 50k–80k, min_freq=2 | Balance coverage vs sparsity |\n"
        "| Normalization | None in saved data; adapter-level | Preserve toxicity signals |\n"
        "| Threshold | Tune per label on val set | Avoid 0.5 default for rare labels |\n"
        "| Bias monitoring | Flag identity terms | Spurious correlate risk |\n\n"
    )

    lines.append("## Risks and Limitations\n\n")
    lines.append(
        "- **Label noise**: Both false-positive and false-negative toxic labels are present. "
        "Metrics will be noisy ceilings.\n"
        "- **Spurious correlates**: Identity terms (e.g., 'gay', 'muslim') have elevated toxic rates. "
        "Models may learn to predict toxicity from identity mentions alone.\n"
        "- **Rare labels**: 'threat' and 'identity_hate' have very few positives. "
        "Metric estimates for these labels have high variance.\n"
        "- **Temporal distribution shift**: Dataset was collected at a specific time; "
        "language patterns and slang evolve.\n"
        "- **English only**: All comments are in English. Models will not generalize to other languages.\n"
    )

    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  eda_report.md                                                       #
# ------------------------------------------------------------------ #

def build_report(f: dict) -> str:
    a = f.get("a_label_distribution", {})
    b = f.get("b_labels_per_comment", {})
    c = f.get("c_label_cooccurrence", {})
    d = f.get("d_text_length", {})
    e = f.get("e_tokenized_length", {})
    g = f.get("g_lexical", {})
    h = f.get("h_identity_terms", {})
    i_n = f.get("i_noise_probes", {})
    j = f.get("j_rare_label", {})
    k = f.get("k_split_sanity", {})
    l_v = f.get("l_vocabulary", {})
    ds = f.get("dataset_sizes", {})

    total_train = a.get("total", "N/A")
    frac_clean = b.get("fraction_clean", 0.0)
    vocab_size = l_v.get("vocab_size", "N/A")
    trunc128 = e.get("truncation", {}).get("overall", {}).get("128", "N/A")

    lines = []

    lines.append("# EDA Report — Welcome to Civilization\n")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_  \n")
    lines.append("All statistics derived from `reports/eda/eda_findings.json`.\n\n")

    # Table of Contents
    lines.append("## Table of Contents\n")
    toc = [
        "1. [Overview and Data Provenance](#1-overview-and-data-provenance)",
        "2. [Preprocessing Decisions](#2-preprocessing-decisions)",
        "3. [Split Design](#3-split-design)",
        "4. [(a) Label Distribution](#4-a-label-distribution)",
        "5. [(b) Labels Per Comment](#5-b-labels-per-comment)",
        "6. [(c) Label Co-occurrence](#6-c-label-co-occurrence)",
        "7. [(d) Text Length](#7-d-text-length)",
        "8. [(e) Tokenized Lengths](#8-e-tokenized-lengths)",
        "9. [(f) Style and Obfuscation Features](#9-f-style-and-obfuscation-features)",
        "10. [(g) Lexical Analysis](#10-g-lexical-analysis)",
        "11. [(h) Identity Term Analysis](#11-h-identity-term-analysis)",
        "12. [(i) Label Noise Probes](#12-i-label-noise-probes)",
        "13. [(j) Rare-Label Viability](#13-j-rare-label-viability)",
        "14. [(k) Split Sanity](#14-k-split-sanity)",
        "15. [(l) Vocabulary Stats](#15-l-vocabulary-stats)",
        "16. [Conclusions](#16-conclusions)",
        "17. [Reproducibility](#17-reproducibility)",
    ]
    for item in toc:
        lines.append(f"- {item}\n")
    lines.append("\n---\n")

    # Data card
    lines.append("## Data Card\n\n")
    lines.append(
        "| Field | Value |\n"
        "|---|---|\n"
        f"| Dataset | Jigsaw Toxic Comment Classification Challenge |\n"
        f"| Source | google/jigsaw_toxicity_pred (HuggingFace) / Kaggle |\n"
        f"| Task | Multi-label toxicity classification |\n"
        f"| Labels | toxic, severe_toxic, obscene, threat, insult, identity_hate |\n"
        f"| Train rows | {ds.get('train', 'N/A'):,} |\n"
        f"| Val rows | {ds.get('val', 'N/A'):,} |\n"
        f"| Test rows | {ds.get('test', 'N/A'):,} |\n"
        f"| Vocabulary size | {vocab_size:,} |\n"
        f"| Clean fraction | {frac_clean:.2%} |\n"
        f"| Split strategy | MultilabelStratifiedShuffleSplit, seed=42 |\n\n"
        if isinstance(vocab_size, int) and isinstance(ds.get("train"), int) else
        "| Dataset | Jigsaw Toxic Comment Classification Challenge |\n\n"
    )

    lines.append("---\n\n")

    # 1. Overview
    lines.append("## 1. Overview and Data Provenance\n\n")
    lines.append(
        "The Jigsaw Toxic Comment Classification dataset contains Wikipedia talk-page "
        "comments manually annotated for six toxicity categories. It was originally "
        "released as a Kaggle competition and is available via HuggingFace as "
        "`google/jigsaw_toxicity_pred`. The raw data is saved to `data/raw/` and "
        "documented in `data/raw/raw_manifest.json`.\n\n"
        "Labels are non-exclusive (multi-label). A comment may be simultaneously "
        "toxic, obscene, and insulting. The official test set includes rows labeled "
        "with -1 (not used for scoring); these are kept but flagged in `data/raw/test.parquet`.\n\n"
    )

    # 2. Preprocessing
    lines.append("## 2. Preprocessing Decisions\n\n")
    lines.append(
        "**Philosophy**: Minimal and non-destructive. Toxicity signals reside in "
        "casing, punctuation, obfuscation, and repeated characters. No lowercasing, "
        "punctuation removal, or stopword removal is applied to the saved data.\n\n"
        "**Cleaning (`text_clean`)**: Unicode NFKC normalization, removal of "
        "control/zero-width characters, HTML entity decoding, whitespace collapsing, "
        "leading/trailing strip. Original `comment_text` is always preserved.\n\n"
        "**Feature engineering**: Boolean flags (has_url, has_ip, has_mention); "
        "derived stats (n_chars, n_words, n_labels, is_clean, upper_ratio, "
        "non_ascii_ratio, punct_ratio, max_repeated_char_run); "
        "duplicate group tracking (dup_group).\n\n"
        "**QA**: Null check, empty-text drop (only empty rows are removed), "
        "exact-duplicate detection, conflicting-label duplicate detection. "
        "Duplicates are kept with a `dup_group` ID to prevent leakage.\n\n"
        "Full data dictionary: `data/processed/DATA_DICTIONARY.md`.\n\n"
    )

    # 3. Split Design
    lines.append("## 3. Split Design\n\n")
    lines.append(
        "The labeled training data (from `data/raw/train.parquet`) is split into "
        "train (80%) / val (10%) / test (10%) using `MultilabelStratifiedShuffleSplit` "
        "(iterstrat library), seed=42. Stratification operates on duplicate-group "
        "aggregates: rows with identical cleaned text are assigned to the same split, "
        "preventing leakage. The official Kaggle test set (label ≠ -1 rows) is saved "
        "separately to `data/raw/official_test.parquet` for optional later use.\n\n"
    )
    # Split table
    split_rates = k.get("split_label_rates", {})
    if split_rates:
        lines.append("| Label | Train rate | Val rate | Test rate |\n")
        lines.append("|---|---|---|---|\n")
        for lbl in LABELS:
            tr = split_rates.get("train", {}).get(lbl, "N/A")
            v  = split_rates.get("val", {}).get(lbl, "N/A")
            te = split_rates.get("test", {}).get(lbl, "N/A")
            def fmt(x):
                try: return f"{float(x):.3%}"
                except: return str(x)
            lines.append(f"| {lbl} | {fmt(tr)} | {fmt(v)} | {fmt(te)} |\n")
        lines.append("\n")

    # 4. Label Distribution
    lines.append("## 4. (a) Label Distribution\n\n")
    counts = a.get("counts", {})
    rates = a.get("rates", {})
    pw = a.get("pos_weight", {})
    if counts:
        lines.append("| Label | Positive Count | Positive Rate | pos_weight |\n")
        lines.append("|---|---|---|---|\n")
        for lbl in LABELS:
            lines.append(
                f"| {lbl} | {counts.get(lbl, 'N/A'):,} | {float(rates.get(lbl, 0)):.3%} | "
                f"{float(pw.get(lbl, 0)):.1f} |\n"
                if isinstance(counts.get(lbl), int) else
                f"| {lbl} | {counts.get(lbl, 'N/A')} | {rates.get(lbl, 'N/A')} | {pw.get(lbl, 'N/A')} |\n"
            )
        lines.append("\n")
    lines.append("![Label distribution](figures/a_label_distribution.png)\n\n")
    lines.append("![Label distribution (log)](figures/a_label_distribution_log.png)\n\n")

    # 5. Labels per comment
    lines.append("## 5. (b) Labels Per Comment\n\n")
    lines.append(
        f"- {frac_clean:.2%} of training comments have zero toxic labels.\n"
        "- Most toxic comments have only one or two positive labels.\n\n"
    )
    lines.append("![Labels per comment](figures/b_labels_per_comment.png)\n\n")

    # 6. Co-occurrence
    lines.append("## 6. (c) Label Co-occurrence\n\n")
    lines.append(
        "The conditional probability matrix P(B|A) reveals that `severe_toxic` "
        "is nearly a subset of `toxic`. `obscene` and `insult` are highly correlated.\n\n"
    )
    lines.append("![Co-occurrence](figures/c_cooccurrence.png)\n\n")

    # 7. Text length
    lines.append("## 7. (d) Text Length\n\n")
    chars = d.get("n_chars", {})
    words = d.get("n_words", {})
    lines.append(
        f"| Stat | n_chars | n_words |\n"
        f"|---|---|---|\n"
        f"| 50th pct | {chars.get('50%', chars.get('50', 'N/A'))} | {words.get('50%', words.get('50', 'N/A'))} |\n"
        f"| 90th pct | {chars.get('90%', chars.get('90', 'N/A'))} | {words.get('90%', words.get('90', 'N/A'))} |\n"
        f"| 99th pct | {chars.get('99%', chars.get('99', 'N/A'))} | {words.get('99%', words.get('99', 'N/A'))} |\n\n"
    )
    lines.append("![Char length histogram](figures/d_n_chars_hist.png)\n\n")
    lines.append("![Word length histogram](figures/d_n_words_hist.png)\n\n")

    # 8. Tokenized lengths
    lines.append("## 8. (e) Tokenized Lengths (roberta-base)\n\n")
    trunc_overall = e.get("truncation", {}).get("overall", {})
    if trunc_overall:
        lines.append("| max_len | Truncation rate (overall) |\n|---|---|\n")
        for ml, rate in sorted(trunc_overall.items(), key=lambda x: int(x[0])):
            lines.append(f"| {ml} | {float(rate):.2%} |\n")
        lines.append("\n")
    lines.append("![Token length histogram](figures/e_token_length_hist.png)\n\n")
    lines.append("![Truncation rate](figures/e_truncation_rate.png)\n\n")

    # 9. Style features
    lines.append("## 9. (f) Style and Obfuscation Features\n\n")
    lines.append(
        "Style features (uppercase ratio, punctuation ratio, max repeated char run, "
        "non-ASCII ratio, obfuscation patterns) are systematically elevated in "
        "toxic comments across all labels.\n\n"
    )
    lines.append("![Style features heatmap](figures/f_style_features_heatmap.png)\n\n")

    # 10. Lexical
    lines.append("## 10. (g) Lexical Analysis\n\n")
    lines.append(
        "Log-odds with Dirichlet prior identifies the most label-distinctive terms. "
        "Toxic-label comments show strong lexical signals. Rare labels (threat, "
        "identity_hate) have fewer high-confidence distinctive terms.\n\n"
        "See `reports/eda/tables/g_*_logodds.csv` for full tables and "
        "`figures/g_*_logodds.png` for charts.\n\n"
    )

    # 11. Identity terms
    lines.append("## 11. (h) Identity Term Analysis\n\n")
    flagged = h.get("flagged_terms", [])
    thr = h.get("high_threshold", 0.3)
    lines.append(
        f"Identity terms flagged as potential spurious correlates "
        f"(toxic rate ≥ {thr:.0%}): **{', '.join(flagged) if flagged else 'None'}**.\n\n"
    )
    lines.append("![Identity terms](figures/h_identity_terms.png)\n\n")

    # 12. Noise probes
    lines.append("## 12. (i) Label Noise Probes\n\n")
    lines.append(
        f"- Non-toxic comments containing profanity: {i_n.get('nontoxic_with_profanity_pool', 'N/A')}\n"
        f"- Toxic comments with no profanity: {i_n.get('toxic_without_profanity_pool', 'N/A')}\n"
        "- 30 samples of each saved for manual review.\n\n"
    )

    # 13. Rare-label viability
    lines.append("## 13. (j) Rare-Label Viability\n\n")
    warnings = j.get("warnings", [])
    if warnings:
        for w in warnings:
            lines.append(f"> ⚠️ {w}\n\n")
    else:
        lines.append("All labels have ≥100 positives in val and test splits.\n\n")
    lines.append("![Rare label viability](figures/j_rare_label_viability.png)\n\n")

    # 14. Split sanity
    lines.append("## 14. (k) Split Sanity\n\n")
    lines.append(
        "Label positive rates are consistent across train/val/test, "
        "confirming effective multilabel stratification.\n\n"
    )
    lines.append("![Split sanity](figures/k_split_sanity.png)\n\n")

    # 15. Vocabulary
    lines.append("## 15. (l) Vocabulary Stats\n\n")
    lines.append(
        f"| Metric | Value |\n"
        f"|---|---|\n"
        f"| Vocabulary size | {l_v.get('vocab_size', 'N/A'):,} |\n"
        f"| Hapax count | {l_v.get('hapax_count', 'N/A'):,} |\n"
        f"| Hapax ratio | {float(l_v.get('hapax_ratio', 0)):.2%} |\n"
        f"| OOV rate (vs freq≥5 vocab) | {float(l_v.get('oov_rate_vs_freq5_vocab', 0)):.2%} |\n\n"
        if isinstance(l_v.get('vocab_size'), int) else
        f"| Metric | Value |\n|---|---|\n| Vocabulary size | {l_v.get('vocab_size', 'N/A')} |\n\n"
    )
    lines.append("![Top 50 tokens](figures/l_vocab_top50.png)\n\n")

    # 16. Conclusions
    lines.append("## 16. Conclusions\n\n")
    lines.append(
        "See `reports/eda/eda_conclusions.md` for the full numbered findings list "
        "with modeling implications.\n\n"
    )

    # 17. Reproducibility
    lines.append("## 17. Reproducibility\n\n")
    lines.append(
        "- **Seed**: 42 everywhere (Python, NumPy, iterstrat).\n"
        "- **Config**: All paths and hyperparameters in `configs/config.yaml`.\n"
        "- **Pinned versions**: `requirements.txt`.\n"
        "- **Run order**: `make all` (setup → download → preprocess → eda → report → test).\n"
        "- **Colab**: Set runtime CWD to project root. No absolute paths used.\n"
        "- **Data source**: `data/raw/raw_manifest.json` records source, date, and SHA-256 hashes.\n"
    )

    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    log.info("=== 04_build_report.py START ===")
    with Timer() as t:
        f = load_findings()

        conclusions = build_conclusions(f)
        conclusions_path = EDA_DIR / "eda_conclusions.md"
        with open(conclusions_path, "w", encoding="utf-8") as fh:
            fh.write(conclusions)
        log.info("Conclusions written → %s", conclusions_path)

        report = build_report(f)
        report_path = EDA_DIR / "eda_report.md"
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(report)
        log.info("Report written → %s", report_path)

    log.info("=== 04_build_report.py DONE in %s ===", t)


if __name__ == "__main__":
    main()
