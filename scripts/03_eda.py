
#!/usr/bin/env python
"""
scripts/03_eda.py
-----------------
Run all 12 EDA analyses (a–l) on the train split.
Saves all figures to reports/eda/figures/ and tables to reports/eda/tables/.
Saves all computed numbers to reports/eda/eda_findings.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wtc.config import CFG, LABELS, PROJECT_ROOT
from wtc.data import (
    get_processed_dir,
    get_splits_dir,
    load_processed,
    load_split,
)
from wtc.eda import (
    identity_term_analysis,
    label_cooccurrence,
    label_distribution,
    label_noise_probes,
    labels_per_comment,
    lexical_analysis,
    rare_label_viability,
    split_sanity,
    style_features_by_label,
    text_length_analysis,
    tokenized_length_analysis,
    vocabulary_stats,
)
from wtc.utils import Timer, get_logger, set_seed

log = get_logger("03_eda", PROJECT_ROOT / CFG["paths"]["logs"])
set_seed(CFG["seed"])

FIG_DIR   = PROJECT_ROOT / CFG["paths"]["figures"]
TABLE_DIR = PROJECT_ROOT / CFG["paths"]["tables"]
EDA_DIR   = PROJECT_ROOT / CFG["paths"]["reports_eda"]

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    log.info("=== 03_eda.py START ===")
    findings: dict = {}

    with Timer() as total_timer:

        # Load data
        log.info("Loading processed data and splits...")
        processed = load_processed()
        train_df  = load_split("train")
        val_df    = load_split("val")
        test_df   = load_split("test")
        split_dfs = {"train": train_df, "val": val_df, "test": test_df}

        n_train = len(train_df)
        log.info(
            "Dataset sizes — total: %d | train: %d | val: %d | test: %d",
            len(processed), n_train, len(val_df), len(test_df),
        )
        findings["dataset_sizes"] = {
            "total_processed": int(len(processed)),
            "train": int(n_train),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        }

        # ---- a) Label distribution ----
        log.info("(a) Label distribution...")
        with Timer() as t:
            findings["a_label_distribution"] = label_distribution(
                train_df, LABELS, TABLE_DIR
            )
            # Also copy figures
            for fname in ["a_label_distribution.png", "a_label_distribution_log.png"]:
                _copy_if_needed(TABLE_DIR / fname, FIG_DIR / fname)
        log.info("  done in %s", t)

        # ---- b) Labels per comment ----
        log.info("(b) Labels per comment...")
        with Timer() as t:
            findings["b_labels_per_comment"] = labels_per_comment(
                train_df, LABELS, TABLE_DIR
            )
            _copy_if_needed(TABLE_DIR / "b_labels_per_comment.png",
                            FIG_DIR / "b_labels_per_comment.png")
        log.info("  done in %s", t)

        # ---- c) Co-occurrence ----
        log.info("(c) Label co-occurrence...")
        with Timer() as t:
            findings["c_label_cooccurrence"] = label_cooccurrence(
                train_df, LABELS, TABLE_DIR
            )
            _copy_if_needed(TABLE_DIR / "c_cooccurrence.png",
                            FIG_DIR / "c_cooccurrence.png")
        log.info("  done in %s", t)

        # ---- d) Text length ----
        log.info("(d) Text length analysis...")
        with Timer() as t:
            findings["d_text_length"] = text_length_analysis(
                train_df, LABELS, TABLE_DIR
            )
            for fname in [
                "d_n_chars_hist.png", "d_n_chars_boxplot.png",
                "d_n_words_hist.png", "d_n_words_boxplot.png",
            ]:
                _copy_if_needed(TABLE_DIR / fname, FIG_DIR / fname)
        log.info("  done in %s", t)

        # ---- e) Tokenized lengths ----
        log.info("(e) Tokenized length analysis (roberta-base)...")
        with Timer() as t:
            tokenizer = _load_tokenizer()
            if tokenizer is not None:
                findings["e_tokenized_length"] = tokenized_length_analysis(
                    train_df,
                    LABELS,
                    tokenizer,
                    CFG["max_len_options"],
                    CFG["profanity_terms"],
                    TABLE_DIR,
                )
                for fname in ["e_token_length_hist.png", "e_truncation_rate.png"]:
                    _copy_if_needed(TABLE_DIR / fname, FIG_DIR / fname)
            else:
                log.warning("Tokenizer unavailable; skipping (e).")
                findings["e_tokenized_length"] = {"skipped": "tokenizer unavailable"}
        log.info("  done in %s", t)

        # ---- f) Style / obfuscation ----
        log.info("(f) Style and obfuscation features...")
        with Timer() as t:
            findings["f_style_features"] = style_features_by_label(
                train_df, LABELS, TABLE_DIR
            )
            _copy_if_needed(TABLE_DIR / "f_style_features_heatmap.png",
                            FIG_DIR / "f_style_features_heatmap.png")
        log.info("  done in %s", t)

        # ---- g) Lexical analysis ----
        log.info("(g) Log-odds lexical analysis...")
        with Timer() as t:
            findings["g_lexical"] = lexical_analysis(
                train_df, LABELS, TABLE_DIR, top_n=20
            )
            for lbl in LABELS:
                for ng in ["unigram", "bigram"]:
                    fname = f"g_{lbl}_{ng}_logodds.png"
                    _copy_if_needed(TABLE_DIR / fname, FIG_DIR / fname)
        log.info("  done in %s", t)

        # ---- h) Identity term analysis ----
        log.info("(h) Identity term analysis...")
        with Timer() as t:
            findings["h_identity_terms"] = identity_term_analysis(
                train_df, LABELS, CFG["identity_terms"], TABLE_DIR
            )
            _copy_if_needed(TABLE_DIR / "h_identity_terms.png",
                            FIG_DIR / "h_identity_terms.png")
        log.info("  done in %s", t)

        # ---- i) Label noise probes ----
        log.info("(i) Label noise probes...")
        with Timer() as t:
            findings["i_noise_probes"] = label_noise_probes(
                train_df, LABELS, TABLE_DIR, n_sample=30, seed=CFG["seed"]
            )
        log.info("  done in %s", t)

        # ---- j) Rare-label viability ----
        log.info("(j) Rare-label viability...")
        with Timer() as t:
            findings["j_rare_label"] = rare_label_viability(
                split_dfs, LABELS, TABLE_DIR
            )
            _copy_if_needed(TABLE_DIR / "j_rare_label_viability.png",
                            FIG_DIR / "j_rare_label_viability.png")
        log.info("  done in %s", t)

        # ---- k) Split sanity ----
        log.info("(k) Split sanity check...")
        with Timer() as t:
            findings["k_split_sanity"] = split_sanity(split_dfs, LABELS, TABLE_DIR)
            _copy_if_needed(TABLE_DIR / "k_split_sanity.png",
                            FIG_DIR / "k_split_sanity.png")
        log.info("  done in %s", t)

        # ---- l) Vocabulary stats ----
        log.info("(l) Vocabulary stats...")
        with Timer() as t:
            findings["l_vocabulary"] = vocabulary_stats(train_df, TABLE_DIR)
            _copy_if_needed(TABLE_DIR / "l_vocab_top50.png",
                            FIG_DIR / "l_vocab_top50.png")
        log.info("  done in %s", t)

        # Save findings JSON
        findings_path = EDA_DIR / "eda_findings.json"
        with open(findings_path, "w", encoding="utf-8") as fh:
            json.dump(findings, fh, indent=2)
        log.info("EDA findings saved → %s", findings_path)

    log.info("=== 03_eda.py DONE in %s ===", total_timer)


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _load_tokenizer():
    try:
        from transformers import AutoTokenizer
        tokenizer_name = CFG.get("tokenizer_name", "roberta-base")
        log.info("Loading tokenizer: %s", tokenizer_name)
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        log.info("Tokenizer loaded.")
        return tok
    except Exception as exc:
        log.warning("Could not load tokenizer: %s", exc)
        return None


def _copy_if_needed(src: Path, dst: Path) -> None:
    """Move figure from src to dst so tables directory only contains CSVs."""
    if src == dst or not src.exists():
        return
    import shutil
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(src, dst)


if __name__ == "__main__":
    main()
