# EDA Report — Welcome to Civilization

_Generated: 2026-10-21 03:01:01_  

All statistics derived from `reports/eda/eda_findings.json`.


## Table of Contents

- 1. [Overview and Data Provenance](#1-overview-and-data-provenance)

- 2. [Preprocessing Decisions](#2-preprocessing-decisions)

- 3. [Split Design](#3-split-design)

- 4. [(a) Label Distribution](#4-a-label-distribution)

- 5. [(b) Labels Per Comment](#5-b-labels-per-comment)

- 6. [(c) Label Co-occurrence](#6-c-label-co-occurrence)

- 7. [(d) Text Length](#7-d-text-length)

- 8. [(e) Tokenized Lengths](#8-e-tokenized-lengths)

- 9. [(f) Style and Obfuscation Features](#9-f-style-and-obfuscation-features)

- 10. [(g) Lexical Analysis](#10-g-lexical-analysis)

- 11. [(h) Identity Term Analysis](#11-h-identity-term-analysis)

- 12. [(i) Lexicon-Coverage Probe](#12-i-lexicon-coverage-probe)

- 13. [(j) Rare-Label Viability](#13-j-rare-label-viability)

- 14. [(k) Split Sanity](#14-k-split-sanity)

- 15. [(l) Vocabulary Stats](#15-l-vocabulary-stats)

- 16. [Conclusions](#16-conclusions)

- 17. [Reproducibility](#17-reproducibility)


---

## Data Card


| Field | Value |
|---|---|
| Dataset | Jigsaw Toxic Comment Classification Challenge |
| Source | Kaggle CSV / google/jigsaw_toxicity_pred (HuggingFace) |
| Task | Multi-label toxicity classification |
| Labels | toxic, severe_toxic, obscene, threat, insult, identity_hate |
| Train rows | 127,861 |
| Val rows | 15,869 |
| Test rows | 15,841 |
| Vocabulary size | 166,654 |
| Clean fraction | 89.86% |
| Split strategy | MultilabelStratifiedShuffleSplit on dup_group_v2, seed=42 |


---


## 1. Overview and Data Provenance


The Jigsaw Toxic Comment Classification dataset contains Wikipedia talk-page comments manually annotated for six toxicity categories. It was originally released as a Kaggle competition and is available via HuggingFace as `google/jigsaw_toxicity_pred`. The raw data is saved to `data/raw/` and documented in `data/raw/raw_manifest.json`.

Labels are non-exclusive (multi-label). A comment may be simultaneously toxic, obscene, and insulting. The official test set includes rows labeled with -1 (not used for scoring); these are kept but flagged in `data/raw/test.parquet`.


## 2. Preprocessing Decisions


**Philosophy**: Minimal and non-destructive. Toxicity signals reside in casing, punctuation, obfuscation, and repeated characters. No lowercasing, punctuation removal, or stopword removal is applied to the saved data.

**Cleaning (`text_clean`)**: Unicode NFKC normalization, removal of control/zero-width characters, HTML entity decoding, whitespace collapsing, leading/trailing strip. Original `comment_text` is always preserved.

**Feature engineering**: Boolean flags (has_url, has_ip, has_mention); derived stats (n_chars, n_words, n_labels, is_clean, upper_ratio, non_ascii_ratio, punct_ratio, max_repeated_char_run); duplicate group tracking (dup_group = exact, dup_group_v2 = near-dup via MinHash-LSH).

**QA**: Null check, empty-text drop (only empty rows are removed), exact-duplicate detection, conflicting-label duplicate detection. Duplicates are kept with a `dup_group_v2` ID to prevent leakage.

Full data dictionary: `data/processed/DATA_DICTIONARY.md`.


## 3. Split Design


The labeled training data (from `data/raw/train.parquet`) is split into train (80%) / val (10%) / test (10%) using `MultilabelStratifiedShuffleSplit` (iterstrat library), seed=42. Stratification operates on near-duplicate-group aggregates (`dup_group_v2`, computed via MinHash-LSH at Jaccard≥0.8 on char-5grams): rows in the same near-dup cluster are assigned to the same split, preventing both exact and near-duplicate leakage. The previous exact-dup-only split is preserved in `data/splits/v1_exact_dup/`. The official Kaggle test set (label ≠ -1 rows) is saved separately to `data/raw/official_test.parquet` for optional later use.


| Label | Train rate | Val rate | Test rate |

|---|---|---|---|

| toxic | 9.559% | 9.635% | 9.741% |

| severe_toxic | 0.992% | 1.027% | 1.029% |

| obscene | 5.271% | 5.375% | 5.410% |

| threat | 0.298% | 0.296% | 0.316% |

| insult | 4.919% | 5.010% | 5.000% |

| identity_hate | 0.877% | 0.889% | 0.896% |



## 4. (a) Label Distribution


| Label | Positive Count | Positive Rate | pos_weight |

|---|---|---|---|

| toxic | 12,222 | 9.559% | 9.5 |

| severe_toxic | 1,269 | 0.992% | 99.8 |

| obscene | 6,739 | 5.271% | 18.0 |

| threat | 381 | 0.298% | 334.6 |

| insult | 6,290 | 4.919% | 19.3 |

| identity_hate | 1,122 | 0.877% | 113.0 |



![Label distribution](figures/a_label_distribution.png)


![Label distribution (log)](figures/a_label_distribution_log.png)


## 5. (b) Labels Per Comment


- 89.86% of training comments have zero toxic labels.
- Most toxic comments have only one or two positive labels.


![Labels per comment](figures/b_labels_per_comment.png)


## 6. (c) Label Co-occurrence


The conditional probability matrix P(B|A) reveals that `severe_toxic` is nearly a subset of `toxic`. `obscene` and `insult` are highly correlated.


![Co-occurrence](figures/c_cooccurrence.png)


## 7. (d) Text Length


| Stat | n_chars | n_words |
|---|---|---|
| 50th pct | 204.0 | 36.0 |
| 90th pct | 886.0 | 152.0 |
| 99th pct | 3408.0 | 570.0 |


![Char length histogram](figures/d_n_chars_hist.png)


![Word length histogram](figures/d_n_words_hist.png)


## 8. (e) Tokenized Lengths (roberta-base)


| max_len | Truncation rate (overall) |
|---|---|

| 64 | 40.11% |

| 128 | 18.50% |

| 256 | 6.59% |

| 512 | 2.05% |



**Profanity probe** (offset-mapping): 167 occurrences of profanity terms after token 128 in long toxic comments — 1.29% of ALL toxic comments, 10.46% of long (>128 tok) toxic comments.


![Token length histogram](figures/e_token_length_hist.png)


![Truncation rate](figures/e_truncation_rate.png)


## 9. (f) Style and Obfuscation Features


- **upper_ratio**: higher in [toxic, severe_toxic, obscene, threat, insult, identity_hate], lower in [].

- **punct_ratio**: higher in [toxic, severe_toxic, obscene, insult], lower in [threat, identity_hate].

- **non_ascii_ratio**: higher in [threat], lower in [toxic, severe_toxic, obscene, insult, identity_hate].

- **max_repeated_char_run**: higher in [toxic, severe_toxic, obscene, threat, insult, identity_hate], lower in [].

- **has_obfuscation**: higher in [obscene, threat], lower in [toxic, severe_toxic, insult, identity_hate].



![Style features heatmap](figures/f_style_features_heatmap.png)


## 10. (g) Lexical Analysis


Log-odds with Dirichlet prior identifies the most label-distinctive terms. Toxic-label comments show strong lexical signals. Rare labels (threat, identity_hate) have fewer high-confidence distinctive terms.

See `reports/eda/tables/g_*_logodds.csv` for full tables and `figures/g_*_logodds.png` for charts.


## 11. (h) Identity Term Analysis


Overall any-toxic baseline rate: **10.14%**.


| Term | Count | Toxic rate | vs Baseline | Direction |

|---|---|---|---|---|

| gay | 692 | 58.67% | +48.53% | above_baseline |

| lesbian | 47 | 40.43% | +30.28% | above_baseline |

| jew | 247 | 33.20% | +23.06% | above_baseline |

| transgender | 19 | 26.32% | +16.17% | above_baseline |

| black | 911 | 15.70% | +5.55% | above_baseline |

| woman | 386 | 15.54% | +5.40% | above_baseline |

| white | 955 | 14.66% | +4.52% | above_baseline |

| muslim | 404 | 14.11% | +3.97% | above_baseline |

| asian | 211 | 13.74% | +3.60% | above_baseline |

| women | 580 | 13.28% | +3.13% | above_baseline |



Identity terms flagged as potential spurious correlates (toxic rate ≥ 30%, count > 10): **gay, lesbian, jew**.


![Identity terms](figures/h_identity_terms.png)


## 12. (i) Lexicon-Coverage Probe


- Non-toxic comments with profanity lexicon match: 682
- Toxic comments with no profanity: 7322
- 30 samples of each saved for manual review.
- **Note**: This is a lexicon-coverage probe, NOT a noise measurement. The 30+30 samples require manual review before any noise claim.


## 13. (j) Rare-Label Viability


> ⚠️ WARNING: threat has only 47 positives in val split (threshold=100)


> ⚠️ WARNING: threat has only 50 positives in test split (threshold=100)


![Rare label viability](figures/j_rare_label_viability.png)


## 14. (k) Split Sanity


Max relative label-rate deviation across splits: **5.91%**.


![Split sanity](figures/k_split_sanity.png)


## 15. (l) Vocabulary Stats


| Metric | Value |
|---|---|
| Vocabulary size | 166,654 |
| Total tokens | 10,915,047 |
| Hapax count | 87,345 |
| Hapax ratio | 52.41% |
| Type-level OOV (vs freq≥5 vocab) | 77.20% |
| Token-level OOV (vs freq≥5 vocab) | 1.78% |


| Vocab size | Token coverage |
|---|---|

| 10000 | 95.00% |

| 20000 | 97.00% |

| 50000 | 98.61% |

| 80000 | 99.21% |



![Top 50 tokens](figures/l_vocab_top50.png)


## 16. Conclusions


See `reports/eda/eda_conclusions.md` for the full numbered findings list with modeling implications.


## 17. Reproducibility


- **Seed**: 42 everywhere (Python, NumPy, iterstrat, MinHash).
- **Config**: All paths and hyperparameters in `configs/config.yaml`.
- **Pinned versions**: `requirements.txt`.
- **Run order**: `make all` (setup → download → preprocess → eda → report → test).
- **Colab**: Set runtime CWD to project root. No absolute paths used.
- **Data source**: `data/raw/raw_manifest.json` records source, date, and SHA-256 hashes.
- **Split version**: v2_near_dup (see `data/splits/split_manifest.json`).
