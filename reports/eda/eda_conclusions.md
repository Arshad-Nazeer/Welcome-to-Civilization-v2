# EDA Conclusions — Welcome to Civilization

_Generated: 2026-10-21 03:01:01_

All numbers computed from `reports/eda/eda_findings.json`.

---

## 1. Severe class imbalance across all labels

- 'toxic' is the most frequent label (12,222 positives, 9.56% of 127,861 training examples).

- 'threat' is the rarest (381 positives, 0.30%).

- Suggested `pos_weight` (neg/pos) from train split: toxic=9.5, severe_toxic=99.8, obscene=18.0, threat=334.6, insult=19.3, identity_hate=113.0.

- **Evidence**: `reports/eda/tables/a_label_distribution.csv`, `figures/a_label_distribution.png`

- **Implication**: Use Binary Cross-Entropy with `pos_weight` per label. Threshold tuning required (default 0.5 will underpredict rare labels). Monitor per-label PR-AUC, ROC-AUC, and F1 at thresholds tuned on validation only.


## 2. Multi-label overlap is common; majority of data is clean

- 89.86% of training comments have no toxic label (clean class).

- **Evidence**: `reports/eda/tables/b_labels_per_comment.csv`, `figures/b_labels_per_comment.png`

- **Implication**: Treat as multi-label (not multi-class). BCE loss, sigmoid output per label.


## 3. 'severe_toxic' is nearly a subset of 'toxic'

- P(toxic | severe_toxic) = 100.00% (100% of severe_toxic comments are also labeled toxic).

- P(severe_toxic | toxic) = 10.38%.

- **Evidence**: `reports/eda/tables/c_cooccurrence_conditional.csv`, `figures/c_cooccurrence.png`

- **Implication**: Hierarchical label structure. Consider label dependency in loss or dedicated head. Label correlation is useful for cascade classifiers.


## 4. Text length varies widely

- Median: 204.0 chars, 36.0 words. 99th percentile: 3408.0 chars, 570.0 words.

- **Evidence**: `figures/d_n_chars_hist.png`, `figures/d_n_words_hist.png`

- **Implication**: Long-tail distribution requires truncation strategy. Max-len choice is a key hyperparameter.


## 5. Truncation impact depends on max_len setting

- 18.50% of training comments exceed 128 tokens; 6.59% exceed 256; 2.05% exceed 512 (roberta-base tokenizer).

- Profanity terms appearing after token 128 in long (>128 tok) toxic comments: 167 occurrences — 1.29% of ALL toxic comments, 10.46% of long toxic comments (true token positions via offset mapping). Lexicon uses word-boundary inflected patterns.

- **Evidence**: `figures/e_token_length_hist.png`, `figures/e_truncation_rate.png`

- **Implication**: Evaluate max_len 128 vs 256 empirically (head vs head+tail strategies). No single max_len setting can be recommended without model-in-the-loop validation.


## 6. Style features show label-dependent patterns

- **upper_ratio** is higher in: toxic, severe_toxic, obscene, threat, insult, identity_hate.

- **punct_ratio** is higher in: toxic, severe_toxic, obscene, insult.

- **punct_ratio** is lower in: threat, identity_hate.

- **non_ascii_ratio** is higher in: threat.

- **non_ascii_ratio** is lower in: toxic, severe_toxic, obscene, insult, identity_hate.

- **max_repeated_char_run** is higher in: toxic, severe_toxic, obscene, threat, insult, identity_hate.

- **has_obfuscation** is higher in: obscene, threat.

- **has_obfuscation** is lower in: toxic, severe_toxic, insult, identity_hate.

- Obfuscation detection excludes email addresses; includes digit leet-speak (e.g. sh1t).

- **Evidence**: `figures/f_style_features_heatmap.png`, `tables/f_style_features_by_label.csv`

- **Implication**: Lowercasing may erase toxicity signals. Model-specific normalization should be tested carefully. Feature-based models (TF-IDF+LR) can benefit from these as extra features.


## 7. Strong lexical signals per label

- Log-odds analysis reveals label-specific vocabulary. Rare labels (threat, identity_hate) have fewer distinctive terms.

- **Evidence**: `tables/g_*_logodds.csv`, `figures/g_*_logodds.png`

- **Implication**: TF-IDF features will work well for high-signal labels. For rare labels, representation learning may be necessary.


## 8. Several identity terms are spurious correlates of toxicity

- Overall any-toxic baseline rate: 10.14%.

- Terms with toxic rate above baseline (count > 10): gay, lesbian, jew, transgender, black, woman, white, muslim, asian, women, female, male, jewish, hispanic, islam.

- Terms flagged (toxic rate ≥ 30% AND count > 10): gay, lesbian, jew.

- **Evidence**: `tables/h_identity_terms.csv`, `figures/h_identity_terms.png`

- **Implication**: Risk of spurious correlations and bias. Later bias analysis must examine false positive rates on identity-mention comments. Do not use identity terms as features directly.


## 9. Lexicon-coverage probe (NOT a noise measurement)

- 682 non-toxic comments contain profanity lexicon matches (potential label gaps — may be context-dependent non-toxicity).

- 7322 toxic comments contain no common profanity (context-dependent or identity-based toxicity).

- **Evidence**: `tables/i_noise_probe_nontoxic_with_profanity.csv`, `tables/i_noise_probe_toxic_no_profanity.csv`

- **Implication**: The 30+30 samples require manual review before any label-noise claim. This probe measures lexicon coverage, not annotation quality.


## 10. Rare-label metric stability

- ⚠️  WARNING: threat has only 47 positives in val split (threshold=100)

- ⚠️  WARNING: threat has only 50 positives in test split (threshold=100)

- **Evidence**: `tables/j_rare_label_viability.csv`, `figures/j_rare_label_viability.png`

- **Implication**: Recommended metrics — per-label PR-AUC (average precision), ROC-AUC, and F1 at thresholds tuned on validation only. Report bootstrap confidence intervals for rare labels (threat, identity_hate).


## 11. Label rates across splits

- Max relative label-rate deviation across splits: 5.91% (within ±15% threshold).

- **Evidence**: `tables/k_split_label_rates.csv`, `figures/k_split_sanity.png`

- **Implication**: Validation metrics are reliable if drift is within bounds. Monitor for systematic drift in production.


## 12. Vocabulary size and OOV analysis

- Vocabulary size (\w+|[^\w\s] tokenization): 166,654 unique types.

- Hapax ratio (type-level, freq=1): 52.41%.

- Type-level OOV rate vs freq≥5 vocab: 77.20%.

- Token-level OOV rate vs freq≥5 vocab: 1.78%.

- Token coverage by vocab size: 10000 types → 95.00%, 20000 types → 97.00%, 50000 types → 98.61%, 80000 types → 99.21%.

- **Evidence**: `tables/l_vocab_top1000.csv`, `figures/l_vocab_top50.png`

- **Implication**: For BiGRU, base vocab size on the coverage curve — aim for ≥95% token coverage (approximately 20k based on computed curve). Pre-trained subword tokenizers (BPE) naturally handle OOV.


---

## Recommended Modeling Decisions


| Label | pos_weight (from train) |
|---|---|
| toxic | 9.46 |
| severe_toxic | 99.76 |
| obscene | 17.97 |
| threat | 334.59 |
| insult | 19.33 |
| identity_hate | 112.96 |


| Decision | Recommendation | Rationale |
|---|---|---|
| Loss function | BCE per label with `pos_weight` | Severe imbalance |
| Output activation | Sigmoid (not softmax) | Multi-label |
| Primary metric | Per-label PR-AUC (average precision) | Most informative under imbalance |
| Secondary metrics | ROC-AUC, F1 at val-tuned thresholds | Complementary views |
| Rare-label CIs | Bootstrap confidence intervals | High variance for threat/identity_hate |
| max_len (transformer) | Evaluate 128 vs 256; head vs head+tail | Computed rates; no single answer |
| BiGRU vocab size | Based on 95% token-coverage threshold | See l_vocabulary |
| Normalization | None in saved data; adapter-level | Preserve toxicity signals |
| Threshold | Tune per label on val set | Avoid 0.5 default for rare labels |
| Bias monitoring | Flag identity terms | Spurious correlate risk |


## Risks and Limitations


- **Label noise**: Lexicon-coverage probes suggest some annotation inconsistency. Requires manual review before quantifying noise level.
- **Spurious correlates**: Identity terms show varying toxic rates relative to baseline. Models may learn spurious identity-toxicity associations.
- **Rare labels**: 'threat' and 'identity_hate' have very few positives. Metric estimates for these labels have high variance; bootstrap CIs recommended.
- **Temporal distribution shift**: Dataset was collected at a specific time; language patterns and slang evolve.
- **English only**: All comments are in English. Models will not generalize to other languages.
