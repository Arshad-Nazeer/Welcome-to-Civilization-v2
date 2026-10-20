# EDA Conclusions — Welcome to Civilization

_Generated: 2026-10-21 00:45:40_

All numbers computed from `reports/eda/eda_findings.json`.

---

## 1. Severe class imbalance across all labels

- 'toxic' is the most frequent label (12,236 positives, 9.58% of 127,664 training examples).

- 'threat' is the rarest (383 positives, 0.30%).

- Suggested `pos_weight` (neg/pos): toxic=9.4, severe_toxic=99.1, obscene=17.9, threat=332.3, insult=19.3, identity_hate=112.7.

- **Evidence**: `reports/eda/tables/a_label_distribution.csv`, `figures/a_label_distribution.png`

- **Implication**: Use Binary Cross-Entropy with `pos_weight` per label. Threshold tuning required (default 0.5 will underpredict rare labels). Monitor per-label AUC and F1, not macro accuracy.


## 2. Multi-label overlap is common; majority of data is clean

- 89.83% of training comments have no toxic label (clean class).

- **Evidence**: `reports/eda/tables/b_labels_per_comment.csv`, `figures/b_labels_per_comment.png`

- **Implication**: Treat as multi-label (not multi-class). BCE loss, sigmoid output per label.


## 3. 'severe_toxic' is nearly a subset of 'toxic'

- P(toxic | severe_toxic) = 100.00% (100% of severe_toxic comments are also labeled toxic).

- P(severe_toxic | toxic) = 10.43%.

- **Evidence**: `reports/eda/tables/c_cooccurrence_conditional.csv`, `figures/c_cooccurrence.png`

- **Implication**: Hierarchical label structure. Consider label dependency in loss or dedicated head. Label correlation is useful for cascade classifiers.


## 4. Text length varies widely

- Median: 203.0 chars, 36.0 words. 99th percentile: 3384.85 chars, 567.0 words.

- **Evidence**: `figures/d_n_chars_hist.png`, `figures/d_n_words_hist.png`

- **Implication**: Long-tail distribution requires truncation strategy. Max-len choice is a key hyperparameter.


## 5. Significant truncation at max_len=128

- 18.30% of training comments exceed 128 tokens; 2.06% exceed 512 tokens (roberta-base tokenizer).

- For toxic comments >128 tokens, 19.66% of profane terms appear after token 128, suggesting head+tail truncation may help.

- **Evidence**: `figures/e_token_length_hist.png`, `figures/e_truncation_rate.png`

- **Implication**: Use max_len=128 for TF-IDF (no truncation needed) and BiGRU. Use max_len=512 for RoBERTa/DistilRoBERTa if GPU allows; otherwise consider head+tail strategy.


## 6. Toxic comments show elevated uppercase ratio and obfuscation

- Style features (upper_ratio, punct_ratio, max_repeated_char_run) are higher in toxic vs clean comments across all labels.

- **Evidence**: `figures/f_style_features_heatmap.png`, `tables/f_style_features_by_label.csv`

- **Implication**: Lowercasing may erase toxicity signals. Model-specific normalization should be tested carefully. Feature-based models (TF-IDF+LR) can benefit from these as extra features.


## 7. Strong lexical signals per label

- Log-odds analysis reveals label-specific vocabulary. Rare labels (threat, identity_hate) have fewer distinctive terms.

- **Evidence**: `tables/g_*_logodds.csv`, `figures/g_*_logodds.png`

- **Implication**: TF-IDF features will work well for high-signal labels. For rare labels, representation learning may be necessary.


## 8. Several identity terms are spurious correlates of toxicity

- Terms flagged (toxic rate ≥ 30%): gay, jew, lesbian.

- **Evidence**: `tables/h_identity_terms.csv`, `figures/h_identity_terms.png`

- **Implication**: Risk of spurious correlations and bias. Later bias analysis must examine false positive rates on identity-mention comments. Do not use identity terms as features directly.


## 9. Label noise is present in both directions

- 430 non-toxic comments contain profanity (potential false negatives).

- 8466 toxic comments contain no common profanity (context-dependent toxicity).

- **Evidence**: `tables/i_noise_probe_nontoxic_with_profanity.csv`, `tables/i_noise_probe_toxic_no_profanity.csv`

- **Implication**: Dataset has label noise. Models should not be penalized heavily for confident errors. Consider label smoothing. Calibration will be noisy.


## 10. Rare-label metric stability

- ⚠️  WARNING: threat has only 47 positives in val split (threshold=100)

- ⚠️  WARNING: threat has only 48 positives in test split (threshold=100)

- **Evidence**: `tables/j_rare_label_viability.csv`, `figures/j_rare_label_viability.png`

- **Implication**: Use AUC-ROC as primary metric (robust to imbalance). Report confidence intervals for rare labels.


## 11. Label rates are consistent across splits

- Multilabel-stratified splitting (iterstrat) successfully maintained similar label positive rates across train/val/test.

- **Evidence**: `tables/k_split_label_rates.csv`, `figures/k_split_sanity.png`

- **Implication**: Validation metrics are reliable. No systematic label-rate drift between splits.


## 12. Large vocabulary with high hapax ratio

- Vocabulary size (whitespace+punct tokenization): 402,135 unique tokens.

- Hapax ratio: 65.15%.

- OOV rate vs freq-filtered vocab (min_freq=5): 86.12%.

- **Evidence**: `tables/l_vocab_top1000.csv`, `figures/l_vocab_top50.png`

- **Implication**: For BiGRU, set vocab size ~50k–80k with min_freq=2–5. Pre-trained subword tokenizers (BPE) naturally handle OOV. High hapax ratio suggests significant noise/obfuscation in toxic comments.


---

## Recommended Modeling Decisions


| Decision | Recommendation | Rationale |
|---|---|---|
| Loss function | BCE per label with `pos_weight` | Severe imbalance |
| Output activation | Sigmoid (not softmax) | Multi-label |
| Primary metric | AUC-ROC (mean across labels) | Robust to imbalance |
| max_len (TF-IDF+BiGRU) | 128–256 tokens | Covers 90%+ of data |
| max_len (RoBERTa) | 512 if GPU allows, else head+tail 256 | Rare toxic signals after token 128 |
| BiGRU vocab size | 50k–80k, min_freq=2 | Balance coverage vs sparsity |
| Normalization | None in saved data; adapter-level | Preserve toxicity signals |
| Threshold | Tune per label on val set | Avoid 0.5 default for rare labels |
| Bias monitoring | Flag identity terms | Spurious correlate risk |


## Risks and Limitations


- **Label noise**: Both false-positive and false-negative toxic labels are present. Metrics will be noisy ceilings.
- **Spurious correlates**: Identity terms (e.g., 'gay', 'muslim') have elevated toxic rates. Models may learn to predict toxicity from identity mentions alone.
- **Rare labels**: 'threat' and 'identity_hate' have very few positives. Metric estimates for these labels have high variance.
- **Temporal distribution shift**: Dataset was collected at a specific time; language patterns and slang evolve.
- **English only**: All comments are in English. Models will not generalize to other languages.
