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
