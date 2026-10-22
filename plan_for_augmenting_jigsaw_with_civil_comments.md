# Plan for Augmenting Jigsaw with Civil Comments

## 1. Objective

- Use the **Civil Comments dataset** to obtain additional genuine positive examples for underrepresented labels in the existing Jigsaw toxicity-classification project.
- Reduce class imbalance and potentially improve performance on rare labels.
- Target approximately **10% positive prevalence per label** in the augmented training set.
- Treat 10% as a **soft target**, not a mandatory quota. Never force the dataset to reach it.

## 2. Labels to Map

The Jigsaw model uses six binary labels:

| Jigsaw label | Civil Comments score |
|---|---|
| `toxic` | `toxicity` |
| `severe_toxic` | `severe_toxicity` |
| `obscene` | `obscene` |
| `threat` | `threat` |
| `insult` | `insult` |
| `identity_hate` | `identity_attack` |

- Civil Comments scores are generally between 0 and 1, while Jigsaw labels are binary.
- The mapping is approximate, especially between **`identity_attack`** and **`identity_hate`**.
- Preserve all six mapped labels for every imported comment, rather than labeling a comment positive only for the rare class that led to its selection.

## 3. Convert Scores to Binary Labels

- Do not automatically assume that **`0.5`** is the best threshold for every label.
- Analyze the actual score distribution separately for each label.
- Compare candidate thresholds such as **`0.3`**, **`0.5`**, and **`0.75`**.
- A higher threshold generally favors stronger annotator agreement but produces fewer positive candidates.
- Retain the original continuous scores and make thresholds configurable.
- Handle missing or invalid scores explicitly; do not silently convert them to negative labels.

## 4. Choose Additional Examples

- Prioritize underrepresented labels, particularly **`threat`**, **`identity_hate`**, and **`severe_toxic`**.
- Select genuine positive comments from the additional Civil Comments data.
- Use eligible examples that improve one or more labels.
- Because this is multi-label classification, a single comment may help several underrepresented classes.
- Append as many suitable, unique examples as are available, even if the 10% target cannot be reached.
- Do not lower thresholds indiscriminately simply to meet the target.

## 5. Prevent Duplicates and Data Leakage

Before appending examples:

- Remove exact duplicates.
- Detect normalized duplicates, such as differences in case or whitespace.
- Flag likely near-duplicates for review.
- Check candidates against **all existing training, validation, and test examples**.
- Never append examples that overlap with validation or test data.
- Keep original comment text intact; normalize text only for duplicate detection.

## 6. Keep Evaluation Valid

- Append examples to the **training set only**.
- Leave the existing validation and test sets unchanged.
- Recalculate positive prevalence after deduplication and appending.
- Continue using class-weighted loss initially; evaluate the effect of the additional data rather than changing several training techniques at once.

## 7. Evaluate the Augmented Model

Compare the new model against the current best model, **Model 2b**, using the same validation set.

Measure:

- Macro F1
- Per-label precision, recall, and F1
- Per-label PR-AUC
- False-positive behavior and probability calibration

Keep the augmented model only if the results justify the additional data and do not cause unacceptable regressions.

## 8. Recommended Implementation Sequence

1. Inspect the exact Civil Comments version, columns, and available score fields.
2. Generate per-label score distributions and compare candidate thresholds.
3. Map the six labels and select eligible positive candidates.
4. Remove duplicates and check for cross-split leakage.
5. Append suitable examples using 10% as a soft target.
6. Produce an augmentation report with counts added, rejected, and skipped, plus final per-label prevalence.
7. Train a separate model and compare it with Model 2b on unchanged validation data.

## Final Decision

Build a reproducible, data-driven augmentation pipeline that uses real Civil Comments examples, applies configurable label thresholds, removes duplicates, preserves multi-label annotations, and adds as many eligible examples as it can without forcing a 10% balance.
