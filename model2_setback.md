# Executive Verdict: Is Model 2 Better Than Model 1?

Yes, Model 2 (BiGRU + Additive Attention, frozen fastText) is objectively superior overall.

It achieves higher discriminative ranking power across the board (Macro PR-AUC 0.7048 vs 0.6750, Mean ROC-AUC 0.9900 vs 0.9858), higher overall classification performance (Macro F1 0.6586 vs 0.6443, Micro F1 0.7773 vs 0.7585), catches substantially more toxic comments (Macro Recall 68.1% vs 63.6%), and runs at 10,789 comments/second on GPU (6.6× faster than Model 1 on CPU).

However, Model 2 is not a clean sweep on every individual label: Model 1 retained a slight F1 edge on the two rarest classes (threat and severe_toxic).

## 1. Head-to-Head Comparison (Test Split: 15,841 comments)

Both evaluated with Val bootstrap median threshold tuning and 1,000 test bootstrap resamples (95% CIs).

| Metric | Model 1 (TF-IDF + LR) | Model 2 (BiGRU + Attn) | Delta (M2 − M1) | Verdict |
|---|---|---|---|---|
| Macro F1 | 0.6443 [0.6162, 0.6689] | 0.6586 [0.6336, 0.6834] | +0.0143 (+1.4 pts) | Model 2 |
| Micro F1 | 0.7585 [0.7462, 0.7706] | 0.7773 [0.7649, 0.7886] | +0.0188 (+1.9 pts) | Model 2 |
| Macro PR-AUC | 0.6750 [0.6457, 0.7091] | 0.7048 [0.6751, 0.7363] | +0.0298 (+3.0 pts) | Model 2 |
| Mean ROC-AUC | 0.9858 [0.9842, 0.9874] | 0.9900 [0.9887, 0.9912] | +0.0042 (+0.4 pts) | Model 2 |
| Macro Recall | 0.6356 [0.6067, 0.6675] | 0.6812 [0.6513, 0.7144] | +0.0456 (+4.6 pts) | Model 2 |
| Macro Precision | 0.6563 [0.6232, 0.6890] | 0.6446 [0.6171, 0.6735] | -0.0117 (−1.2 pts) | Model 1 |
| Exact Match Acc | 0.9196 [0.9155, 0.9237] | 0.9234 [0.9194, 0.9274] | +0.0038 (+0.4 pts) | Model 2 |
| Hamming Loss | 0.0175 [0.0165, 0.0186] | 0.0166 [0.0156, 0.0176] | -0.0009 (better) | Model 2 |

### Per-Label F1 & PR-AUC Breakdown

| Label | Support | Metric | Model 1 (TF-IDF + LR) | Model 2 (BiGRU + Attn) | Delta |
|---|---|---|---|---|---|
| toxic | 1,543 (9.7%) | F1 / PR-AUC | 0.8078 / 0.8945 | 0.8269 / 0.9092 | +1.9 pts F1, +1.5 pts PR-AUC — Model 2 wins |
| severe_toxic | 163 (1.0%) | F1 / PR-AUC | 0.5419 / 0.4816 | 0.5242 / 0.5399 | −1.8 pts F1, +5.8 pts PR-AUC — Tradeoff (M2 ranks better, M1 F1) |
| obscene | 857 (5.4%) | F1 / PR-AUC | 0.8284 / 0.9003 | 0.8414 / 0.9133 | +1.3 pts F1, +1.3 pts PR-AUC — Model 2 wins |
| threat | 50 (0.3%) | F1 / PR-AUC | 0.4948 / 0.4979 | 0.4567 / 0.5092 | −3.8 pts F1, +1.1 pts PR-AUC — Tradeoff (M2 higher recall, M1 higher prec) |
| insult | 792 (5.0%) | F1 / PR-AUC | 0.7000 / 0.7728 | 0.7269 / 0.7773 | +2.7 pts F1, +0.5 pts PR-AUC — Model 2 wins |
| identity_hate | 142 (0.9%) | F1 / PR-AUC | 0.4930 / 0.5028 | 0.5754 / 0.5801 | +8.2 pts F1, +7.7 pts PR-AUC — Model 2 dominates |

## 2. Useful Insights: Why Did Model 2 Win Most Labels?

### Semantic Generalization Rescues Low-Resource Classes (identity_hate +8.2% F1):

In Model 1, TF-IDF required exact lexical n-gram matches. If an identity group slur or target had a slight variation or synonym not heavily weighted in training, TF-IDF predicted 0.

In Model 2, the 300-d fastText embeddings projected slurs, protected identity groups, and hateful descriptors into dense semantic clusters. This yielded a massive jump for identity_hate (F1 0.4930 → 0.5754, PR-AUC 0.5028 → 0.5801).

### Context & Attention Pooling Overcomes Bag-of-Words Blindness:

Model 1 treats words as an unordered bag. Words like "kill" or "die" trigger toxicity weights even in benign contexts ("this joke made me die laughing").

Model 2's bidirectional GRU processes sequential context, and the additive attention layer dynamically focuses on the toxic climax of a sentence while suppressing benign context.

Test set attention analysis reveals the highest-attended tokens were toxic anchors (idiot, bitch, stupid, fucking, asshole), demonstrating that attention functions as an effective saliency filter.

### High-Throughput Production Profile:

Model 2 delivers 10,789 comments/second on an entry-level mobile GPU (RTX 3060), processing 1,000 comments in 92.7 ms (including text cleaning, regex tokenization, and GPU inference). This is 6.6× faster than Model 1 on an 8-core CPU (1,631.9 comments/sec).

## 3. What "Went Wrong" with Model 2? (Why Didn't It Beat Model 1 on Every Label?)

Despite the architectural upgrade, Model 2 fell behind Model 1 on F1 for threat (0.4567 vs 0.4948) and severe_toxic (0.5242 vs 0.5419). Here is the technical root cause:

### A. Lowercasing Discarded the Strongest EDA Signal (ALL CAPS)

EDA Finding: In Analysis (h), ALL CAPS and uppercase ratio were among the strongest predictors of severe toxicity and threats (shouting, aggressive rants).

The Gap: Model 1 used character 3–5 grams that preserved casing (DIE, FUCK, KILL had distinct high-weight features).

Model 2 Constraint: The Model 2 specification required lowercasing before tokenization. This threw away the capitalization signal entirely. Model 2 is the only model in the benchmark that discarded casing.

### B. Character N-Grams vs Word FastText on Obfuscation & Leetspeak

Online toxicity relies heavily on deliberate character obfuscations (f**k, b!tch, a$$hole, repeated letters fuuuuuck).

Model 1's 100,000 character n-grams effortlessly caught partial sub-strings (e.g. f**, $$h).

Model 2 used a word regex tokenizer with a 50k vocabulary. Any obfuscated token not in the top 50k vocabulary was mapped to `<unk>` (id=1). Furthermore, 13.42% of vocabulary types were missing from fastText and received random normal vectors. Model 1 had zero out-of-vocabulary issues at the character level.

### C. Frozen Pretrained Embeddings (Domain Mismatch)

The fastText vectors were trained on Common Crawl and Wikipedia articles, which are predominantly clean, formal prose.

Because embeddings were frozen during training, the vector space could not adjust to informal internet toxicity, slang, or colloquial insult semantics.

### D. Calibration Distortion & High Thresholds on Minority Labels

Model 2 trained with high positive class weights ($w_1 = \sqrt{N_{neg}/N_{pos}}$: $18.3$ for threat, $10.6$ for identity_hate, $10.0$ for severe_toxic).

This caused Model 2's uncalibrated output probabilities to shift upwards, requiring high decision thresholds (0.83 for threat, 0.88 for severe_toxic).

On threat (only 50 positive examples in 15,841 test rows), Model 2 caught more actual threats than Model 1 (29 vs 24 caught, recall 58% vs 48%), but incurred 48 false positives (vs Model 1's 23), pulling precision down to 37.7% and hurting F1.

## 4. Summary Takeaway for Model 3 (DistilRoBERTa)

| Weakness in Model 1 | Addressed by Model 2? | How Model 3 Will Solve Both |
|---|---|---|
| Bag-of-words (no word order) | Yes (BiGRU + Attention) | Self-attention across all transformer layers |
| No semantic generalization | Yes (fastText embeddings) | Pretrained contextual representations |
| Loss of casing signal | No (forced lowercasing) | Byte-level BPE preserves casing natively |
| Obfuscation/OOV words | No (word-level, 13.4% OOV) | Subword BPE breaks down any typo/obfuscation |
| Frozen representations | No (frozen fastText) | End-to-end transformer fine-tuning |

Model 2 proved that sequence modeling and semantic embeddings produce superior general toxicity detection (+3.0 pts PR-AUC, +8.2 pts identity hate F1). Model 3 (DistilRoBERTa) is positioned to address the remaining subword/casing limitations.