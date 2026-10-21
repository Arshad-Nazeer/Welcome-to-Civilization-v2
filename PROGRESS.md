# Welcome to Civilization — Project Progress

**Project**: Multi-label toxicity classification on Jigsaw Toxic Comment dataset  
**Last Updated**: 2026-10-21 00:52 IST  
**Global Seed**: 42  
**Virtual Environment**: `.venv` (Python 3.12.6)

---

## Stage Tracking

| Stage | Name | Status | Date Verified | Outputs / Deliverables | Open Issues |
|---|---|---|---|---|---|
| **1** | **Environment** | **DONE-VERIFIED** | 2026-10-21 | `.venv`, `requirements.txt` (18 pinned deps), `configs/config.yaml`, `src/wtc/config.py`, `src/wtc/utils.py`, `conftest.py`, `.gitignore` | None |
| **2** | **Project Structure** | **DONE-VERIFIED** | 2026-10-21 | Folder structure, `src/wtc/` package, `scripts/01-04`, `notebooks/01_eda.ipynb`, `tests/test_data.py` | None |
| **3** | **Data Import & Raw Save** | **DONE-VERIFIED** | 2026-10-21 | `data/raw/train.parquet` (159,571 rows), `data/raw/test.parquet` (153,164 rows), `data/raw/official_test.parquet` (63,978 rows), `data/raw/raw_manifest.json`, `logs/01_download_2026-10-21.log` | None |
| **4** | **Preprocessing & Splits** | **DONE-VERIFIED** | 2026-10-21 | `data/processed/processed.{parquet,csv}` (159,571 rows, 21 cols), `data/splits/{train,val,test}_idx.json`, `split_manifest.json`, `DATA_DICTIONARY.md`, `qa_report.json`, pytest suite (12/12 passed) | None |
| **5** | **EDA (Analyses a–l)** | **DONE-VERIFIED** | 2026-10-21 | `reports/eda/eda_findings.json`, 24 CSV tables in `reports/eda/tables/`, 27 PNG figures in `reports/eda/figures/` (200 dpi), `logs/03_eda_2026-10-21.log` | None |
| **6** | **Report Building** | **DONE-VERIFIED** | 2026-10-21 | `reports/eda/eda_conclusions.md`, `reports/eda/eda_report.md`, `logs/04_build_report_2026-10-21.log` | None |
| **7** | **Verification** | **DONE-VERIFIED** | 2026-10-21 | End-to-end execution verified (`run_all.ps1`, `run_all.sh`), pytest full suite (12/12 passed), `README.md` updated | None |
| **8** | **Model 1 (TF-IDF + LR)** | **DONE-VERIFIED** | 2026-10-21 | `models/1_tfidf_lr/`, `predictions/1_tfidf_lr/`, `reports/1_tfidf_lr/`, `visualizations/1_tfidf_lr/`, `scripts/10,20,21`, `src/wtc/{evaluation,plotting,inference}.py` | None |
| **9** | **Model 2 (BiGRU + Attn)** | **DONE-VERIFIED** | 2026-10-21 | `models/2_bigru_attn/`, `predictions/2_bigru_attn/`, `reports/2_bigru_attn/`, `visualizations/2_bigru_attn/`, `scripts/11_train_bigru.py`, `src/wtc/bigru_module.py`, `src/wtc/model_specific.py` | None |

---

## Log of Actions

### 2026-10-21
- **Audit completed**: Full audit of existing disk contents; classified stages 1-7; identified Stage 3 as the earliest incomplete stage.
- **Progress tracking initialized**: Created `PROGRESS.md`.
- **Bug fix (`src/wtc/utils.py`)**: Fixed `Timer.__str__` attribute error when invoked within context manager block. (Backup saved to `backup_20261021_0025/`).
- **Data download refinement (`scripts/01_download_data.py`)**: Hugging Face dataset script dropped the required `id` column and omitted the 89,186 unscored test rows. Added validation to trigger raw CSV fallback so `id` is retained and all 153,164 test rows are preserved with `label_available` flag per specification.
- **Stage 3 executed & verified**:
  - `data/raw/train.parquet`: 159,571 rows, 8 columns (`id`, `comment_text`, 6 labels).
  - `data/raw/test.parquet`: 153,164 rows, 9 columns (with `label_available`).
  - `data/raw/official_test.parquet`: 63,978 rows (scored subset).
  - `data/raw/raw_manifest.json`: Verified SHA-256 hashes match files on disk.
- **Stage 4 executed & verified**:
  - `scripts/02_preprocess.py` ran in 77.2s.
  - Features engineered: `text_clean`, `has_url`, `has_ip`, `has_mention`, `n_chars`, `n_words`, `n_labels`, `is_clean`, `upper_ratio`, `non_ascii_ratio`, `punct_ratio`, `max_repeated_char_run`, `dup_group`.
  - QA checks: 0 nulls across all 21 columns, 0 empty comments dropped, 198 exact duplicate texts, 41 conflicting duplicate groups.
  - Processed files saved: `data/processed/processed.parquet` (85.5 MB) and `processed.csv` (144.7 MB).
  - Splitting: Multilabel-stratified 80/10/10 split on duplicate groups.
    - Train: 127,664 rows (80.004%)
    - Val: 15,956 rows (9.999%)
    - Test: 15,951 rows (9.996%)
    - Zero duplicate-group leakage across splits.
  - Test suite: `pytest tests/test_data.py` executed: **12 passed in 4.92s**.
- **Stage 5 executed & verified**:
  - Fixed output routing in `scripts/03_eda.py` (`shutil.move`) so `reports/eda/tables/` contains strictly CSVs and `reports/eda/figures/` contains PNGs. (Backup saved to `backup_20261021_0032/`).
  - Downloaded and cached `roberta-base` Fast Tokenizer.
  - Ran `scripts/03_eda.py` (289.6s runtime).
  - Generated all 12 EDA analyses (a–l):
    - 24 CSV tables in `reports/eda/tables/`.
    - 27 PNG figures in `reports/eda/figures/` (verified 200 DPI).
    - `reports/eda/eda_findings.json` populated with all statistics.
- **Stage 6 executed & verified**:
  - Added explicit reporting of `P(toxic | severe_toxic) = 100%` alongside `P(severe_toxic | toxic) = 10.43%` in `scripts/04_build_report.py`. (Backup saved to `backup_20261021_0045/`).
  - Ran `scripts/04_build_report.py`.
  - Generated `reports/eda/eda_conclusions.md` (12 numbered findings, recommended decisions, risks/limitations, zero hardcoded numbers).
  - Generated `reports/eda/eda_report.md` (comprehensive report, Data Card, TOC, markdown tables, verified image paths).
- **Stage 7 executed & verified**:
  - Created `run_all.ps1` for native Windows PowerShell execution alongside `run_all.sh` and `Makefile`.
  - Fixed project root resolution in `notebooks/01_eda.ipynb` so kernel locates `src/` whether started from root or `notebooks/`. (Backup saved to `backup_20261021_0050/`).
  - Updated `README.md` with accurate 12-test suite description and PowerShell run instructions.
  - Full test suite passed: `12 passed in 4.81s`.
- **Stage 8 executed & verified (Model 1: TF-IDF + Logistic Regression)**:
  - Configuration: `configs/models.yaml` (word 50k, char 100k, L2, liblinear, C=2.0, sqrt neg/pos class weights).
  - Modules implemented: `src/wtc/evaluation.py` (reusable metrics & bootstrapping), `src/wtc/plotting.py` (8x 200 DPI figures), `src/wtc/inference.py` (`Moderator` registry engine).
  - Scripts implemented: `scripts/10_train_tfidf.py`, `scripts/20_evaluate.py`, `scripts/21_build_model_report.py`.
  - Full training ran in 173.6s (2.89 min); 150k features, 8.03 MB model size.
  - Evaluation executed: Val bootstrap median threshold tuning, 1,000 bootstrap resamples on test for 95% CIs.
  - Headline metrics (Test): Macro F1 = 0.6443, Micro F1 = 0.7585, Macro PR-AUC = 0.6750, Mean ROC-AUC = 0.9858.
  - CPU throughput: 1,631.9 comments/sec (0.61 ms/comment).
  - Generated full report: `reports/1_tfidf_lr/model_report.md`.
- **Stage 9 executed & verified (Model 2: BiGRU + Additive Attention)**:
  - Installed PyTorch CUDA 12.4 in `.venv`, pinned in `requirements-gpu.txt`. RTX 3060 Laptop GPU (6.00 GB VRAM) verified.
  - Downloaded and cached 300-d fastText embeddings from `crawl-300d-2M.vec.zip` into `models/2_bigru_attn/embedding_cache.npy` (hit rate: 86.58% types, vocab 50,002).
  - Implemented `src/wtc/bigru_module.py` and `scripts/11_train_bigru.py` with packed padded sequences, spatial dropout 0.25, float32 masked attention pooling, length-bucketed batching, AMP fp16, and resumable checkpoints.
  - Full training completed in 144.92s (2.42 min, 6 epochs, mean epoch time 21.22s, peak VRAM 2,466 MB); best checkpoint at Epoch 5 (val macro PR-AUC: 0.7150).
  - Evaluated with `scripts/20_evaluate.py`: val bootstrap threshold tuning, 1,000 test bootstrap resamples. Headline metrics (Test): Macro F1 = 0.6586, Micro F1 = 0.7773, Macro PR-AUC = 0.7048, Mean ROC-AUC = 0.9900.
  - Inference speed: 10,789.0 comments/sec (0.093 ms/comment on GPU), 1,003.3 comments/sec on CPU (16 threads).
  - Generated full report: `reports/2_bigru_attn/model_report.md` and regenerated Model 1 report with bug fixes.

