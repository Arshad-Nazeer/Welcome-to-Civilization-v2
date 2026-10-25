# Welcome to Civilization — Toxicity Moderation System

A production-grade, full-stack multi-label content moderation platform built on top of the Jigsaw Toxic Comment dataset (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).

## 1. System Architecture

- **Backend**: FastAPI (`backend/`), Python 3.12, PyTorch 2.6.0+cu124, Motor / PyMongo, Pydantic v2.
- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS, Lucide icons, Recharts (`frontend/`).
- **Database**: MongoDB Atlas (`wtc` database, `comments` collection).
- **Models Evaluated**:
  1. `3_distilroberta` (Headline Serving Model): Fine-tuned DistilRoBERTa-base, layer-wise learning rate decay, damped class weighting (cap 14.0, power 0.577), log-odds bias initialization.
  2. `2b_bigru_attn_v2` (Deep Corroborator): Bidirectional GRU with self-attention, spatial dropout (0.25), layer normalization, and gradient clipping (1.0).
  3. `1_tfidf_lr` (Linear Baseline): Word (1-2) and char (2-6) n-gram TF-IDF vectorizer + Logistic Regression.
  4. `detoxify` (External Upper-Bound Baseline): Unitary AI pretrained transformer trained on raw 2018 Kaggle competition split.

---

## 2. Quickstart & Running Instructions (Windows PowerShell)

All commands are designed to be run from the repository root: `c:\Users\arsha\Documents\Welcome-to-Civilization-v2`.

### 🚀 One-Command / Double-Click Launcher (Recommended)

To start both the FastAPI backend and Vite frontend simultaneously:

- **PowerShell**:
  ```powershell
  .\start_app.ps1
  ```
- **Windows Explorer**:
  Double-click **`start_app.bat`** directly from File Explorer.

Both servers will launch in their own labeled console windows, and your browser will automatically open to `http://localhost:5173`. To stop both servers, run `.\stop_app.ps1`.

---

### Step 1: Environment Setup

The repository uses the existing Python virtual environment (`.venv`):

```powershell
# Verify Python and GPU availability
& ".\.venv\Scripts\python.exe" -c "import torch; print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Install backend dependencies
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-backend.txt
```

### Step 2: Policy Calibration (Validation Split Only)

To regenerate or verify the asymmetric decision policy without test set contamination:

```powershell
& ".\.venv\Scripts\python.exe" backend/scripts/calibrate_policy.py
```
This reads `df_val` (15,869 validation comments) and calibrates `backend/policy.yaml` with asymmetric `remove_cut` values where validation precision $\ge 90\%$ with $\ge 10$ positives. High-sparsity classes (`threat`, `severe_toxic`) are safely capped at `null` ("Hide with warning").

### Step 3: Run Backend Server

```powershell
# Start FastAPI backend with hot reload on port 8000
& ".\.venv\Scripts\python.exe" -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Base URL: `http://localhost:8000`
- Interactive OpenAPI Docs: `http://localhost:8000/docs`
- Health Endpoint: `http://localhost:8000/api/health`

### Step 4: Run Frontend Development Server

Open a new PowerShell terminal:

```powershell
cd frontend
npm run dev
```
- Web Application: `http://localhost:5173`
- Vite automatically proxies `/api` and `/static` to `http://127.0.0.1:8000`.

### Step 5: (Optional) Local MongoDB via Docker Compose

If running without MongoDB Atlas, start a local MongoDB container:

```powershell
cd backend
docker compose up -d
```
Update `backend/.env` with `MONGODB_URI=mongodb://localhost:27017`.

---

## 3. Moderation Policy Specification

### Four-Tier Action Taxonomy
1. **Red ("Remove")**: Automated deletion. Requires high-precision validation cut ($\ge 90\%$ precision) **AND** corroboration from $\ge 2$ models.
2. **Orange ("Hide with warning")**: Content collapsed behind a click-to-reveal disclaimer. Triggered when any label exceeds its decision threshold, or for high-risk uncorroborated flags.
3. **Yellow ("Flag for review")**: Dispatched to human moderation queue (`needs_review: true`). Triggered when normalized scores fall into the Watch Zone $[0.30, 0.50)$.
4. **Green ("Allow")**: Clean content published directly.

### Normalization & Severity
- **Piecewise Linear Score Normalization**: Maps raw model threshold $t \to 0.50$ ($[0, t] \to [0, 0.5]$ and $[t, 1] \to [0.5, 1]$).
- **Composite Severity Score (0–10)**:
  $$\text{Severity} = 10 \times \min\left(1.0, \sum_i w_i \times \tilde{s}_i\right)$$
  - Critical Tier ($w=1.0$): `severe_toxic`, `threat`, `identity_hate`
  - Moderate Tier ($w=0.7$): `obscene`, `insult`
  - Base Tier ($w=0.5$): `toxic`
- **Bands**: Low ($[0, 3.0)$), Medium ($[3.0, 6.0)$), High ($[6.0, 8.5)$), Critical ($[8.5, 10.0]$).

### Profanity & Word Masking
- Case-insensitive tokenization with inflection stemming (`-ing`, `-ed`, `-s`).
- Leetspeak and wildcard vowel translation (`@` $\to$ a, `$` $\to$ s, `0` $\to$ o, `1` $\to$ i, `!` $\to$ i, `*` wildcard).
- Whitelist protection prevents false positives on benign words (`classic`, `pass`, `cocktail`, `document`).

---

## 4. API Endpoints Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Service uptime, GPU status, model availability, MongoDB connectivity |
| `GET` | `/api/models` | Metadata, devices, calibrated thresholds, and remove cuts for all 4 models |
| `POST` | `/api/predict` | Multi-model toxicity prediction, policy enforcement, censor, severity, and audit logging |
| `POST` | `/api/censor` | Standalone profanity detection and segmented word masking |
| `GET` | `/api/comments` | Paginated moderation audit history with action filtering |
| `GET` | `/api/comments/{id}` | Single comment audit record with full prediction breakdown |
| `DELETE` | `/api/comments/{id}` | Permanently delete a comment from MongoDB and in-memory history |
| `GET` | `/api/research/overview` | Executive cross-model test benchmark table |
| `GET` | `/api/research/eda` | Corpus profiling, label co-occurrence, and figure links |
| `GET` | `/api/research/splits` | 80/10/10 split manifest and MinHash LSH deduplication stats |
| `GET` | `/api/research/models` | Architecture specs, 5 Optuna trials, and TF-IDF blend analysis |
| `GET` | `/api/research/results` | Test split performance, per-label metrics, and 95% bootstrap CIs |
| `GET` | `/api/research/comparisons` | Paired bootstrap hypothesis tests vs Model 2b and Detoxify |
| `GET` | `/api/research/imbalance` | Class imbalance ratios and asymmetric policy derivation |
| `GET` | `/api/research/limitations` | 128-token truncation, single-seed serving, and Detoxify footnote |

---

## 5. Verification & Testing

### Run Backend Unit & Parity Tests
```powershell
& ".\.venv\Scripts\python.exe" -m pytest backend/tests/ -v
```
Verified results: **18 of 18 passed** (policy enforcement, severity calculation, profanity regex engine, and single-sample numerical parity against `probs_test.npy`).

### Build Frontend
```powershell
cd frontend
npm run build
```
Verified results: TypeScript check passed and production bundle compiled with **zero warnings/errors**.
