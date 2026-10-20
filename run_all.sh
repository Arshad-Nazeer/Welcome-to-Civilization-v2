
#!/usr/bin/env bash
# run_all.sh — Alternative to make for Windows without make installed
# Usage: bash run_all.sh
set -e

PYTHON=".venv/Scripts/python"
PIP=".venv/Scripts/pip"

echo "=== setup ==="
python -m venv .venv
$PIP install --upgrade pip
$PIP install -r requirements.txt

echo "=== download ==="
$PYTHON scripts/01_download_data.py

echo "=== preprocess ==="
$PYTHON scripts/02_preprocess.py

echo "=== eda ==="
$PYTHON scripts/03_eda.py

echo "=== report ==="
$PYTHON scripts/04_build_report.py

echo "=== test ==="
.venv/Scripts/pytest tests/ -v

echo "=== ALL DONE ==="
