
#!/usr/bin/env bash
# run_all.sh — Run the full pipeline on Linux/Mac/Colab or Windows Git Bash.
# Usage: bash run_all.sh
# Override python: PYTHON=/path/to/python bash run_all.sh
set -e

# Detect virtual environment python (cross-platform)
if [ -z "$PYTHON" ]; then
  if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
    PIP=".venv/bin/pip"
    PYTEST=".venv/bin/pytest"
  elif [ -f ".venv/Scripts/python" ]; then
    PYTHON=".venv/Scripts/python"
    PIP=".venv/Scripts/pip"
    PYTEST=".venv/Scripts/pytest"
  else
    PYTHON="python"
    PIP="pip"
    PYTEST="pytest"
  fi
fi

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
$PYTEST tests/ -v

echo "=== ALL DONE ==="
