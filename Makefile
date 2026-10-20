
# ============================================================
#  Welcome to Civilization — Makefile
# ============================================================

PYTHON := .venv/Scripts/python
PIP    := .venv/Scripts/pip

.PHONY: all setup download preprocess eda report test clean

all: setup download preprocess eda report test

setup:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

download:
	$(PYTHON) scripts/01_download_data.py

preprocess:
	$(PYTHON) scripts/02_preprocess.py

eda:
	$(PYTHON) scripts/03_eda.py

report:
	$(PYTHON) scripts/04_build_report.py

test:
	.venv/Scripts/pytest tests/ -v

clean:
	rm -rf data/processed/* data/splits/* reports/eda/figures/* \
	       reports/eda/tables/* reports/eda/*.json reports/eda/*.md \
	       logs/*.log
