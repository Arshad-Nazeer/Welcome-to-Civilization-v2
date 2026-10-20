
# ============================================================
#  Welcome to Civilization — Makefile
# ============================================================
# Cross-platform: detects .venv/bin (Linux/Mac) vs .venv/Scripts (Windows).
# Override: make PYTHON=/path/to/python preprocess

ifeq ($(OS),Windows_NT)
  VENV_PYTHON := .venv/Scripts/python
  VENV_PIP    := .venv/Scripts/pip
  VENV_PYTEST := .venv/Scripts/pytest
else
  VENV_PYTHON := .venv/bin/python
  VENV_PIP    := .venv/bin/pip
  VENV_PYTEST := .venv/bin/pytest
endif

PYTHON ?= $(VENV_PYTHON)
PIP    ?= $(VENV_PIP)
PYTEST ?= $(VENV_PYTEST)

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
	$(PYTEST) tests/ -v

clean:
	rm -rf data/processed/* data/splits/* reports/eda/figures/* \
	       reports/eda/tables/* reports/eda/*.json reports/eda/*.md \
	       logs/*.log
