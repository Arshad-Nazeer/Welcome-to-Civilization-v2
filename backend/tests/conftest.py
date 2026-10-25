# conftest.py
import sys
import sentencepiece
import torch
import transformers

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))
