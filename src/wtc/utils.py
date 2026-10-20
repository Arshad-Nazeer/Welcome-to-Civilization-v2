
"""
src/wtc/utils.py
----------------
Shared utilities: logging, seeding, hashing, timing.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ #
#  Logging                                                             #
# ------------------------------------------------------------------ #

def get_logger(name: str, log_dir: Path | str | None = None) -> logging.Logger:
    """
    Return a logger that writes to both stdout and (optionally) a file.
    File is named <name>_<date>.log inside log_dir.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        from datetime import date
        log_file = log_dir / f"{name}_{date.today().isoformat()}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ------------------------------------------------------------------ #
#  Seeding                                                             #
# ------------------------------------------------------------------ #

def set_seed(seed: int) -> None:
    """Set global random seeds for Python, NumPy (and optionally PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ------------------------------------------------------------------ #
#  Hashing                                                             #
# ------------------------------------------------------------------ #

def sha256_file(path: Path | str) -> str:
    """Return hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ #
#  Timing                                                              #
# ------------------------------------------------------------------ #

class Timer:
    """Simple wall-clock timer context manager."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start

    def __str__(self) -> str:
        if hasattr(self, "elapsed"):
            return f"{self.elapsed:.2f}s"
        return f"{time.perf_counter() - self._start:.2f}s"
