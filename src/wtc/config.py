
"""
src/wtc/config.py
-----------------
Load and expose the central YAML config.
All paths are resolved relative to the project root, so the same config
works locally and in Google Colab.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _find_project_root() -> Path:
    """Walk up from this file until we find configs/config.yaml."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "configs" / "config.yaml").exists():
            return parent
    # Fallback: assume cwd is project root (Colab use-case)
    return Path(os.getcwd())


PROJECT_ROOT: Path = _find_project_root()


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Return the parsed YAML config dict."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def resolve_path(relative: str) -> Path:
    """Resolve a config-relative path to an absolute Path."""
    return PROJECT_ROOT / relative


# Module-level convenience
CFG: dict[str, Any] = load_config()
LABELS: list[str] = CFG["labels"]
SEED: int = CFG["seed"]
TEXT_COL: str = CFG["text_column"]
ID_COL: str = CFG["id_column"]
