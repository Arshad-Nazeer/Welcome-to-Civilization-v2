"""
backend/app/services/censor.py
------------------------------
Lexicon-based profanity censoring engine for Welcome to Civilization.
- Independent of ML models and moderation policy decisions.
- Built from backend/data/curse_words.txt and configs/config.yaml.
- Handles case-insensitive word-boundary matching, common inflections (-s, -ed, -ing, -er),
  and common leetspeak substitutions (@ -> a, $ -> s, 0 -> o, 1/! -> i, * -> vowel).
- Avoids false matches inside harmless words (e.g., classic, password, cocktail, document).
- Returns structured segments [{text: str, censored: bool}] and masked strings.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any
import yaml

from ..config import settings

logger = logging.getLogger("wtc.censor")

# Benign whitelist to prevent false positives
WHITELIST_WORDS = {
    "classic", "classical", "class", "classes", "pass", "passage", "passport", "passed", "passing",
    "compass", "compassion", "harass", "harassment", "bass", "glass", "glasses", "mass", "massive",
    "grass", "brass", "cassette", "embassy", "assassin", "assassinate", "asset", "assets", "assert",
    "assertion", "assist", "assistance", "assistant", "associate", "association", "assume", "assumption",
    "assure", "assurance", "document", "cocktail", "cockpit", "cockatoo", "cockatoos", "peacock", "peacocks",
    "shuttlecock", "woodcock", "hitchcock", "therapist", "psychotherapist", "basement", "scunthorpe",
}

_CHAR_LEET_REGEX = {
    "a": r"[a@4\*\^]",
    "b": r"[b8]",
    "c": r"[c\(\<]",
    "e": r"[e3\*]",
    "i": r"[i1!\|\*]",
    "l": r"[l1\|]",
    "o": r"[o0\*\(\)]",
    "s": r"[s\$5]",
    "t": r"[t\+7]",
    "u": r"[u\*v]",
}

_LEXICON_SET: set[str] | None = None
_COMPILED_REGEX: re.Pattern | None = None


def load_lexicon() -> set[str]:
    """Load base profanity terms from curse_words.txt and configs/config.yaml."""
    global _LEXICON_SET
    if _LEXICON_SET is not None:
        return _LEXICON_SET

    words = set()

    # 1. From curse_words.txt
    lexicon_file = settings.CURSE_WORDS_PATH
    if lexicon_file.exists():
        with open(lexicon_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().lower()
                if line and not line.startswith("#"):
                    words.add(line)

    # 2. From configs/config.yaml profanity_terms
    cfg_file = settings.REPO_ROOT_PATH / "configs" / "config.yaml"
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                terms = data.get("profanity_terms", [])
                for t in terms:
                    if isinstance(t, str):
                        words.add(t.strip().lower())
        except Exception as e:
            logger.warning("Could not read profanity_terms from config.yaml: %s", e)

    _LEXICON_SET = words
    logger.info("Loaded profanity lexicon: %d root terms.", len(words))
    return _LEXICON_SET


def _build_regex_for_word(root: str) -> str:
    """Build leetspeak regex pattern for a single root word with inflection support."""
    char_patterns = []
    for char in root:
        char_patterns.append(_CHAR_LEET_REGEX.get(char, re.escape(char)))
    root_pattern = "".join(char_patterns)
    # Inflections: -s, -es, -ed, -ing, -er, -ers, -y, -ie, -ies, -head, -heads, -hole, -holes
    inflections = r"(?:s|es|ed|ing|er|ers|y|ie|ies|head|heads|hole|holes)?"
    return rf"{root_pattern}{inflections}"


def get_compiled_regex() -> re.Pattern:
    """Compile combined word-boundary regex for all lexicon roots."""
    global _COMPILED_REGEX
    if _COMPILED_REGEX is not None:
        return _COMPILED_REGEX

    lexicon = load_lexicon()
    # Sort by length descending to match longer compounds first (e.g. asshole before ass)
    sorted_roots = sorted(lexicon, key=len, reverse=True)
    patterns = [_build_regex_for_word(r) for r in sorted_roots if r]

    combined = r"(?i)\b(?:" + "|".join(patterns) + r")\b"
    _COMPILED_REGEX = re.compile(combined)
    return _COMPILED_REGEX


def censor_text(text: str) -> dict[str, Any]:
    """
    Process input text into segments and mask profane words.
    Returns:
      {
        "original": str,
        "masked": str,
        "has_profanity": bool,
        "segments": list[{"text": str, "censored": bool}]
      }
    """
    if not text:
        return {
            "original": "",
            "masked": "",
            "has_profanity": False,
            "segments": [],
        }

    pattern = get_compiled_regex()
    segments: list[dict[str, Any]] = []
    last_idx = 0
    has_profanity = False
    masked_chars = list(text)

    for match in pattern.finditer(text):
        start, end = match.span()
        matched_str = match.group()

        # Check against whitelist
        clean_norm = re.sub(r"[^\w]", "", matched_str.lower())
        if clean_norm in WHITELIST_WORDS:
            continue

        # Non-profane text before match
        if start > last_idx:
            segments.append({"text": text[last_idx:start], "censored": False})

        # Profane segment
        has_profanity = True
        segments.append({"text": matched_str, "censored": True})

        # Mask characters
        for i in range(start, end):
            masked_chars[i] = "*"

        last_idx = end

    # Trailing clean text
    if last_idx < len(text):
        segments.append({"text": text[last_idx:], "censored": False})

    return {
        "original": text,
        "masked": "".join(masked_chars),
        "has_profanity": has_profanity,
        "segments": segments,
    }
