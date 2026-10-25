"""
backend/app/services/policy.py
------------------------------
Imbalance-aware moderation policy engine driven by backend/policy.yaml.
Handles piecewise linear normalization, hierarchical severe_toxic propagation,
asymmetric critical label remove cuts, multi-model corroboration, watch zones,
and human-readable rule reasoning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import yaml

from ..config import settings
from .severity import calculate_severity

logger = logging.getLogger("wtc.policy")

CRITICAL_LABELS = ["threat", "identity_hate", "severe_toxic"]
MODERATE_LABELS = ["insult", "obscene"]
BASE_LABELS = ["toxic"]
ALL_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

_POLICY_CACHE: dict[str, Any] | None = None


def load_policy_config() -> dict[str, Any]:
    """Load policy configuration from backend/policy.yaml."""
    global _POLICY_CACHE
    if _POLICY_CACHE is not None:
        return _POLICY_CACHE

    p_file = settings.POLICY_PATH
    if not p_file.exists():
        logger.warning("policy.yaml not found at %s. Using default policy.", p_file)
        return {
            "policy_version": "1.0.0-fallback",
            "watch_zone": {"min_normalized": 0.30, "max_normalized": 0.50},
            "models": {},
        }

    try:
        with open(p_file, "r", encoding="utf-8") as f:
            _POLICY_CACHE = yaml.safe_load(f)
            return _POLICY_CACHE
    except Exception as exc:
        logger.error("Failed to parse policy.yaml: %s. Using defaults.", exc)
        return {
            "policy_version": "1.0.0-fallback",
            "watch_zone": {"min_normalized": 0.30, "max_normalized": 0.50},
            "models": {},
        }


def normalize_probability(p: float, threshold: float) -> float:
    """
    Piecewise linear normalization:
      - [0, threshold] -> [0, 0.5]
      - [threshold, 1] -> [0.5, 1.0]
    Ensures that score >= 0.5 strictly corresponds to probability >= model threshold.
    """
    p = float(max(0.0, min(1.0, p)))
    t = float(max(0.001, min(0.999, threshold)))

    if p <= t:
        n = 0.5 * (p / t)
    else:
        n = 0.5 + 0.5 * ((p - t) / (1.0 - t))

    return round(float(max(0.0, min(1.0, n))), 4)


def evaluate_model_decision(
    model_id: str,
    raw_probs: dict[str, float],
    all_models_raw_probs: dict[str, dict[str, float]],
    policy_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate action, reasoning, and severity for a single model according to policy rules.
    """
    model_policy = policy_cfg.get("models", {}).get(model_id, {})
    thresholds = model_policy.get("thresholds", {l: 0.5 for l in ALL_LABELS})
    remove_cuts = model_policy.get("remove_cuts", {})

    # 1. Normalize probabilities
    norm_scores = {
        lbl: normalize_probability(raw_probs.get(lbl, 0.0), thresholds.get(lbl, 0.5))
        for lbl in ALL_LABELS
    }

    # 2. Hierarchy rule: if severe_toxic fires (n >= 0.5), toxic must also fire
    if norm_scores.get("severe_toxic", 0.0) >= 0.5:
        if norm_scores.get("toxic", 0.0) < 0.5:
            norm_scores["toxic"] = 0.50

    per_label_flags = {lbl: (norm_scores[lbl] >= 0.5) for lbl in ALL_LABELS}

    # Re-apply hierarchy flag
    if per_label_flags.get("severe_toxic", False):
        per_label_flags["toxic"] = True

    # 3. Severity Score & Band
    severity = calculate_severity(norm_scores)

    # 4. Check Action Rules (Top-to-Bottom, First Match Wins)
    action: str = "Allow"
    action_color: str = "green"
    needs_review: bool = False
    reasoning: list[str] = []

    # Rule A: Check RED ("Remove")
    # Critical label has p >= remove_cut (where remove_cut is not null)
    # AND at least one OTHER available model also fires that label (n >= 0.5).
    fired_red_label: str | None = None
    red_corroborator: str | None = None

    for crit in CRITICAL_LABELS:
        cut = remove_cuts.get(crit)
        p_val = raw_probs.get(crit, 0.0)

        if cut is not None and p_val >= cut:
            # Check corroboration across other models
            for other_id, other_probs in all_models_raw_probs.items():
                if other_id == model_id:
                    continue
                other_t = policy_cfg.get("models", {}).get(other_id, {}).get("thresholds", {}).get(crit, 0.5)
                other_n = normalize_probability(other_probs.get(crit, 0.0), other_t)
                if other_n >= 0.5:
                    fired_red_label = crit
                    red_corroborator = other_id
                    break
            if fired_red_label:
                break

    if fired_red_label and red_corroborator:
        cut_val = remove_cuts[fired_red_label]
        p_val = raw_probs[fired_red_label]
        other_n = normalize_probability(
            all_models_raw_probs[red_corroborator].get(fired_red_label, 0.0),
            policy_cfg.get("models", {}).get(red_corroborator, {}).get("thresholds", {}).get(fired_red_label, 0.5),
        )
        action = "Remove"
        action_color = "red"
        needs_review = False
        reasoning.append(
            f"Critical tier label '{fired_red_label}' probability (p={p_val:.3f}) >= calibrated remove cut ({cut_val:.3f}, precision >= 90% on validation) and corroborated by {red_corroborator} (normalized score n={other_n:.2f} >= 0.50) -> Action: Remove."
        )
        reasoning.append(
            "Imbalance Handling Applied: High precision remove cut prevents false deletion of rare category comments, and multi-model corroboration eliminates single-model hallucination."
        )

    # Rule B: Check ORANGE ("Hide with warning", needs_review = True)
    # 1. Any critical label fires (n >= 0.5) but Red was not satisfied;
    # 2. OR a moderate/base label has n >= 0.75;
    # 3. OR 3+ labels fire (n >= 0.5).
    if action == "Allow":
        critical_fired = [c for c in CRITICAL_LABELS if per_label_flags.get(c, False)]
        high_moderate_base = [l for l in (MODERATE_LABELS + BASE_LABELS) if norm_scores.get(l, 0.0) >= 0.75]
        total_fired = [l for l, f in per_label_flags.items() if f]

        if critical_fired:
            action = "Hide with warning"
            action_color = "orange"
            needs_review = True
            cut_val = remove_cuts.get(critical_fired[0])
            if cut_val is None:
                cut_explanation = "validation precision never reliably achieved >= 90% to justify automated deletion"
            else:
                cut_explanation = f"requires p >= {cut_val:.3f} and corroboration from a second model"
            reasoning.append(
                f"Critical label '{critical_fired[0]}' fired (normalized score n={norm_scores[critical_fired[0]]:.2f} >= 0.50), but Red criteria not met ({cut_explanation}) -> Action: Hide with warning."
            )
            reasoning.append(
                "Imbalance Handling Applied: Extreme class sparsity warrants cautionary hiding without permanent deletion until human review."
            )
        elif high_moderate_base:
            action = "Hide with warning"
            action_color = "orange"
            needs_review = True
            reasoning.append(
                f"High-confidence toxicity detected in '{high_moderate_base[0]}' (normalized score n={norm_scores[high_moderate_base[0]]:.2f} >= 0.75) -> Action: Hide with warning."
            )
        elif len(total_fired) >= 3:
            action = "Hide with warning"
            action_color = "orange"
            needs_review = True
            reasoning.append(
                f"Multi-label compound toxicity: 3+ categories fired concurrently ({', '.join(total_fired)}) -> Action: Hide with warning."
            )

    # Rule C: Check YELLOW ("Flag for review")
    # 1. Any label fires with n < 0.75 (non-critical);
    # 2. OR a critical label is in the watch zone (0.30 <= n < 0.50);
    # 3. OR models strongly disagree on a label.
    if action == "Allow":
        fired_non_critical = [l for l in (MODERATE_LABELS + BASE_LABELS) if per_label_flags.get(l, False) and norm_scores.get(l, 0.0) < 0.75]
        watch_zone_labels = [c for c in CRITICAL_LABELS if 0.30 <= norm_scores.get(c, 0.0) < 0.50]

        # Check strong disagreement against other models (e.g. difference >= 0.40 in normalized score)
        disagreement_labels: list[str] = []
        for l in ALL_LABELS:
            my_n = norm_scores[l]
            for other_id, other_probs in all_models_raw_probs.items():
                if other_id == model_id:
                    continue
                other_t = policy_cfg.get("models", {}).get(other_id, {}).get("thresholds", {}).get(l, 0.5)
                other_n = normalize_probability(other_probs.get(l, 0.0), other_t)
                if abs(my_n - other_n) >= 0.45:
                    disagreement_labels.append(f"{l} (this model n={my_n:.2f} vs {other_id} n={other_n:.2f})")
                    break

        if fired_non_critical:
            action = "Flag for review"
            action_color = "yellow"
            needs_review = True
            reasoning.append(
                f"Toxicity label '{fired_non_critical[0]}' triggered near calibrated threshold (normalized score n={norm_scores[fired_non_critical[0]]:.2f} in [0.50, 0.75)) -> Action: Flag for review."
            )
        elif watch_zone_labels:
            action = "Flag for review"
            action_color = "yellow"
            needs_review = True
            reasoning.append(
                f"Critical label '{watch_zone_labels[0]}' is in the sensitive watch zone (normalized score n={norm_scores[watch_zone_labels[0]]:.2f} in [0.30, 0.50)) -> Action: Flag for review."
            )
            reasoning.append(
                "Imbalance Handling Applied: Lower boundary watch zone ensures rare threats and identity attacks near decision margin are flagged for human oversight."
            )
        elif disagreement_labels:
            action = "Flag for review"
            action_color = "yellow"
            needs_review = True
            reasoning.append(
                f"Inter-model disagreement detected across architectures: {disagreement_labels[0]} -> Action: Flag for review."
            )

    # Rule D: GREEN ("Allow")
    if action == "Allow":
        reasoning.append("All category probabilities remain below calibrated decision thresholds -> Action: Allow.")

    return {
        "model_id": model_id,
        "probabilities": raw_probs,
        "normalized_scores": norm_scores,
        "per_label_flags": per_label_flags,
        "severity": severity,
        "action": action,
        "action_color": action_color,
        "needs_review": needs_review,
        "reasoning": reasoning,
    }


def evaluate_all_models_decision(
    all_models_raw_probs: dict[str, dict[str, float]],
    available_statuses: dict[str, bool],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Evaluates policy decisions for all models simultaneously,
    then derives the official headline recommendation from Model 3 (DistilRoBERTa).
    """
    policy_cfg = load_policy_config()
    model_evaluations: dict[str, Any] = {}

    for model_id, raw_p in all_models_raw_probs.items():
        if not available_statuses.get(model_id, True):
            continue
        eval_res = evaluate_model_decision(
            model_id=model_id,
            raw_probs=raw_p,
            all_models_raw_probs=all_models_raw_probs,
            policy_cfg=policy_cfg,
        )
        model_evaluations[model_id] = eval_res

    # Select Headline recommendation: Model 3 first, then 2b, then 1, then detoxify
    headline_id = "3_distilroberta"
    if headline_id not in model_evaluations:
        for fallback_id in ["2b_bigru_attn_v2", "1_tfidf_lr", "detoxify"]:
            if fallback_id in model_evaluations:
                headline_id = fallback_id
                break

    headline_model_eval = model_evaluations.get(headline_id, {})
    headline = {
        "model_id": headline_id,
        "model_name": "Model 3: DistilRoBERTa" if headline_id == "3_distilroberta" else headline_id,
        "action": headline_model_eval.get("action", "Allow"),
        "action_color": headline_model_eval.get("action_color", "green"),
        "severity": headline_model_eval.get("severity", {"score": 0.0, "band": "Low"}),
        "needs_review": headline_model_eval.get("needs_review", False),
        "reasoning": headline_model_eval.get("reasoning", ["All probabilities below threshold."]),
    }

    return model_evaluations, headline
