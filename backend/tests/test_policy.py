import pytest
from app.services.policy import (
    normalize_probability,
    evaluate_model_decision,
    evaluate_all_models_decision,
)


def test_normalize_probability():
    t = 0.66
    # Below threshold
    assert normalize_probability(0.0, t) == 0.0
    assert normalize_probability(0.33, t) == 0.25
    # At threshold: must map exactly to 0.5
    assert normalize_probability(t, t) == 0.5
    # Above threshold
    assert normalize_probability(1.0, t) == 1.0
    assert normalize_probability(0.83, t) == 0.75


def test_hierarchy_severe_toxic_forces_toxic():
    # Model where severe_toxic fires (p=0.8 > threshold 0.71 -> n > 0.5)
    # but toxic probability is low (p=0.1 < threshold 0.66)
    policy_cfg = {
        "models": {
            "3_distilroberta": {
                "thresholds": {"toxic": 0.66, "severe_toxic": 0.71, "obscene": 0.65, "threat": 0.68, "insult": 0.71, "identity_hate": 0.81},
                "remove_cuts": {"threat": None, "identity_hate": 0.90, "severe_toxic": None},
            }
        }
    }
    raw_p = {
        "toxic": 0.10,
        "severe_toxic": 0.80,
        "obscene": 0.05,
        "threat": 0.01,
        "insult": 0.05,
        "identity_hate": 0.01,
    }
    res = evaluate_model_decision("3_distilroberta", raw_p, {"3_distilroberta": raw_p}, policy_cfg)
    # severe_toxic fired
    assert res["per_label_flags"]["severe_toxic"] is True
    # toxic MUST be treated as fired due to hierarchy
    assert res["per_label_flags"]["toxic"] is True
    assert res["normalized_scores"]["toxic"] >= 0.5


def test_null_remove_cut_cannot_trigger_red():
    # severe_toxic has remove_cut: None
    policy_cfg = {
        "models": {
            "3_distilroberta": {
                "thresholds": {"toxic": 0.66, "severe_toxic": 0.71, "obscene": 0.65, "threat": 0.68, "insult": 0.71, "identity_hate": 0.81},
                "remove_cuts": {"threat": None, "identity_hate": 0.90, "severe_toxic": None},
            },
            "2b_bigru_attn_v2": {
                "thresholds": {"toxic": 0.71, "severe_toxic": 0.76, "obscene": 0.73, "threat": 0.90, "insult": 0.79, "identity_hate": 0.77},
                "remove_cuts": {"threat": None, "identity_hate": 0.99, "severe_toxic": None},
            }
        }
    }
    raw_p1 = {"toxic": 0.99, "severe_toxic": 0.99, "obscene": 0.0, "threat": 0.0, "insult": 0.0, "identity_hate": 0.0}
    raw_p2 = {"toxic": 0.99, "severe_toxic": 0.99, "obscene": 0.0, "threat": 0.0, "insult": 0.0, "identity_hate": 0.0}
    all_p = {"3_distilroberta": raw_p1, "2b_bigru_attn_v2": raw_p2}

    res = evaluate_model_decision("3_distilroberta", raw_p1, all_p, policy_cfg)
    # Even though both models output 0.99 on severe_toxic, remove_cut is None, so it can NEVER trigger Red!
    # Max action is Orange ("Hide with warning")
    assert res["action"] == "Hide with warning"
    assert res["action_color"] == "orange"
    assert res["needs_review"] is True


def test_red_remove_with_corroboration():
    # identity_hate has remove_cut = 0.90 for Model 3
    policy_cfg = {
        "models": {
            "3_distilroberta": {
                "thresholds": {"toxic": 0.66, "severe_toxic": 0.71, "obscene": 0.65, "threat": 0.68, "insult": 0.71, "identity_hate": 0.81},
                "remove_cuts": {"threat": None, "identity_hate": 0.90, "severe_toxic": None},
            },
            "1_tfidf_lr": {
                "thresholds": {"toxic": 0.49, "severe_toxic": 0.51, "obscene": 0.46, "threat": 0.42, "insult": 0.58, "identity_hate": 0.49},
                "remove_cuts": {"threat": None, "identity_hate": 0.975, "severe_toxic": None},
            }
        }
    }
    # Model 3: p=0.95 >= remove_cut 0.90
    m3_p = {"toxic": 0.8, "severe_toxic": 0.0, "obscene": 0.0, "threat": 0.0, "insult": 0.0, "identity_hate": 0.95}
    # Model 1 corroborates: p=0.70 >= threshold 0.49 (n >= 0.50)
    m1_p = {"toxic": 0.6, "severe_toxic": 0.0, "obscene": 0.0, "threat": 0.0, "insult": 0.0, "identity_hate": 0.70}
    all_p = {"3_distilroberta": m3_p, "1_tfidf_lr": m1_p}

    res = evaluate_model_decision("3_distilroberta", m3_p, all_p, policy_cfg)
    assert res["action"] == "Remove"
    assert res["action_color"] == "red"
    assert any("corroborated" in r for r in res["reasoning"])


def test_red_downgraded_without_corroboration():
    # Model 3 has p=0.95 >= remove_cut 0.90 on identity_hate,
    # BUT Model 1 does NOT fire (p=0.05 < 0.49)
    policy_cfg = {
        "models": {
            "3_distilroberta": {
                "thresholds": {"toxic": 0.66, "severe_toxic": 0.71, "obscene": 0.65, "threat": 0.68, "insult": 0.71, "identity_hate": 0.81},
                "remove_cuts": {"threat": None, "identity_hate": 0.90, "severe_toxic": None},
            },
            "1_tfidf_lr": {
                "thresholds": {"toxic": 0.49, "severe_toxic": 0.51, "obscene": 0.46, "threat": 0.42, "insult": 0.58, "identity_hate": 0.49},
                "remove_cuts": {"threat": None, "identity_hate": 0.975, "severe_toxic": None},
            }
        }
    }
    m3_p = {"toxic": 0.8, "severe_toxic": 0.0, "obscene": 0.0, "threat": 0.0, "insult": 0.0, "identity_hate": 0.95}
    m1_p = {"toxic": 0.1, "severe_toxic": 0.0, "obscene": 0.0, "threat": 0.0, "insult": 0.0, "identity_hate": 0.05}
    all_p = {"3_distilroberta": m3_p, "1_tfidf_lr": m1_p}

    res = evaluate_model_decision("3_distilroberta", m3_p, all_p, policy_cfg)
    # Lacks corroboration -> must downgrade to Orange ("Hide with warning")
    assert res["action"] == "Hide with warning"
    assert res["action_color"] == "orange"
    assert res["needs_review"] is True


def test_watch_zone_triggers_yellow():
    # Critical label in [0.30, 0.50) normalized
    policy_cfg = {
        "models": {
            "3_distilroberta": {
                "thresholds": {"toxic": 0.66, "severe_toxic": 0.71, "obscene": 0.65, "threat": 0.68, "insult": 0.71, "identity_hate": 0.81},
                "remove_cuts": {"threat": None, "identity_hate": 0.90, "severe_toxic": None},
            }
        }
    }
    # For threat (t=0.68): p=0.48 gives n = 0.5 * (0.48 / 0.68) = 0.353 (inside [0.30, 0.50))
    raw_p = {"toxic": 0.05, "severe_toxic": 0.0, "obscene": 0.0, "threat": 0.48, "insult": 0.0, "identity_hate": 0.0}
    res = evaluate_model_decision("3_distilroberta", raw_p, {"3_distilroberta": raw_p}, policy_cfg)
    assert res["action"] == "Flag for review"
    assert res["action_color"] == "yellow"
    assert res["needs_review"] is True
    assert any("watch zone" in r for r in res["reasoning"])


def test_allow_green_for_benign_text():
    policy_cfg = {
        "models": {
            "3_distilroberta": {
                "thresholds": {"toxic": 0.66, "severe_toxic": 0.71, "obscene": 0.65, "threat": 0.68, "insult": 0.71, "identity_hate": 0.81},
                "remove_cuts": {"threat": None, "identity_hate": 0.90, "severe_toxic": None},
            }
        }
    }
    raw_p = {"toxic": 0.01, "severe_toxic": 0.001, "obscene": 0.002, "threat": 0.001, "insult": 0.003, "identity_hate": 0.002}
    res = evaluate_model_decision("3_distilroberta", raw_p, {"3_distilroberta": raw_p}, policy_cfg)
    assert res["action"] == "Allow"
    assert res["action_color"] == "green"
    assert res["needs_review"] is False
