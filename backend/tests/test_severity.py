import pytest
from app.services.severity import calculate_severity


def test_severity_all_zeros():
    res = calculate_severity({
        "toxic": 0.0,
        "severe_toxic": 0.0,
        "obscene": 0.0,
        "threat": 0.0,
        "insult": 0.0,
        "identity_hate": 0.0,
    })
    assert res["score"] == 0.0
    assert res["band"] == "Low"


def test_severity_tier_weights():
    # Critical label at n=1.0 gives 10.0
    res_crit = calculate_severity({"threat": 1.0, "toxic": 0.0})
    assert res_crit["score"] == 10.0
    assert res_crit["band"] == "Critical"

    # Moderate label at n=1.0 gives 7.0 (High)
    res_mod = calculate_severity({"obscene": 1.0, "toxic": 0.0})
    assert res_mod["score"] == 7.0
    assert res_mod["band"] == "High"

    # Base label at n=1.0 gives 5.0 (Medium)
    res_base = calculate_severity({"toxic": 1.0})
    assert res_base["score"] == 5.0
    assert res_base["band"] == "Medium"


def test_severity_bands():
    # Band Low: score < 3.0
    assert calculate_severity({"toxic": 0.4})["band"] == "Low"  # 10 * 0.5 * 0.4 = 2.0

    # Band Medium: 3.0 <= score < 6.0
    assert calculate_severity({"toxic": 0.8})["band"] == "Medium"  # 10 * 0.5 * 0.8 = 4.0

    # Band High: 6.0 <= score < 8.5
    assert calculate_severity({"threat": 0.7})["band"] == "High"  # 10 * 1.0 * 0.7 = 7.0

    # Band Critical: >= 8.5
    assert calculate_severity({"identity_hate": 0.95})["band"] == "Critical"  # 10 * 1.0 * 0.95 = 9.5
