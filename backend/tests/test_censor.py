import pytest
from app.services.censor import censor_text


def test_clean_text_no_profanity():
    text = "Hello, this is a wonderful and civilized conversation."
    res = censor_text(text)
    assert res["has_profanity"] is False
    assert res["masked"] == text
    assert all(seg["censored"] is False for seg in res["segments"])


def test_case_insensitive_profanity():
    res1 = censor_text("You are a BITCH and a Shit!")
    assert res1["has_profanity"] is True
    assert "*****" in res1["masked"]
    assert "****" in res1["masked"]

    censored_words = [s["text"] for s in res1["segments"] if s["censored"]]
    assert "BITCH" in censored_words
    assert "Shit" in censored_words


def test_inflections():
    text = "Stop fucking talking to those assholes and bitches."
    res = censor_text(text)
    assert res["has_profanity"] is True
    censored_words = [s["text"].lower() for s in res["segments"] if s["censored"]]
    assert "fucking" in censored_words
    assert "assholes" in censored_words
    assert "bitches" in censored_words


def test_leetspeak_and_masking():
    text = "What the f*ck, you b!tch and a$$hole."
    res = censor_text(text)
    assert res["has_profanity"] is True
    censored_words = [s["text"] for s in res["segments"] if s["censored"]]
    assert "f*ck" in censored_words
    assert "b!tch" in censored_words
    assert "a$$hole" in censored_words


def test_no_false_positives_on_benign_words():
    # Words containing substrings like 'ass' or 'cock' must NOT be censored
    benign = "We had a classic cocktail while reviewing the passport document in the basement with our therapist."
    res = censor_text(benign)
    assert res["has_profanity"] is False
    assert res["masked"] == benign
    assert all(seg["censored"] is False for seg in res["segments"])


def test_segments_reconstruction():
    text = "Hey fuck you, you jerk!"
    res = censor_text(text)
    # Reconstructed text from segments must match original text exactly
    reconstructed = "".join(seg["text"] for seg in res["segments"])
    assert reconstructed == text
