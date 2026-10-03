"""Text cleaning: lossless normalization only, spam signals preserved."""

import pandas as pd

from src.text_cleaning import clean_dataset, clean_text


def test_lowercasing():
    assert clean_text("FREE ENTRY NOW") == "free entry now"


def test_whitespace_normalization():
    assert clean_text("hello \n\t  world   again") == "hello world again"


def test_empty_and_none_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   \n  ") == ""


def test_unicode_normalization_fullwidth():
    # full-width "ＦＲＥＥ" folds to ASCII "free" via NFKC
    assert clean_text("ＦＲＥＥ") == "free"


def test_spam_signals_are_preserved():
    text = clean_text("WIN £1000! Call 09061701461 or visit HTTP://Win.CO now!!")
    assert "£1000" in text
    assert "09061701461" in text
    assert "http://win.co" in text
    assert "!!" in text


def test_non_string_input_is_coerced():
    assert clean_text(12345) == "12345"


def test_clean_dataset_drops_duplicates_and_empties():
    df = pd.DataFrame(
        {
            "label": ["ham", "ham", "spam", "spam"],
            "message": ["same text", "same text", "  ", "win cash"],
        }
    )
    cleaned = clean_dataset(df)
    assert len(cleaned) == 2  # 1 duplicate + 1 empty removed
    assert cleaned["message"].tolist() == ["same text", "win cash"]
