"""Validation: good data passes, bad data fails loudly."""

import pandas as pd
import pytest

from src.data_validation import DataValidationError, validate_dataset


def valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": ["ham", "spam", "ham"],
            "message": ["see you later", "win cash now", "ok thanks"],
        }
    )


def test_valid_dataset_passes():
    report = validate_dataset(valid_df())
    assert report.n_rows == 3
    assert report.class_counts == {"ham": 2, "spam": 1}
    assert report.spam_percentage == pytest.approx(33.33, abs=0.01)
    assert report.n_duplicate_rows == 0


def test_empty_dataset_fails():
    with pytest.raises(DataValidationError, match="empty"):
        validate_dataset(pd.DataFrame(columns=["label", "message"]))


def test_none_dataset_fails():
    with pytest.raises(DataValidationError):
        validate_dataset(None)


def test_missing_target_column_fails():
    df = pd.DataFrame({"message": ["hello"]})
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_dataset(df)


def test_missing_message_column_fails():
    df = pd.DataFrame({"label": ["ham"]})
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_dataset(df)


def test_invalid_labels_fail():
    df = pd.DataFrame({"label": ["ham", "junk"], "message": ["a", "b"]})
    with pytest.raises(DataValidationError, match="Unexpected labels"):
        validate_dataset(df)


def test_missing_labels_fail():
    df = pd.DataFrame({"label": ["ham", None], "message": ["a", "b"]})
    with pytest.raises(DataValidationError, match="missing labels"):
        validate_dataset(df)


def test_missing_messages_fail():
    df = pd.DataFrame({"label": ["ham", "spam"], "message": ["a", None]})
    with pytest.raises(DataValidationError, match="missing .* messages"):
        validate_dataset(df)


def test_duplicates_are_reported_not_fatal():
    df = pd.DataFrame(
        {
            "label": ["ham", "ham", "spam"],
            "message": ["same text", "same text", "win now"],
        }
    )
    report = validate_dataset(df)
    assert report.n_duplicate_rows == 1
    assert any("duplicate" in w for w in report.warnings)
