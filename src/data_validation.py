"""Stage 2 - data validation.

Essential problems (empty dataset, missing columns, missing/invalid labels,
missing messages) raise DataValidationError so the pipeline FAILS LOUDLY.
Non-essential issues (duplicates, empty messages) are counted, logged and
reported - they are handled in the cleaning stage, never silently ignored.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

import config

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when the dataset fails an essential validation check."""


@dataclass
class ValidationReport:
    n_rows: int
    n_columns: int
    n_duplicate_rows: int
    n_empty_messages: int
    class_counts: dict
    warnings: list = field(default_factory=list)

    @property
    def spam_percentage(self) -> float:
        return round(self.class_counts.get("spam", 0) / self.n_rows * 100, 2)


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    """Validate the raw dataset. Raises DataValidationError on hard failures."""
    if df is None or df.empty:
        raise DataValidationError("Dataset is empty.")

    missing_cols = {config.LABEL_COLUMN, config.MESSAGE_COLUMN} - set(df.columns)
    if missing_cols:
        raise DataValidationError(
            f"Missing required columns: {sorted(missing_cols)}. "
            f"Found: {list(df.columns)}"
        )

    if df[config.LABEL_COLUMN].isna().any():
        raise DataValidationError(
            f"Found {int(df[config.LABEL_COLUMN].isna().sum())} missing labels."
        )

    unexpected = set(df[config.LABEL_COLUMN].unique()) - set(config.EXPECTED_LABELS)
    if unexpected:
        raise DataValidationError(
            f"Unexpected labels {sorted(unexpected)}; expected {config.EXPECTED_LABELS}."
        )

    if df[config.MESSAGE_COLUMN].isna().any():
        raise DataValidationError(
            f"Found {int(df[config.MESSAGE_COLUMN].isna().sum())} missing (NaN) messages."
        )

    warnings: list[str] = []

    n_empty = int((df[config.MESSAGE_COLUMN].str.strip() == "").sum())
    if n_empty:
        warnings.append(
            f"{n_empty} empty/whitespace-only messages (dropped in cleaning stage)"
        )

    n_dup = int(
        df.duplicated(subset=[config.LABEL_COLUMN, config.MESSAGE_COLUMN]).sum()
    )
    if n_dup:
        warnings.append(
            f"{n_dup} exact duplicate rows (removed BEFORE the train/test split "
            "to avoid train/test leakage)"
        )

    class_counts = df[config.LABEL_COLUMN].value_counts().to_dict()
    report = ValidationReport(
        n_rows=len(df),
        n_columns=df.shape[1],
        n_duplicate_rows=n_dup,
        n_empty_messages=n_empty,
        class_counts=class_counts,
        warnings=warnings,
    )

    logger.info("Validation PASSED: %d rows, %d columns", report.n_rows, report.n_columns)
    logger.info(
        "Class distribution: ham=%d (%.2f%%) spam=%d (%.2f%%)",
        class_counts.get("ham", 0),
        class_counts.get("ham", 0) / report.n_rows * 100,
        class_counts.get("spam", 0),
        report.spam_percentage,
    )
    for w in warnings:
        logger.warning("Validation warning: %s", w)
    return report
