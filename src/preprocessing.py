"""Stage 4 - label encoding and the stratified train/test split.

CRITICAL (data leakage): the split happens BEFORE any TF-IDF fitting.
The TfidfVectorizer lives inside each sklearn Pipeline and is fitted only
on training messages. Test messages never influence the vocabulary, and
test labels are never touched during training.
"""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

import config

logger = logging.getLogger(__name__)


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map ham -> 0, spam -> 1 into a new `label_id` column."""
    out = df.copy()
    out[config.LABEL_ID_COLUMN] = out[config.LABEL_COLUMN].map(config.LABEL_MAP)
    if out[config.LABEL_ID_COLUMN].isna().any():
        bad = sorted(
            out.loc[out[config.LABEL_ID_COLUMN].isna(), config.LABEL_COLUMN].unique()
        )
        raise ValueError(f"Unmappable labels found: {bad}")
    out[config.LABEL_ID_COLUMN] = out[config.LABEL_ID_COLUMN].astype(int)
    return out


def split_data(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified, reproducible split. Returns (train_df, test_df)."""
    if config.LABEL_ID_COLUMN not in df.columns:
        raise ValueError("Call encode_labels() before split_data().")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[config.LABEL_ID_COLUMN],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    spam_train = train_df[config.LABEL_ID_COLUMN].mean() * 100
    spam_test = test_df[config.LABEL_ID_COLUMN].mean() * 100
    logger.info(
        "Split: train=%d (spam %.2f%%) | test=%d (spam %.2f%%) | seed=%d stratified",
        len(train_df), spam_train, len(test_df), spam_test, random_state,
    )
    return train_df, test_df
