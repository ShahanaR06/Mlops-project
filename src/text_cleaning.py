"""Stage 3 - text cleaning.

Deliberately CONSERVATIVE preprocessing. SMS spam is signaled heavily by
exactly the things naive cleaning removes:

  * currency symbols / amounts ..... "£1000", "$500 prize"
  * phone numbers / shortcodes ..... "call 09061701461", "text FA to 87121"
  * URLs ........................... "http://win-free-stuff.co"
  * punctuation / casing ........... "WINNER!!", "FREE*"

So we keep all of that and only do lossless normalization:
  1. Unicode NFKC normalization (folds full-width/compat characters)
  2. lowercasing (TF-IDF is case-sensitive by default)
  3. whitespace collapsing (tabs/newlines/extra spaces -> single space)

This stage also removes exact duplicate rows (counted in validation) and
drops messages that became empty - both actions are logged, never silent.
Duplicates are removed BEFORE the train/test split to prevent leakage.
"""

import logging
import re
import unicodedata

import pandas as pd

import config

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize a single message. Safe on None/empty/non-string input."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning to the whole dataset; drop empties and exact duplicates."""
    out = df.copy()
    out[config.MESSAGE_COLUMN] = out[config.MESSAGE_COLUMN].map(clean_text)

    n_before = len(out)
    out = out[out[config.MESSAGE_COLUMN] != ""]
    n_empty = n_before - len(out)
    if n_empty:
        logger.warning("Dropped %d messages that were empty after cleaning", n_empty)

    n_before = len(out)
    out = out.drop_duplicates(subset=[config.LABEL_COLUMN, config.MESSAGE_COLUMN])
    n_dup = n_before - len(out)
    logger.info(
        "Removed %d exact duplicate rows (leakage prevention): %d -> %d rows",
        n_dup, n_before, len(out),
    )

    return out.reset_index(drop=True)
