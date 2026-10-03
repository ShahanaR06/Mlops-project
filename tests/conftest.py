"""Shared fixtures: small synthetic SMS dataset (no download needed)."""

import random

import pandas as pd
import pytest

SPAM_WORDS = ["free", "win", "winner", "prize", "cash", "claim", "urgent",
              "congratulations", "offer", "£1000", "call", "text", "now"]
HAM_WORDS = ["meeting", "lunch", "see", "you", "tomorrow", "thanks", "okay",
             "home", "later", "love", "yes", "no", "maybe"]


def _make_messages(words, n, rng):
    return [" ".join(rng.choices(words, k=rng.randint(5, 12))) for _ in range(n)]


@pytest.fixture()
def synthetic_sms_df() -> pd.DataFrame:
    """120 messages (60 ham / 60 spam) with clearly separated vocabulary."""
    rng = random.Random(7)
    messages = (
        _make_messages(HAM_WORDS, 60, rng) + _make_messages(SPAM_WORDS, 60, rng)
    )
    labels = ["ham"] * 60 + ["spam"] * 60
    df = pd.DataFrame({"label": labels, "message": messages})
    return df.sample(frac=1.0, random_state=7).reset_index(drop=True)
