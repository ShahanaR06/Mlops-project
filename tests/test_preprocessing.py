"""Split/encoding correctness + explicit data-leakage guards."""

import pytest

from src import preprocessing
from src.train import build_pipeline


def test_encode_labels(synthetic_sms_df):
    encoded = preprocessing.encode_labels(synthetic_sms_df)
    assert set(encoded["label_id"].unique()) == {0, 1}
    mapping = dict(zip(encoded["label"], encoded["label_id"]))
    assert mapping == {"ham": 0, "spam": 1}


def test_encode_labels_rejects_unknown_label(synthetic_sms_df):
    df = synthetic_sms_df.replace({"label": {"spam": "junk"}})
    with pytest.raises(ValueError, match="Unmappable labels"):
        preprocessing.encode_labels(df)


def test_split_is_stratified(synthetic_sms_df):
    encoded = preprocessing.encode_labels(synthetic_sms_df)
    train_df, test_df = preprocessing.split_data(encoded, test_size=0.2, random_state=42)
    train_ratio = train_df["label_id"].mean()
    test_ratio = test_df["label_id"].mean()
    assert abs(train_ratio - 0.5) < 1e-9
    assert abs(test_ratio - 0.5) < 1e-9
    assert len(train_df) == 96 and len(test_df) == 24


def test_split_is_reproducible(synthetic_sms_df):
    encoded = preprocessing.encode_labels(synthetic_sms_df)
    train_a, test_a = preprocessing.split_data(encoded, 0.2, 42)
    train_b, test_b = preprocessing.split_data(encoded, 0.2, 42)
    assert train_a["message"].tolist() == train_b["message"].tolist()
    assert test_a["message"].tolist() == test_b["message"].tolist()


def test_vectorizer_is_unfitted_before_training():
    """Leakage guard: a freshly built pipeline must NOT have a vocabulary."""
    pipe = build_pipeline("naive_bayes")
    assert not hasattr(pipe.named_steps["tfidf"], "vocabulary_")


def test_vocabulary_comes_from_training_data_only(synthetic_sms_df):
    """After fitting on train, test-only tokens must be absent from the vocab."""
    encoded = preprocessing.encode_labels(synthetic_sms_df)
    train_df, test_df = preprocessing.split_data(encoded, 0.2, 42)

    pipe = build_pipeline("naive_bayes")
    pipe.fit(train_df["message"], train_df["label_id"])
    vocab = set(pipe.named_steps["tfidf"].vocabulary_)

    # craft tokens that exist ONLY in the test split
    test_only = test_df["message"].iloc[0] + " zzzunique_token_zzz qqq_other_qqq"
    for token in ("zzzunique_token_zzz", "qqq_other_qqq"):
        assert token not in vocab
    # and the pipeline still handles that unseen text without crashing
    pipe.predict([test_only])


def test_no_row_overlap_between_train_and_test(synthetic_sms_df):
    encoded = preprocessing.encode_labels(synthetic_sms_df)
    train_df, test_df = preprocessing.split_data(encoded, 0.2, 42)
    overlap = set(train_df["message"]) & set(test_df["message"])
    assert overlap == set()  # dedup in cleaning guarantees disjoint messages
