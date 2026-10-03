"""Training + evaluation on synthetic data (MLflow disabled here)."""

import numpy as np

from src import preprocessing
from src.evaluate import METRIC_NAMES, compute_metrics, evaluate_pipeline
from src.train import MODEL_BUILDERS, build_pipeline, train_all


def test_both_models_train_and_predict(synthetic_sms_df, tmp_path, monkeypatch):
    monkeypatch.setattr("config.MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr("config.ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr("config.RUN_IDS_PATH", tmp_path / "artifacts" / "run_ids.json")

    encoded = preprocessing.encode_labels(synthetic_sms_df)
    train_df, _ = preprocessing.split_data(encoded, 0.2, 42)

    results = train_all(train_df, log_to_mlflow=False)

    assert set(results) == set(MODEL_BUILDERS)  # naive_bayes + logistic_regression
    for name, (pipe, run_id) in results.items():
        assert run_id is None  # MLflow disabled
        preds = pipe.predict(["win free cash now", "see you tomorrow"])
        assert set(preds) <= {0, 1}
        proba = pipe.predict_proba(["free prize"])
        assert proba.shape == (1, 2)
        assert (tmp_path / "models" / f"{name}.joblib").exists()


def test_pipeline_handles_raw_text_end_to_end(synthetic_sms_df):
    encoded = preprocessing.encode_labels(synthetic_sms_df)
    train_df, _ = preprocessing.split_data(encoded, 0.2, 42)
    pipe = build_pipeline("logistic_regression")
    pipe.fit(train_df["message"], train_df["label_id"])
    # raw, unseen, uncleaned text straight in - pipeline does TF-IDF itself
    assert pipe.predict(["URGENT! You WON £500 cash!!!"])[0] in (0, 1)


def test_metrics_are_generated(synthetic_sms_df, tmp_path):
    encoded = preprocessing.encode_labels(synthetic_sms_df)
    train_df, test_df = preprocessing.split_data(encoded, 0.2, 42)

    pipe = build_pipeline("naive_bayes")
    pipe.fit(train_df["message"], train_df["label_id"])

    metrics = evaluate_pipeline(
        "naive_bayes", pipe, test_df["message"], test_df["label_id"], tmp_path
    )
    for key in METRIC_NAMES:  # accuracy, precision, recall, f1_score, roc_auc
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0

    # artifacts were really written
    for fname in ("confusion_matrix.png", "roc_curve.png",
                  "classification_report.txt", "top_spam_terms.txt",
                  "top_ham_terms.txt"):
        assert (tmp_path / fname).exists()


def test_compute_metrics_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y_true, y_pred, y_score)
    assert metrics == {k: 1.0 for k in METRIC_NAMES}
