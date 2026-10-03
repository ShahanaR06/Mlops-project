"""Stage 6 - evaluation.

Scores each fitted pipeline on the HELD-OUT test set, generates evaluation
artifacts (confusion matrix, ROC curve, classification report, top spam/ham
terms) and logs metrics + artifacts to the SAME MLflow runs created during
training.

Metric intuition for spam detection (spam = positive class):
  * precision - of the messages we flagged as spam, how many really were?
    (high precision = few legitimate messages wrongly blocked)
  * recall    - of all real spam messages, how many did we catch?
    (high recall = little spam reaches the inbox)
  * f1_score  - harmonic balance of the two
Accuracy alone is misleading here: the data is ~87% ham, so a lazy
"everything is ham" classifier already scores ~87%.
"""

import json
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless - no display available under Airflow/WSL

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import config
from src.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)

METRIC_NAMES = ("accuracy", "precision", "recall", "f1_score", "roc_auc")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """Binary metrics with spam (1) as the positive class."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, path: Path) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["ham", "spam"],
        yticklabels=["ham", "spam"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_roc_curve(y_true, y_score, path: Path, model_name: str) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_classification_report(y_true, y_pred, path: Path) -> Path:
    report = classification_report(
        y_true, y_pred, target_names=["ham", "spam"], digits=4
    )
    Path(path).write_text(report, encoding="utf-8")
    return path


def save_top_terms(pipe, path_prefix: Path, top_n: int = 25) -> list[Path]:
    """Rank features by their spam-vs-ham signal and write two text files.

    Naive Bayes:  log P(term|spam) - log P(term|ham)
    Logistic Regr: fitted coefficient for the term
    """
    vectorizer = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["classifier"]
    terms = np.asarray(vectorizer.get_feature_names_out())

    if hasattr(clf, "coef_"):                      # LogisticRegression
        scores = clf.coef_[0]
    elif hasattr(clf, "feature_log_prob_"):        # MultinomialNB
        scores = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    else:
        logger.warning("No term-scoring method for %s - skipping", type(clf).__name__)
        return []

    order = np.argsort(scores)
    spam_terms = terms[order[::-1][:top_n]]
    ham_terms = terms[order[:top_n]]

    spam_path = path_prefix.with_name("top_spam_terms.txt")
    ham_path = path_prefix.with_name("top_ham_terms.txt")
    spam_path.write_text("\n".join(spam_terms), encoding="utf-8")
    ham_path.write_text("\n".join(ham_terms), encoding="utf-8")
    return [spam_path, ham_path]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def evaluate_pipeline(
    name: str,
    pipe,
    X_test,
    y_test,
    out_dir: Path,
    run_id: str | None = None,
    tracking_uri: str | None = None,
) -> dict[str, float]:
    """Score one pipeline, write artifacts, and (optionally) log to MLflow."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_pred = pipe.predict(X_test)
    # positive-class scores for ROC-AUC
    if hasattr(pipe, "predict_proba"):
        y_score = pipe.predict_proba(X_test)[:, 1]
    else:
        y_score = pipe.decision_function(X_test)

    metrics = compute_metrics(y_test, y_pred, y_score)
    logger.info(
        "[%s] " + " | ".join(f"{k}={metrics[k]:.4f}" for k in METRIC_NAMES), name
    )

    artifacts = [
        save_confusion_matrix(y_test, y_pred, out_dir / "confusion_matrix.png"),
        save_roc_curve(y_test, y_score, out_dir / "roc_curve.png", name),
        save_classification_report(y_test, y_pred, out_dir / "classification_report.txt"),
    ]
    artifacts += save_top_terms(pipe, out_dir / "top_terms")

    if run_id:
        # Always point MLflow at the project store: Airflow tasks run in
        # separate processes, so in-process tracking-URI state never carries
        # over (a fresh default sqlite db would silently be created instead).
        setup_mlflow(tracking_uri)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)
            for artifact in artifacts:
                mlflow.log_artifact(str(artifact))
        logger.info("[%s] metrics + %d artifacts logged to run %s", name, len(artifacts), run_id)

    return metrics


def evaluate_all(tracking_uri: str | None = None) -> dict[str, dict]:
    """Evaluate every model produced by train_all() on the test split.

    Loads test.csv + run_ids.json + models/<name>.joblib from disk so it can
    run as an independent Airflow task. Writes artifacts/summary.json.
    """
    setup_mlflow(tracking_uri)

    test_df = pd.read_csv(config.TEST_DATA_PATH)
    X_test = test_df[config.MESSAGE_COLUMN]
    y_test = test_df[config.LABEL_ID_COLUMN]

    run_ids: dict[str, str] = json.loads(config.RUN_IDS_PATH.read_text(encoding="utf-8"))

    summary: dict[str, dict] = {}
    for name, run_id in run_ids.items():
        pipe = joblib.load(config.MODELS_DIR / f"{name}.joblib")
        metrics = evaluate_pipeline(
            name=name,
            pipe=pipe,
            X_test=X_test,
            y_test=y_test,
            out_dir=config.ARTIFACTS_DIR / name,
            run_id=run_id,
        )
        summary[name] = {"run_id": run_id, **metrics}

    config.SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Evaluation summary -> %s", config.SUMMARY_PATH)
    return summary
