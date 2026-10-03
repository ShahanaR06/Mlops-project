"""Stage 5 - model training.

Builds complete sklearn Pipelines (TfidfVectorizer -> classifier) and fits
them on the TRAINING messages only. Because the vectorizer is a pipeline
step, test data cannot leak into the vocabulary, and a loaded pipeline can
classify raw SMS text end-to-end.

Each model gets its own MLflow run with parameters and the full pipeline
logged as an MLflow model. Fitted pipelines are also exported to models/
via joblib so the evaluation stage and predict.py can load them cheaply
(MLflow remains the primary experiment record).
"""

import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import config
from src.mlflow_utils import build_run_params, setup_mlflow

logger = logging.getLogger(__name__)

MODEL_BUILDERS = {
    "naive_bayes": lambda: MultinomialNB(**config.NAIVE_BAYES_PARAMS),
    "logistic_regression": lambda: LogisticRegression(**config.LOGISTIC_PARAMS),
}
MODEL_PARAMS = {
    "naive_bayes": dict(config.NAIVE_BAYES_PARAMS),
    "logistic_regression": dict(config.LOGISTIC_PARAMS),
}


def build_pipeline(model_name: str) -> Pipeline:
    """Create an UNFITTED TfidfVectorizer -> classifier pipeline."""
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {sorted(MODEL_BUILDERS)}"
        )
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**config.TFIDF_PARAMS)),
            ("classifier", MODEL_BUILDERS[model_name]()),
        ]
    )


def train_all(
    train_df: pd.DataFrame,
    log_to_mlflow: bool = True,
    tracking_uri: str | None = None,
) -> dict[str, tuple[Pipeline, str | None]]:
    """Fit every model on the training split.

    Returns {model_name: (fitted_pipeline, mlflow_run_id)} and persists:
      * models/<name>.joblib     - fitted pipeline for evaluation/serving
      * artifacts/run_ids.json   - so the evaluation stage can attach to runs
    """
    X_train = train_df[config.MESSAGE_COLUMN]
    y_train = train_df[config.LABEL_ID_COLUMN]
    logger.info("Training %d models on %d messages", len(MODEL_BUILDERS), len(train_df))

    if log_to_mlflow:
        setup_mlflow(tracking_uri)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, tuple[Pipeline, str | None]] = {}
    run_ids: dict[str, str] = {}

    for name in MODEL_BUILDERS:
        pipe = build_pipeline(name)
        pipe.fit(X_train, y_train)  # <-- TF-IDF is fitted HERE, on train only
        vocab_size = len(pipe.named_steps["tfidf"].vocabulary_)
        logger.info("[%s] fitted. TF-IDF vocabulary: %d terms", name, vocab_size)

        joblib.dump(pipe, config.MODELS_DIR / f"{name}.joblib")

        run_id = None
        if log_to_mlflow:
            with mlflow.start_run(run_name=name) as run:
                run_id = run.info.run_id
                mlflow.log_params(build_run_params(name, MODEL_PARAMS[name]))
                mlflow.log_param("tfidf_vocabulary_size", vocab_size)
                # NB: MLflow 3 input_example does not accept a bare Series,
                # so we pass a plain list of strings (signature uses a 1-col DF).
                signature = infer_signature(
                    X_train.head(5).to_frame(), pipe.predict(X_train.head(5))
                )
                mlflow.sklearn.log_model(
                    pipe,
                    name="model",  # MLflow 3.x API (was artifact_path in 2.x)
                    signature=signature,
                    input_example=X_train.head(5).tolist(),
                )
                logger.info("[%s] MLflow run created: %s", name, run_id)

        run_ids[name] = run_id
        results[name] = (pipe, run_id)

    config.RUN_IDS_PATH.write_text(json.dumps(run_ids, indent=2), encoding="utf-8")
    return results
