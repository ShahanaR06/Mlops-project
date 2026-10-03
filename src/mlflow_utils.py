"""Thin, centralized MLflow helpers used by the training/evaluation stages
and by the Airflow DAG. MLflow 3.x API."""

import json
import logging
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

import config

logger = logging.getLogger(__name__)


def setup_mlflow(tracking_uri: str | None = None, artifact_location: str | None = None) -> str:
    """Point MLflow at the SQLite backend store and select the experiment.

    The experiment is created (once) with artifact_location pinned to the
    project's mlruns/ directory (unless overridden), so artifacts land in a
    known place no matter which working directory a task runs from.
    Returns the experiment id.
    """
    uri = tracking_uri or config.MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    if client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME) is None:
        location = artifact_location or str(config.MLRUNS_DIR)
        if artifact_location is None:
            config.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
        client.create_experiment(config.MLFLOW_EXPERIMENT_NAME, artifact_location=location)
    experiment = mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    logger.info("MLflow tracking URI: %s | experiment: %s (id=%s)",
                uri, config.MLFLOW_EXPERIMENT_NAME, experiment.experiment_id)
    return experiment.experiment_id


def build_run_params(model_name: str, model_params: dict) -> dict:
    """Assemble the parameter dict logged for one model run."""
    params = {
        "model_name": model_name,
        "test_size": config.TEST_SIZE,
        "random_state": config.RANDOM_STATE,
    }
    # TF-IDF params (ngram_range logged as a readable string)
    for key, value in config.TFIDF_PARAMS.items():
        params[f"tfidf_{key}"] = str(value) if key == "ngram_range" else value
    # Model hyperparameters
    params.update(model_params)
    return params


def list_experiment_runs(tracking_uri: str | None = None) -> pd.DataFrame:
    """Return all runs of our experiment as a DataFrame (for reporting/tests)."""
    uri = tracking_uri or config.MLFLOW_TRACKING_URI
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        return pd.DataFrame()
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    return pd.DataFrame(
        [
            {
                "run_id": r.info.run_id,
                "run_name": r.info.run_name,
                "status": r.info.status,
                **{f"metrics.{k}": v for k, v in r.data.metrics.items()},
                **{f"params.{k}": v for k, v in r.data.params.items()},
            }
            for r in runs
        ]
    )


def write_final_report(tracking_uri: str | None = None) -> Path:
    """Final pipeline step: write artifacts/pipeline_summary.md.

    Combines the evaluation summary (metrics per model) with the authoritative
    list of MLflow runs, so the Airflow log + the file both show a clean,
    verifiable record of what was produced.
    """
    summary = json.loads(config.SUMMARY_PATH.read_text(encoding="utf-8"))
    runs_df = list_experiment_runs(tracking_uri)

    lines = [
        "# SMS Spam Detection - Pipeline Summary",
        "",
        f"MLflow experiment: `{config.MLFLOW_EXPERIMENT_NAME}`",
        f"Tracking URI: `{tracking_uri or config.MLFLOW_TRACKING_URI}`",
        "",
        "## Test-set metrics",
        "",
        "| model | accuracy | precision | recall | f1_score | roc_auc | run_id |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in summary.items():
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} "
            f"| {m['recall']:.4f} | {m['f1_score']:.4f} | {m['roc_auc']:.4f} "
            f"| `{m['run_id'][:8]}...` |"
        )
    lines += [
        "",
        f"## MLflow runs recorded: {len(runs_df)}",
        "",
        "Inspect interactively:",
        "",
        f"    mlflow ui --backend-store-uri \"{config.MLFLOW_TRACKING_URI}\" --port 5000",
        "",
    ]
    config.FINAL_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Final report written to %s", config.FINAL_REPORT_PATH)
    return config.FINAL_REPORT_PATH
