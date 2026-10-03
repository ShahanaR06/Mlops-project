"""Airflow DAG: SMS spam detection NLP pipeline with MLflow tracking.

    load_data -> validate_data -> clean_text -> split_data
        -> train_models -> evaluate_models -> log_results

Every task is a thin orchestration wrapper around the reusable functions in
src/ (the same functions used by run_pipeline.py and the tests). Data flows
between tasks through versioned files (artifacts/, models/), not XComs -
tasks run as independent processes. Training and evaluation create REAL
MLflow runs in the `sms-spam-classification` experiment.
"""

import sys
from datetime import datetime
from pathlib import Path

from airflow.sdk import dag, task

# Make config + src importable when Airflow parses this file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

DEFAULT_ARGS = {"retries": 0}  # fail fast and visibly during development


@dag(
    dag_id="sms_spam_detection",
    description="NLP spam pipeline: load -> validate -> clean -> split -> train -> evaluate -> report (MLflow-tracked)",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["nlp", "spam-detection", "mlflow", "mlops"],
)
def sms_spam_detection():
    @task()
    def load_data() -> None:
        import logging

        from src import data_loader

        logging.basicConfig(level=logging.INFO)
        raw_path = data_loader.download_dataset()
        df = data_loader.load_dataset(raw_path)
        logging.getLogger(__name__).info(
            "load_data OK: %d rows available at %s", len(df), raw_path
        )

    @task()
    def validate_data() -> None:
        import logging

        from src import data_loader, data_validation

        logging.basicConfig(level=logging.INFO)
        df = data_loader.load_dataset()
        report = data_validation.validate_dataset(df)  # raises on hard failures
        logging.getLogger(__name__).info(
            "validate_data OK: %s rows, spam %.2f%%, warnings: %s",
            report.n_rows, report.spam_percentage, report.warnings or "none",
        )

    @task()
    def clean_text() -> None:
        import logging

        from src import data_loader, text_cleaning

        logging.basicConfig(level=logging.INFO)
        df = data_loader.load_dataset()
        cleaned = text_cleaning.clean_dataset(df)
        config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(config.CLEAN_DATA_PATH, index=False)
        logging.getLogger(__name__).info(
            "clean_text OK: %d clean rows -> %s", len(cleaned), config.CLEAN_DATA_PATH
        )

    @task()
    def split_data() -> None:
        import logging

        import pandas as pd

        from src import preprocessing

        logging.basicConfig(level=logging.INFO)
        df = pd.read_csv(config.CLEAN_DATA_PATH)
        df = preprocessing.encode_labels(df)
        train_df, test_df = preprocessing.split_data(
            df, config.TEST_SIZE, config.RANDOM_STATE
        )
        train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
        test_df.to_csv(config.TEST_DATA_PATH, index=False)
        logging.getLogger(__name__).info(
            "split_data OK: train=%d test=%d (TF-IDF will be fitted on train only)",
            len(train_df), len(test_df),
        )

    @task()
    def train_models() -> None:
        import logging

        import pandas as pd

        from src import train

        logging.basicConfig(level=logging.INFO)
        train_df = pd.read_csv(config.TRAIN_DATA_PATH)
        results = train.train_all(train_df)  # creates one MLflow run per model
        logging.getLogger(__name__).info(
            "train_models OK: %s", {k: v[1] for k, v in results.items()}
        )

    @task()
    def evaluate_models() -> None:
        import logging

        from src import evaluate

        logging.basicConfig(level=logging.INFO)
        summary = evaluate.evaluate_all()  # logs metrics + artifacts to MLflow
        logging.getLogger(__name__).info("evaluate_models OK: %s", summary)

    @task()
    def log_results() -> None:
        import logging

        from src.mlflow_utils import write_final_report

        logging.basicConfig(level=logging.INFO)
        report_path = write_final_report()
        logging.getLogger(__name__).info(
            "PIPELINE COMPLETED SUCCESSFULLY - report: %s", report_path
        )

    # Linear dependency chain (each stage needs the previous stage's files)
    load_data() >> validate_data() >> clean_text() >> split_data() >> train_models() >> evaluate_models() >> log_results()


sms_spam_detection()
