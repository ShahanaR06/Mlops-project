"""Run the complete pipeline end-to-end WITHOUT Airflow.

Uses exactly the same stage functions as the Airflow DAG - useful for quick
local verification and debugging:

    python run_pipeline.py
"""

import logging

import pandas as pd

import config
from src import data_loader, data_validation, evaluate, preprocessing, text_cleaning, train
from src.mlflow_utils import write_final_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_pipeline")


def main() -> dict:
    # 1 - load
    raw_path = data_loader.download_dataset()
    df = data_loader.load_dataset(raw_path)

    # 2 - validate (raises on hard failures)
    report = data_validation.validate_dataset(df)
    logger.info("Validation report: %s", report)

    # 3 - clean
    cleaned = text_cleaning.clean_dataset(df)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(config.CLEAN_DATA_PATH, index=False)

    # 4 - split (BEFORE any TF-IDF fitting -> no leakage)
    encoded = preprocessing.encode_labels(cleaned)
    train_df, test_df = preprocessing.split_data(encoded)
    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    test_df.to_csv(config.TEST_DATA_PATH, index=False)

    # 5 - train (creates one MLflow run per model)
    train.train_all(train_df)

    # 6 - evaluate (logs metrics + artifacts to those runs)
    summary = evaluate.evaluate_all()

    # 7 - final report
    report_path = write_final_report()

    print("\n================ FINAL RESULTS (test set) ================")
    for name, metrics in summary.items():
        print(
            f"{name:>20}: accuracy={metrics['accuracy']:.4f} "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
            f"f1={metrics['f1_score']:.4f} roc_auc={metrics['roc_auc']:.4f}"
        )
    print(f"\nFull report: {report_path}")
    print(f"MLflow UI:   mlflow ui --backend-store-uri \"{config.MLFLOW_TRACKING_URI}\" --port 5000")
    return summary


if __name__ == "__main__":
    main()
