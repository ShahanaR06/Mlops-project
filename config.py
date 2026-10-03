"""Central configuration for the SMS spam MLOps project.

All paths are derived from this file's location, so the project is portable
and contains no hardcoded absolute paths. These values are shared by the
standalone runner (run_pipeline.py), the Airflow DAG (dags/sms_spam_dag.py),
the prediction demo (predict.py) and the test suite.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "SMSSpamCollection"
DATASET_ZIP_PATH = RAW_DATA_DIR / "smsspamcollection.zip"
# UCI ML Repository - SMS Spam Collection (verified working, Oct 2026)
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/"
    "smsspamcollection.zip"
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CLEAN_DATA_PATH = ARTIFACTS_DIR / "cleaned.csv"
TRAIN_DATA_PATH = ARTIFACTS_DIR / "train.csv"
TEST_DATA_PATH = ARTIFACTS_DIR / "test.csv"
RUN_IDS_PATH = ARTIFACTS_DIR / "run_ids.json"
SUMMARY_PATH = ARTIFACTS_DIR / "summary.json"
FINAL_REPORT_PATH = ARTIFACTS_DIR / "pipeline_summary.md"

MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# ---------------------------------------------------------------------------
# Dataset schema (verified by inspecting the real file - see README)
# ---------------------------------------------------------------------------
LABEL_COLUMN = "label"
MESSAGE_COLUMN = "message"
LABEL_ID_COLUMN = "label_id"
EXPECTED_LABELS = ("ham", "spam")
LABEL_MAP = {"ham": 0, "spam": 1}  # ham -> legitimate, spam -> unwanted
DATASET_ENCODING = "utf-8"  # verified: file decodes cleanly as UTF-8

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# TF-IDF parameters.
# The vectorizer lives INSIDE each sklearn Pipeline and is therefore fitted
# on training messages only -> no data leakage from the test set.
# ---------------------------------------------------------------------------
TFIDF_PARAMS = {
    "ngram_range": (1, 2),  # bigrams catch phrases like "call now", "text STOP"
    "min_df": 2,            # drop one-off tokens/typos -> smaller, stabler vocab
    "max_df": 0.98,         # drop near-ubiquitous tokens
    "max_features": 10000,  # cap vocabulary; more than enough for ~5k messages
    "sublinear_tf": True,   # log-scale TF so long messages don't dominate
}

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
NAIVE_BAYES_PARAMS = {"alpha": 1.0}  # standard Laplace smoothing baseline
LOGISTIC_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,       # generous so liblinear always converges
    "solver": "liblinear",  # solid choice for small, sparse, binary problems
    "random_state": RANDOM_STATE,
}

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_EXPERIMENT_NAME = "sms-spam-classification"
# MLflow 3.x put the plain filesystem store into maintenance mode, so we use
# the recommended local SQLite backend store. Run artifacts (plots, reports,
# logged models) still live under ./mlruns (set as the experiment's
# artifact_location in mlflow_utils.setup_mlflow).
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
