"""MLflow integration test: a real run with params, metrics, artifacts and a
logged model is created in a TEMPORARY file store (project mlruns untouched)."""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import config
from src import preprocessing
from src.mlflow_utils import list_experiment_runs, setup_mlflow
from src.train import build_pipeline


def test_mlflow_run_lifecycle(synthetic_sms_df, tmp_path):
    # MLflow 3.x: plain filesystem stores are deprecated -> use SQLite backend
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    exp_id = setup_mlflow(tracking_uri, artifact_location=str(tmp_path / "artifacts"))
    assert exp_id is not None

    # train a tiny pipeline and log everything like the real stage does
    encoded = preprocessing.encode_labels(synthetic_sms_df)
    train_df, _ = preprocessing.split_data(encoded, 0.2, 42)
    pipe = build_pipeline("naive_bayes")
    pipe.fit(train_df["message"], train_df["label_id"])

    artifact = tmp_path / "report.txt"
    artifact.write_text("precision 1.0", encoding="utf-8")

    with mlflow.start_run(run_name="pytest_run") as run:
        mlflow.log_params({"model_name": "naive_bayes", "alpha": 1.0})
        mlflow.log_metrics({"accuracy": 0.99, "roc_auc": 0.99})
        mlflow.log_artifact(str(artifact))
        model_info = mlflow.sklearn.log_model(pipe, name="model")
        run_id = run.info.run_id

    # verify through the client - nothing is mocked
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    assert experiment is not None

    fetched = client.get_run(run_id)
    assert fetched.data.params["model_name"] == "naive_bayes"
    assert fetched.data.metrics["accuracy"] == 0.99

    run_artifacts = {a.path for a in client.list_artifacts(run_id)}
    assert "report.txt" in run_artifacts

    # MLflow 3: logged models are first-class entities (not run artifacts)
    logged_model = client.get_logged_model(model_info.model_id)
    assert logged_model.name == "model"

    runs_df = list_experiment_runs(tracking_uri)
    assert len(runs_df) == 1
    assert runs_df.iloc[0]["run_name"] == "pytest_run"
