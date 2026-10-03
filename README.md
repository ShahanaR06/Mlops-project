# SMS Spam Detection — NLP MLOps Pipeline with MLflow + Apache Airflow

An end-to-end, reproducible **NLP machine-learning pipeline** that classifies SMS
messages as **ham** (legitimate) or **spam**, orchestrated by **Apache Airflow**
with experiment tracking in **MLflow**.

```
Raw SMS (UCI dataset)
   ↓
Data Validation          <- fail loudly on bad data
   ↓
Text Cleaning            <- conservative, keeps spam signals
   ↓
Train/Test Split         <- stratified, seeded, BEFORE any fitting
   ↓
TF-IDF                   <- fitted on TRAIN only (inside sklearn Pipeline)
   ↓
ML Models                <- Multinomial Naive Bayes + Logistic Regression
   ↓
Evaluation               <- accuracy/precision/recall/F1/ROC-AUC + plots
   ↓
MLflow Tracking          <- params, metrics, artifacts, logged pipelines
        ▲
        │ orchestrates every stage
   Apache Airflow
```

## Why NLP?

SMS messages are raw text, which ML models cannot consume directly. We convert
each message into a numeric vector with **TF-IDF** (Term Frequency–Inverse
Document Frequency): words (and word pairs) that are frequent in one message
but rare overall get high weight — exactly the "FREE", "£1000", "call now"
patterns that separate spam from ham.

## Dataset

**UCI SMS Spam Collection** (verified by direct inspection — not assumed):

| Property | Verified value |
|---|---|
| Source | https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip |
| File | `SMSSpamCollection` (477,907 bytes), TAB-separated, no header |
| Encoding | UTF-8 (file contains a few literal U+FFFD replacement chars) |
| Rows | **5,574** (matches the official `readme`: 4,827 ham + 747 spam) |
| Columns | `label` (`ham`/`spam`), `message` (raw SMS text) |
| Missing values | 0 (verified) |
| Empty messages | 0 (verified) |
| Exact duplicates | **403 rows** — removed before splitting (see below) |
| Class distribution | ham 86.60% / spam 13.40% |
| Metadata | none — only the two columns |

**Duplicate handling (anti-leakage):** 403 rows are exact duplicates. If kept,
identical messages could land in both train and test, inflating metrics. The
cleaning stage drops them **before** the split (5,574 → 5,171 unique rows).

## Tech stack

- **Python 3.14** (inside WSL2 — see "Why WSL" below)
- **pandas / NumPy** — data handling
- **scikit-learn** — `TfidfVectorizer`, `MultinomialNB`, `LogisticRegression`, `Pipeline`
- **matplotlib / seaborn** — confusion matrix + ROC curve artifacts
- **MLflow 3.x** — experiment tracking, metrics/artifacts, logged models
- **Apache Airflow 3.3.2** — pipeline orchestration (installed with official constraints)
- **pytest** — unit + integration tests

## Project structure

```
.
├── config.py                 # central configuration (paths, params, MLflow)
├── dags/
│   └── sms_spam_dag.py       # Airflow DAG (7 tasks)
├── src/
│   ├── data_loader.py        # download + load raw dataset
│   ├── data_validation.py    # hard-fail validation + report
│   ├── text_cleaning.py      # conservative normalization + dedup
│   ├── preprocessing.py      # label encoding + stratified split
│   ├── train.py              # TF-IDF pipelines + MLflow runs
│   ├── evaluate.py           # metrics, plots, artifacts
│   └── mlflow_utils.py       # MLflow setup + reporting helpers
├── tests/                    # pytest suite (unit + integration)
├── scripts/                  # WSL helper scripts (Airflow env, start UIs)
├── data/raw/                 # downloaded dataset (git-ignored)
├── artifacts/                # generated CSVs, plots, reports (git-ignored)
├── models/                   # fitted pipelines (.joblib, git-ignored)
├── mlruns/                   # MLflow run artifacts (git-ignored)
├── mlflow.db                 # MLflow SQLite tracking store (git-ignored)
├── run_pipeline.py           # run everything WITHOUT Airflow
├── predict.py                # demo inference on raw SMS text
└── requirements.txt
```

## Why WSL2? (Windows note)

Apache Airflow officially supports **POSIX systems only** — on Windows the
supported local option is **WSL2** (see the Airflow PyPI page). The project
files stay in `D:\Mlops project` (visible in WSL as `/mnt/d/Mlops project`),
while the Python environment and `AIRFLOW_HOME` live inside Ubuntu for speed.

## Installation (verified commands)

All commands run **inside WSL Ubuntu** (`wsl -d Ubuntu`), with the venv that was
created during setup.

> **Small-host note (this machine: 8 GB RAM):** WSL2 is capped via
> `C:\Users\<you>\.wslconfig` → `memory=3GB`, `swap=4GB`. Without the swap,
> the `train_models` task gets OOM-killed on this host. Adjust to your machine.

```bash
# 1. create the environment (once)
python3 -m venv /root/sms-venv

# 2. install everything in one resolved command (Airflow pinned + constraints)
/root/sms-venv/bin/pip install \
    numpy pandas scikit-learn matplotlib seaborn joblib pytest "mlflow>=3.0" \
    "apache-airflow==3.3.2" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.3.2/constraints-3.14.txt"

# 3. quick check
cd "/mnt/d/Mlops project"
/root/sms-venv/bin/python -c "import sklearn, mlflow, airflow; print(sklearn.__version__, mlflow.__version__, airflow.__version__)"
```

## Running the project

### Option A — standalone (no Airflow)

```bash
cd "/mnt/d/Mlops project"
/root/sms-venv/bin/python run_pipeline.py
```

Runs all 7 stages and prints the real test metrics.

### Option B — Airflow (full MLOps experience)

```bash
# 1. start Airflow (first run initializes the DB + admin user)
#    recommended: detached, with the project's env (PATH, PYTHONWARNINGS, executor)
/mnt/d/Mlops\ project/scripts/start_airflow_detached.sh   # UI on http://localhost:8080
#    or foreground:  source /mnt/d/Mlops\ project/scripts/airflow_env.sh && airflow standalone
```

The generated admin password is stored in
`/root/airflow/simple_auth_manager_passwords.json.generated` (Airflow 3.3).

2. Open **http://localhost:8080**, log in as `admin`.
3. **Unpause** the **`sms_spam_detection`** DAG (Airflow creates DAGs paused by
   default — a CLI-triggered run stays `queued` until you do):
   `airflow dags unpause sms_spam_detection` or the toggle in the UI.
4. Trigger the DAG (play button, or `airflow dags trigger sms_spam_detection`).
5. Watch the 7 tasks go green; click a task → *Logs* to see stage output.

### MLflow UI

```bash
/root/sms-venv/bin/mlflow ui \
  --backend-store-uri 'sqlite:////mnt/d/Mlops project/mlflow.db' --port 5000
# or simply:  /mnt/d/Mlops\ project/scripts/start_mlflow_ui.sh
```

Open **http://localhost:5000** → experiment **`sms-spam-classification`** →
compare the `naive_bayes` and `logistic_regression` runs (params, metrics,
artifacts, logged models).

> Note: MLflow 3.x put the plain *filesystem* store in maintenance mode, so
> this project uses the recommended **SQLite backend store** (`mlflow.db`);
> run artifacts (plots, reports, logged models) live under `mlruns/`.

### Tests

```bash
cd "/mnt/d/Mlops project"
/root/sms-venv/bin/python -m pytest -v
```

### Demo prediction

```bash
cd "/mnt/d/Mlops project"
/root/sms-venv/bin/python predict.py
/root/sms-venv/bin/python predict.py -m "Free entry! Text WIN to 87121"
/root/sms-venv/bin/python predict.py --model naive_bayes -m "Are we meeting at 5 pm today?"
```

## Evaluation — what the metrics mean for spam

- **Accuracy** — overall correctness. Misleading alone: a dumb "always ham"
  model already scores ~87% here.
- **Precision** — of everything flagged as spam, how much really was spam?
  High precision = few legitimate messages wrongly blocked.
- **Recall** — of all real spam, how much did we catch?
  High recall = little spam reaches the inbox.
- **F1** — harmonic balance of precision and recall.
- **ROC-AUC** — ranking quality across all thresholds.

### Verified results (generated by the pipeline, not hand-written)

Test set: **1,032 messages** (12.40% spam), TF-IDF vocabulary: 10,000 terms.
Identical across the standalone run and the Airflow run (seed = 42).

| model | accuracy | precision | recall | f1_score | roc_auc |
|---|---|---|---|---|---|
| naive_bayes | 0.9632 | 1.0000 | 0.7031 | 0.8257 | 0.9758 |
| logistic_regression | 0.9651 | 1.0000 | 0.7188 | 0.8364 | 0.9959 |

Reading: both models are *conservative* — when they flag a message it is
essentially always spam (precision 1.0), at the cost of letting ~28–30% of
spam through (recall ≈ 0.70–0.72). Logistic Regression is strictly better
here (higher recall and ROC-AUC). Precision/recall trade-off could be shifted
by tuning the decision threshold — intentionally left as-is (no over-tuning).

## MLOps architecture

| Concern | Tool | What it does here |
|---|---|---|
| Orchestration | **Airflow** | Runs the 7 pipeline tasks in order, retries/alerts on failure, shows status/duration/logs in the UI at :8080 |
| Experiment tracking | **MLflow** | One run per model: params (TF-IDF + model hyperparams), 5 metrics, plots/reports as artifacts, full sklearn pipeline logged as a model. UI at :5000 |
| Reproducibility | `config.py` + fixed seed 42 + constraints file | Same split, same models, same numbers on re-run |

Airflow → MLflow integration: the `train_models` task creates the MLflow runs
(parameters + logged pipelines) and `evaluate_models` attaches metrics and
artifacts to those same runs. Triggering the DAG produces brand-new, visible
runs in the MLflow UI.

## Engineering decisions

- **Conservative text cleaning** — only Unicode NFKC + lowercase + whitespace
  normalization. Numbers, punctuation, URLs and currency symbols are *kept*:
  they are the strongest spam signals (`£1000`, `0906…`, `http://…`, `!!`).
- **Leakage prevention** — TF-IDF is a step *inside* each sklearn `Pipeline`,
  so it can only ever be fitted on training data; duplicates are dropped
  before splitting; tests assert the vocabulary contains no test-only tokens.
- **File-based task interfaces** — Airflow tasks exchange `artifacts/*.csv`
  and `models/*.joblib` instead of XComs (recommended for anything but tiny
  metadata), and each task is independently re-runnable.
- **MLflow file store** — a plain `mlruns/` directory; no server or database
  needed for tracking (the UI reads the same folder).

## Troubleshooting (issues actually encountered)

| Problem | Fix |
|---|---|
| `D:\Mlops project` was read-only for the user | granted full control via elevated `icacls ... /grant <user>:(OI)(CI)F` |
| Airflow won't install on Windows | expected — POSIX-only; installed WSL2 + Ubuntu, project accessed via `/mnt/d/...` |
| `wsl --install` quoting/UAC | ran an elevated PowerShell script file (arguments with spaces get mangled otherwise) |
| Ubuntu 26.04 ships Python 3.14 | verified Airflow 3.2+ supports 3.14 and a `constraints-3.14.txt` exists before installing |
| MLflow 3 `log_model(artifact_path=...)` deprecated | use `name="model"` (MLflow 3 API) |
| MLflow 3.16 rejects the filesystem tracking store | switched to the recommended `sqlite:///mlflow.db` backend; artifacts under `mlruns/` |
| `input_example=<pandas.Series>` rejected by MLflow | pass a plain `list` (Series→`tolist()`) |
| MLflow 3 logged models aren't in run artifacts | they're first-class *Logged Models* (experiment level) — tests use `client.get_logged_model()` |
| pandas merged 2 messages starting with `"` (5,574→5,572 rows) | read with `quoting=csv.QUOTE_NONE`; verified 5,574 records matches the UCI readme |
| `evaluate_models` created runs "not found" under Airflow | task processes don't share in-process MLflow state — `setup_mlflow()` must run in **every** task (fixed in `src/evaluate.py`) |
| `airflow standalone` can't spawn children | venv `bin` must be on `PATH` (done in `scripts/airflow_env.sh`) |
| Airflow 3.3: CLI-triggered run stuck `queued` | DAGs are created **paused** — `airflow dags unpause sms_spam_detection` |
| Airflow 3.3 API `POST /dagRuns` → 422 | `logical_date` is required in the request body |
| `airflow dags list-runs` not found | removed in Airflow 3.x — use the REST API (`/api/v2/...`) with a JWT from `POST /auth/token` |
| mlflow→skops warning storm froze task workers | the warning class isn't a plain `DeprecationWarning`; set `PYTHONWARNINGS=ignore` for Airflow (in `airflow_env.sh`) |
| `train_models` SIGKILLed (OOM) on 8 GB host | `.wslconfig` → `memory=3GB swap=4GB`; MLflow UI with `--workers 1`; `AIRFLOW__CORE__EXECUTOR=SequentialExecutor` |
| api-server dies under memory pressure | restart with `scripts/start_airflow_detached.sh`; keep other heavy apps closed on small hosts |
| `pkill -f mlflow` kills your own shell | the pattern matched the script path `start_mlflow_ui_detached.sh` — use `pkill -f 'sms-venv/bin/mlflow'` |
