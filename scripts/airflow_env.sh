# Shared Airflow environment for this project (source this file).
# Keeps AIRFLOW_HOME inside WSL (fast, POSIX) while DAGs/code stay in the repo.
export AIRFLOW_HOME=/root/airflow
export AIRFLOW__CORE__DAGS_FOLDER='/mnt/d/Mlops project/dags'
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT=60
# `airflow standalone` re-invokes `airflow` sub-processes from PATH:
export PATH="/root/sms-venv/bin:$PATH"
# CRITICAL (verified): mlflow->skops emits a giant warning storm when imported
# inside an Airflow task worker (thousands of structlog JSON lines through the
# supervision channel -> memory/CPU pressure; on this 8 GB host the task then
# gets OOM-killed). The warning class is not a plain DeprecationWarning, so we
# suppress warnings wholesale in the Airflow environment. This does not affect
# pipeline behavior or results.
export PYTHONWARNINGS="ignore"
# Memory discipline on small hosts: run tasks sequentially inside the
# scheduler process space instead of forking parallel workers.
export AIRFLOW__CORE__EXECUTOR=SequentialExecutor
