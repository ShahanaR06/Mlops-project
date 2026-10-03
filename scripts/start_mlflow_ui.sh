#!/usr/bin/env bash
# Start the MLflow UI against the project's SQLite tracking store.
# UI: http://localhost:5000
set -e
exec /root/sms-venv/bin/mlflow ui \
  --backend-store-uri 'sqlite:////mnt/d/Mlops project/mlflow.db' \
  --port 5000 --workers 1
