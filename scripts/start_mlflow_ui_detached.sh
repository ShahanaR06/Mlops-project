#!/usr/bin/env bash
# Start the MLflow UI FULLY DETACHED (survives the launching terminal).
# UI: http://localhost:5000 ; logs -> /root/mlflow_ui.log
set -e
if [ -f /root/mlflow_ui.pid ] && kill -0 "$(cat /root/mlflow_ui.pid)" 2>/dev/null; then
  echo "ALREADY RUNNING pid=$(cat /root/mlflow_ui.pid)"
  exit 0
fi
setsid nohup /root/sms-venv/bin/mlflow ui \
  --backend-store-uri 'sqlite:////mnt/d/Mlops project/mlflow.db' \
  --port 5000 --workers 1 > /root/mlflow_ui.log 2>&1 &
echo $! > /root/mlflow_ui.pid
echo "STARTED pid=$(cat /root/mlflow_ui.pid)"
