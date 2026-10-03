#!/usr/bin/env bash
# Start Airflow standalone FULLY DETACHED (survives the launching terminal).
# Recommended for long-running use; logs -> /root/airflow_standalone.log
set -e
source "$(dirname "$0")/airflow_env.sh"
mkdir -p /root/airflow
if [ -f /root/airflow/standalone.pid ] && kill -0 "$(cat /root/airflow/standalone.pid)" 2>/dev/null; then
  echo "ALREADY RUNNING pid=$(cat /root/airflow/standalone.pid)"
  exit 0
fi
setsid nohup airflow standalone > /root/airflow_standalone.log 2>&1 &
echo $! > /root/airflow/standalone.pid
echo "STARTED pid=$(cat /root/airflow/standalone.pid)"
