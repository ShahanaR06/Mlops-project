#!/usr/bin/env bash
# Start the full local Airflow stack (db init + admin user + scheduler + UI).
# UI: http://localhost:8080  (admin password printed on first launch and
# stored in /root/airflow/standalone_admin_password.txt)
set -e
source "$(dirname "$0")/airflow_env.sh"
exec /root/sms-venv/bin/airflow standalone
