#!/usr/bin/env bash
# Source this file:  source scripts/airflow_env.sh
# Points Airflow at this repo (local SQLite metadata DB, SequentialExecutor, no auth needed for `dags test`).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AIRFLOW_HOME="$REPO_ROOT/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$REPO_ROOT/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__UNIT_TEST_MODE=False
export AIRFLOW__LOGGING__LOGGING_LEVEL=INFO
export MLOPS_DATA_DIR="${MLOPS_DATA_DIR:-$REPO_ROOT/data}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
