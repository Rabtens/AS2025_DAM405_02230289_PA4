"""Central configuration. Paths and switches come from the environment (or Airflow
Variables in the DAG), never from hard-coded values. No credentials are used or
needed by the pipeline itself: when deploying to AWS, SAM/boto3 resolve
credentials from the standard provider chain (env vars, profile, or IAM role)."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("MLOPS_DATA_DIR", REPO_ROOT / "data"))
LAMBDA_DIR = REPO_ROOT / "lambda_fn"
BUILD_DIR = Path(os.environ.get("MLOPS_BUILD_DIR", REPO_ROOT / "build"))
EVENTS_DIR = REPO_ROOT / "events"

TARGET_COLUMN = "target"
RANDOM_SEED = 42
# Minimum test accuracy a candidate model must reach before it may be deployed.
MIN_TEST_ACCURACY = float(os.environ.get("MLOPS_MIN_ACCURACY", "0.93"))
CLASS_LABELS = {"0": "malignant", "1": "benign"}  # sklearn convention


def partition(root: Path, stage: str, ds: str) -> Path:
    """Hive-style partition directory: <root>/<stage>/ds=<YYYY-MM-DD>."""
    return Path(root) / stage / f"ds={ds}"
