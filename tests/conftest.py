import json
import os
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("AIRFLOW_HOME", str(REPO_ROOT / ".airflow"))
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")

from pipeline import config, steps  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _airflow_metadata_db():
    """DagBag needs an initialised Airflow metadata DB. Create it if this is a fresh checkout
    (idempotent: a no-op when the tables already exist)."""
    from airflow.utils import db

    db.initdb()

TINY_MODEL = {
    "feature_names": ["a", "b"],
    "weights": [1.0, -1.0],
    "bias": 0.0,
    "class_labels": {"0": "malignant", "1": "benign"},
    "model_version": "test-1",
}


@pytest.fixture()
def tiny_handler(tmp_path):
    """A freshly imported handler bound to a 2-feature model (fresh cold-start state)."""
    path = tmp_path / "model.json"
    path.write_text(json.dumps(TINY_MODEL))
    return steps.load_handler(path)


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Run extract -> validate -> transform -> train in an isolated data dir."""
    data = tmp_path_factory.mktemp("data")
    ds = "2026-01-01"
    raw = steps.extract(ds, data_dir=data)
    result = steps.validate(raw, ds, data_dir=data)
    processed = steps.transform(raw, ds, data_dir=data)
    model = steps.train(processed, ds, data_dir=data)
    return {"data": data, "ds": ds, "raw": raw, "validation": result, "processed": processed, "model": model}


@pytest.fixture()
def isolated_lambda_dir(tmp_path, monkeypatch):
    """Copy lambda_fn/ so package() tests never touch the committed model.json."""
    dst = tmp_path / "lambda_fn"
    shutil.copytree(config.LAMBDA_DIR, dst)
    monkeypatch.setattr(config, "LAMBDA_DIR", dst)
    monkeypatch.setattr(config, "BUILD_DIR", tmp_path / "build")
    monkeypatch.setattr(config, "EVENTS_DIR", tmp_path / "events")
    return dst
