import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from pipeline import config, quality, steps
from pipeline.io_utils import sha256_file, write_json_atomic

REPO_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------- data-quality gate ------------------------------ #
def _raw(pipeline_run):
    df = pd.read_csv(pipeline_run["raw"])
    return df, [c for c in df.columns if c != config.TARGET_COLUMN]


def test_gate_passes_on_clean_data(pipeline_run):
    assert pipeline_run["validation"]["success"] is True
    assert pipeline_run["validation"]["unsuccessful_expectations"] == 0


def test_gate_fails_on_nulls(pipeline_run):
    df, feats = _raw(pipeline_run)
    df.loc[:20, "mean radius"] = np.nan
    result = quality.run_quality_gate(df, feats)
    assert result["success"] is False
    assert any(f["column"] == "mean radius" for f in result["failures"])


def test_gate_fails_on_impossible_values(pipeline_run):
    df, feats = _raw(pipeline_run)
    df.loc[:5, "mean area"] = -1.0
    assert quality.run_quality_gate(df, feats)["success"] is False


def test_gate_fails_on_schema_drift(pipeline_run):
    df, feats = _raw(pipeline_run)
    assert quality.run_quality_gate(df.drop(columns=["mean radius"]), feats)["success"] is False


def test_gate_fails_on_truncated_extract(pipeline_run):
    df, feats = _raw(pipeline_run)
    assert quality.run_quality_gate(df.head(50), feats)["success"] is False


def test_gate_fails_on_label_corruption(pipeline_run):
    df, feats = _raw(pipeline_run)
    df.loc[:5, config.TARGET_COLUMN] = 7
    assert quality.run_quality_gate(df, feats)["success"] is False


# ------------------------- model correctness / parity ------------------------- #
def test_handler_matches_sklearn_probabilities(pipeline_run):
    """Folding the scaler into the weights must not change the model's output."""
    proc = Path(pipeline_run["processed"])
    train = pd.read_csv(proc / "train.csv")
    feats = [c for c in train.columns if c != config.TARGET_COLUMN]
    model = json.loads(Path(pipeline_run["model"]).read_text())
    C = model["metadata"]["C"]
    clf = LogisticRegression(C=C, max_iter=2000, random_state=config.RANDOM_SEED).fit(
        train[feats], train[config.TARGET_COLUMN]
    )
    scaler = json.loads((proc / "scaler.json").read_text())
    raw_test = pd.read_csv(proc / "test_raw.csv")[feats]
    scaled = (raw_test.to_numpy() - np.array(scaler["mean"])) / np.array(scaler["scale"])
    expected = clf.predict_proba(pd.DataFrame(scaled, columns=feats))[:, 1]

    handler = steps.load_handler(pipeline_run["model"])
    resp = handler.lambda_handler({"body": json.dumps({"instances": raw_test.values.tolist()})}, None)
    got = np.array([p["probability_benign"] for p in json.loads(resp["body"])["predictions"]])
    assert np.max(np.abs(got - expected)) < 1e-5  # handler rounds to 6 dp


def test_no_test_leakage_in_scaler(pipeline_run):
    proc = Path(pipeline_run["processed"])
    train = pd.read_csv(proc / "train.csv")
    feats = [c for c in train.columns if c != config.TARGET_COLUMN]
    assert train[feats].mean().abs().max() < 1e-9  # train scaled with its own statistics


def test_splits_are_stratified_and_disjoint_sizes(pipeline_run):
    proc = Path(pipeline_run["processed"])
    sizes = {n: len(pd.read_csv(proc / f"{n}.csv")) for n in ("train", "val", "test")}
    assert sum(sizes.values()) == 569
    assert abs(sizes["train"] / 569 - 0.70) < 0.01


# ----------------------------- idempotency / atomicity ------------------------ #
def test_steps_are_idempotent(pipeline_run, tmp_path):
    data, ds = pipeline_run["data"], pipeline_run["ds"]
    before = {p: sha256_file(p) for p in Path(data).rglob("*") if p.is_file()}
    raw = steps.extract(ds, data_dir=data)
    processed = steps.transform(raw, ds, data_dir=data)
    steps.train(processed, ds, data_dir=data)
    after = {p: sha256_file(p) for p in Path(data).rglob("*") if p.is_file()}
    assert before == after  # byte-identical on re-run, no extra files


def test_package_is_idempotent_and_byte_identical(pipeline_run, isolated_lambda_dir):
    a = steps.package(pipeline_run["model"], pipeline_run["processed"], pipeline_run["ds"])
    b = steps.package(pipeline_run["model"], pipeline_run["processed"], pipeline_run["ds"])
    assert a["zip_sha256"] == b["zip_sha256"]


def test_atomic_write_leaves_no_partial_file(tmp_path):
    target = tmp_path / "out" / "x.json"
    with pytest.raises(TypeError):
        write_json_atomic({"bad": object()}, target)  # serialisation fails before replace
    assert not target.exists()
    assert list((tmp_path / "out").glob("*.tmp")) == []


def test_deployable_files_are_world_readable(pipeline_run, isolated_lambda_dir):
    """Regression: Lambda runs as a different user, so 0600 files fail with PermissionError."""
    steps.package(pipeline_run["model"], pipeline_run["processed"], pipeline_run["ds"])
    for f in (isolated_lambda_dir / "model.json", isolated_lambda_dir / "handler.py"):
        assert f.stat().st_mode & 0o044 == 0o044, f"{f.name} is not group/world readable"


def test_smoke_test_passes_with_real_model(pipeline_run, isolated_lambda_dir):
    steps.package(pipeline_run["model"], pipeline_run["processed"], pipeline_run["ds"])
    assert steps.smoke_test() == {"benign": "benign", "malignant": "malignant"}


# --------------------------------- DAG checks --------------------------------- #
def test_dag_imports_cleanly_and_has_expected_structure():
    from airflow.models import DagBag

    bag = DagBag(dag_folder=str(REPO_ROOT / "dags"), include_examples=False)
    assert bag.import_errors == {}
    dag = bag.get_dag("tumor_pipeline")
    assert {t.task_id for t in dag.tasks} == {
        "extract_data", "validate_data", "transform_data", "train_model",
        "evaluate_model", "package_model", "deploy_function",
    }
    assert dag.catchup is False and dag.max_active_runs == 1
    # The quality gate sits directly after extract and upstream of EVERYTHING else.
    assert dag.get_task("validate_data").upstream_task_ids == {"extract_data"}
    assert dag.get_task("validate_data").get_flat_relative_ids(upstream=False) == {
        "transform_data", "train_model", "evaluate_model", "package_model", "deploy_function",
    }
    # Deployment is reachable only through the model-quality gate.
    assert "evaluate_model" in dag.get_task("package_model").upstream_task_ids
    assert all(t.retries >= 1 for t in dag.tasks)


def test_no_hardcoded_secrets_in_repo():
    patterns = [r"AKIA[0-9A-Z]{16}", r"aws_secret_access_key\s*=", r"(?i)password\s*=\s*['\"]\w+", r"(?i)secret_key\s*=\s*['\"]"]
    offenders = []
    for path in list(REPO_ROOT.glob("**/*.py")) + list(REPO_ROOT.glob("**/*.yaml")) + list(REPO_ROOT.glob("**/*.toml")):
        if any(part in {".venv", ".venv-sam", ".airflow", ".aws-sam", "tests"} for part in path.parts):
            continue
        text = path.read_text(errors="ignore")
        offenders += [str(path) for p in patterns if re.search(p, text)]
    assert offenders == []
