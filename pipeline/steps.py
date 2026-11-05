"""Pipeline step implementations (pure Python, no Airflow imports).

Every step takes the logical date ``ds`` (YYYY-MM-DD) and reads/writes only inside
its own ``ds=<date>`` partition, using atomic writes. Re-running a step for the
same ``ds`` therefore produces identical output (idempotent) and never leaves a
partial file behind (atomic).
"""

import importlib.util
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pipeline import config
from pipeline.io_utils import (
    sha256_file,
    write_bytes_atomic,
    write_csv_atomic,
    write_json_atomic,
)
from pipeline.quality import run_quality_gate

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 1. extract
# --------------------------------------------------------------------------- #
def extract(ds: str, data_dir=None, inject_fault: bool = False) -> str:
    """Pull the source dataset and land it as a raw, date-partitioned CSV.

    Source: the Wisconsin Diagnostic Breast Cancer dataset bundled with
    scikit-learn (569 rows, 30 numeric features). Bundling keeps the pipeline
    runnable offline; in production this task would read from S3 or a database.

    ``inject_fault`` corrupts the extract on purpose so the quality gate's
    blocking behaviour can be demonstrated. It is off by default.
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()  # 30 features + 'target'
    if inject_fault:
        rng = np.random.default_rng(0)
        bad = rng.choice(len(df), size=int(0.05 * len(df)), replace=False)
        df.loc[bad, "mean radius"] = np.nan  # missing values
        df.loc[bad[:10], "mean area"] = -1.0  # impossible negative area
    out = config.partition(data_dir, "raw", ds) / "tumor.csv"
    write_csv_atomic(df, out)
    log.info("extract: wrote %d rows x %d cols to %s", *df.shape, out)
    return str(out)


# --------------------------------------------------------------------------- #
# 2. validate (data-quality gate)
# --------------------------------------------------------------------------- #
def validate(raw_path: str, ds: str, data_dir=None) -> dict:
    """Run the Great Expectations suite and persist the result for audit."""
    data_dir = Path(data_dir or config.DATA_DIR)
    df = pd.read_csv(raw_path)
    features = [c for c in df.columns if c != config.TARGET_COLUMN]
    result = run_quality_gate(df, features)
    out = config.partition(data_dir, "validation", ds) / "result.json"
    write_json_atomic(result, out)
    log.info(
        "validate: %d/%d expectations passed (success=%s) -> %s",
        result["successful_expectations"],
        result["evaluated_expectations"],
        result["success"],
        out,
    )
    return result


# --------------------------------------------------------------------------- #
# 3. transform
# --------------------------------------------------------------------------- #
def transform(raw_path: str, ds: str, data_dir=None) -> str:
    """Stratified 70/15/15 split; scaler is fitted on the training split only."""
    data_dir = Path(data_dir or config.DATA_DIR)
    df = pd.read_csv(raw_path)
    X, y = df.drop(columns=[config.TARGET_COLUMN]), df[config.TARGET_COLUMN]
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=config.RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=config.RANDOM_SEED
    )
    scaler = StandardScaler().fit(X_train)  # no leakage from val/test
    out_dir = config.partition(data_dir, "processed", ds)

    def _scaled(Xs, ys):
        d = pd.DataFrame(scaler.transform(Xs), columns=X.columns)
        d[config.TARGET_COLUMN] = ys.to_numpy()
        return d

    write_csv_atomic(_scaled(X_train, y_train), out_dir / "train.csv")
    write_csv_atomic(_scaled(X_val, y_val), out_dir / "val.csv")
    write_csv_atomic(_scaled(X_test, y_test), out_dir / "test.csv")
    raw_test = X_test.copy()
    raw_test[config.TARGET_COLUMN] = y_test.to_numpy()
    write_csv_atomic(raw_test, out_dir / "test_raw.csv")  # original units, for evaluation/smoke test
    write_json_atomic(
        {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "feature_names": list(X.columns)},
        out_dir / "scaler.json",
    )
    log.info("transform: train=%d val=%d test=%d -> %s", len(X_train), len(X_val), len(X_test), out_dir)
    return str(out_dir)


# --------------------------------------------------------------------------- #
# 4. train
# --------------------------------------------------------------------------- #
def train(processed_dir: str, ds: str, data_dir=None) -> str:
    """Select C on the validation split, fit, and export a Lambda-ready model.json.

    The scaler is folded into the weights, so the handler can score *raw*
    feature values with a single dot product and no scikit-learn dependency:
        z = b - sum(w_i * mu_i / s_i) + sum((w_i / s_i) * x_i)
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    pdir = Path(processed_dir)
    train_df, val_df = pd.read_csv(pdir / "train.csv"), pd.read_csv(pdir / "val.csv")
    scaler = json.loads((pdir / "scaler.json").read_text())
    feats = scaler["feature_names"]
    Xtr, ytr = train_df[feats], train_df[config.TARGET_COLUMN]
    Xva, yva = val_df[feats], val_df[config.TARGET_COLUMN]

    best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        clf = LogisticRegression(C=C, max_iter=2000, random_state=config.RANDOM_SEED).fit(Xtr, ytr)
        acc = accuracy_score(yva, clf.predict(Xva))
        log.info("train: C=%-5s val_accuracy=%.4f", C, acc)
        if best is None or acc > best[0]:  # first (most regularised) wins ties
            best = (acc, C, clf)
    val_acc, C, clf = best

    mu, sd = np.array(scaler["mean"]), np.array(scaler["scale"])
    w_scaled, b_scaled = clf.coef_[0], float(clf.intercept_[0])
    w_raw = w_scaled / sd
    b_raw = b_scaled - float(np.sum(w_scaled * mu / sd))

    model = {
        "feature_names": feats,
        "weights": w_raw.tolist(),
        "bias": b_raw,
        "class_labels": config.CLASS_LABELS,
        "metadata": {
            "algorithm": "LogisticRegression",
            "C": C,
            "validation_accuracy": round(val_acc, 6),
            "n_train": int(len(train_df)),
            "train_data_sha256": sha256_file(pdir / "train.csv"),
            "sklearn_version": sklearn.__version__,
            "partition": ds,
        },
    }
    digest = sha256_hex_of(model)
    model["model_version"] = f"{ds}-{digest[:8]}"
    out = config.partition(data_dir, "models", ds) / "model.json"
    write_json_atomic(model, out)
    log.info("train: chose C=%s (val_acc=%.4f); exported %s (version %s)", C, val_acc, out, model["model_version"])
    return str(out)


def sha256_hex_of(obj) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# 5. evaluate
# --------------------------------------------------------------------------- #
def load_handler(model_path: str):
    """Import lambda_fn/handler.py against a specific model file, i.e. the exact
    code that will run in Lambda, so evaluation == production behaviour."""
    spec = importlib.util.spec_from_file_location("lambda_handler_under_test", config.LAMBDA_DIR / "handler.py")
    module = importlib.util.module_from_spec(spec)
    previous = os.environ.get("MODEL_PATH")
    os.environ["MODEL_PATH"] = str(model_path)
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            os.environ.pop("MODEL_PATH", None)
        else:
            os.environ["MODEL_PATH"] = previous
    return module


def evaluate(model_path: str, processed_dir: str, ds: str, data_dir=None) -> dict:
    """Score the held-out test split through the real handler code path."""
    data_dir = Path(data_dir or config.DATA_DIR)
    handler = load_handler(model_path)
    test = pd.read_csv(Path(processed_dir) / "test_raw.csv")
    y_true = test[config.TARGET_COLUMN].to_numpy()
    X = test.drop(columns=[config.TARGET_COLUMN])
    event = {"body": json.dumps({"instances": X.values.tolist()})}
    resp = handler.lambda_handler(event, None)
    assert resp["statusCode"] == 200, resp
    preds = json.loads(resp["body"])["predictions"]
    y_pred = np.array([p["label_id"] for p in preds])
    p_benign = np.array([p["probability_benign"] for p in preds])

    metrics = {
        "n_test": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro")), 6),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro")), 6),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 6),
        "roc_auc": round(float(roc_auc_score(y_true, p_benign)), 6),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "model_version": handler.MODEL_VERSION,
        "min_accuracy_required": config.MIN_TEST_ACCURACY,
    }
    metrics["passed_threshold"] = metrics["accuracy"] >= config.MIN_TEST_ACCURACY
    out = config.partition(data_dir, "metrics", ds) / "metrics.json"
    write_json_atomic(metrics, out)
    log.info("evaluate: accuracy=%.4f f1_macro=%.4f auc=%.4f passed=%s", metrics["accuracy"],
             metrics["f1_macro"], metrics["roc_auc"], metrics["passed_threshold"])
    return metrics


# --------------------------------------------------------------------------- #
# 6. package + deploy
# --------------------------------------------------------------------------- #
def _deterministic_zip(files: dict, out: Path) -> Path:
    """Zip with fixed timestamps so identical inputs give a byte-identical archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname in sorted(files):
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, Path(files[arcname]).read_bytes())
    return write_bytes_atomic(buf.getvalue(), out)


def package(model_path: str, processed_dir: str, ds: str) -> dict:
    """Promote the model into lambda_fn/, build a deployable zip, write sample events."""
    model_dst = config.LAMBDA_DIR / "model.json"
    write_bytes_atomic(Path(model_path).read_bytes(), model_dst)

    zip_path = _deterministic_zip(
        {"handler.py": config.LAMBDA_DIR / "handler.py", "model.json": model_dst},
        config.BUILD_DIR / f"ds={ds}" / "function.zip",
    )

    test = pd.read_csv(Path(processed_dir) / "test_raw.csv")
    feats = [c for c in test.columns if c != config.TARGET_COLUMN]
    for cls, name in ((1, "benign"), (0, "malignant")):
        row = test[test[config.TARGET_COLUMN] == cls].iloc[0]
        payload = {"features": [float(row[f]) for f in feats]}
        event = {
            "version": "2.0",
            "routeKey": "POST /predict",
            "rawPath": "/predict",
            "headers": {"content-type": "application/json"},
            "body": json.dumps(payload),
            "isBase64Encoded": False,
        }
        write_json_atomic(event, config.EVENTS_DIR / f"{name}.json")  # for `sam local invoke -e`
        write_json_atomic(payload, config.EVENTS_DIR / f"{name}_body.json")  # for `curl -d @...`
    info = {"zip": str(zip_path), "zip_bytes": zip_path.stat().st_size, "zip_sha256": sha256_file(zip_path)}
    log.info("package: %s (%d bytes)", zip_path, info["zip_bytes"])
    return info


def smoke_test() -> dict:
    """Invoke the promoted handler with the sample events and check the contract."""
    handler = load_handler(config.LAMBDA_DIR / "model.json")
    results = {}
    for name in ("benign", "malignant"):
        event = json.loads((config.EVENTS_DIR / f"{name}.json").read_text())
        resp = handler.lambda_handler(event, None)
        body = json.loads(resp["body"])
        assert resp["statusCode"] == 200, resp
        pred = body["prediction"]
        assert pred["label"] in ("benign", "malignant"), body
        assert 0.0 <= pred["probability_benign"] <= 1.0, body
        results[name] = pred["label"]  # contract check only; accuracy is gated by evaluate()
    return results


def deploy(ds: str, target: str = "local") -> dict:
    """Deploy with SAM when ``target == 'aws'``; otherwise emulate locally.

    Credentials are never read here: ``sam`` resolves them from the standard AWS
    provider chain (env vars, shared profile, or an attached IAM role).
    """
    if target == "aws":
        sam = shutil.which("sam")
        if not sam:
            raise RuntimeError("DEPLOY_TARGET=aws but the SAM CLI is not installed")
        subprocess.run([sam, "build"], check=True, cwd=config.REPO_ROOT)
        subprocess.run(
            [sam, "deploy", "--no-confirm-changeset", "--no-fail-on-empty-changeset"],
            check=True, cwd=config.REPO_ROOT,
        )
        return {"target": "aws", "status": "deployed"}
    results = smoke_test()
    log.info("deploy: target=local (emulated); smoke test passed %s", results)
    return {"target": "local", "status": "smoke-tested", "smoke": results}
