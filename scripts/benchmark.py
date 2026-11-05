#!/usr/bin/env python3
"""Measure what the serverless design choices cost, on THIS machine.

These are local timings. They show relative effects (stdlib-only vs scikit-learn
packaging) but are NOT AWS cold-start numbers: real Lambda adds sandbox creation,
code download and runtime start-up that cannot be reproduced locally.
"""

import json
import pickle
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
OUT = REPO_ROOT / "evidence" / "benchmark.json"
N_COLD, N_WARM = 20, 2000

HANDLER_CHILD = """
import json, sys, time
t0 = time.perf_counter()
sys.path.insert(0, {lambda_dir!r})
import handler
t1 = time.perf_counter()
event = json.load(open({event!r}))
resp = handler.lambda_handler(event, None)
t2 = time.perf_counter()
assert resp["statusCode"] == 200
print(json.dumps({{"import_ms": (t1 - t0) * 1000, "first_invoke_ms": (t2 - t1) * 1000}}))
"""

SKLEARN_CHILD = """
import json, pickle, sys, time
t0 = time.perf_counter()
import numpy as np
from sklearn.linear_model import LogisticRegression
model = pickle.load(open({pkl!r}, "rb"))
t1 = time.perf_counter()
event = json.load(open({event!r}))
x = np.array(json.loads(event["body"])["features"]).reshape(1, -1)
model.predict_proba(x)
t2 = time.perf_counter()
print(json.dumps({{"import_ms": (t1 - t0) * 1000, "first_invoke_ms": (t2 - t1) * 1000}}))
"""


def pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def run_child(code):
    t = time.perf_counter()
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    wall = (time.perf_counter() - t) * 1000
    return {**json.loads(out.stdout.strip().splitlines()[-1]), "process_wall_ms": wall}


def summarise(rows, key):
    vals = [r[key] for r in rows]
    return {"median": round(statistics.median(vals), 2), "p95": round(pct(vals, 95), 2), "min": round(min(vals), 2)}


def dir_size_mb(pkg):
    import importlib.util

    spec = importlib.util.find_spec(pkg)
    if not spec or not spec.submodule_search_locations:
        return 0.0
    root = Path(list(spec.submodule_search_locations)[0])
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 1e6


def main():
    event = REPO_ROOT / "events" / "benign.json"
    zip_files = sorted((REPO_ROOT / "build").glob("ds=*/function.zip"))
    zip_bytes = zip_files[-1].stat().st_size if zip_files else None

    # ---- cold path: fresh interpreter each time (handler import == Lambda INIT) ----
    run_child(HANDLER_CHILD.format(lambda_dir=str(REPO_ROOT / "lambda_fn"), event=str(event)))  # warm the FS cache
    handler_rows = [
        run_child(HANDLER_CHILD.format(lambda_dir=str(REPO_ROOT / "lambda_fn"), event=str(event)))
        for _ in range(N_COLD)
    ]

    # ---- comparison: the same model served via scikit-learn + pickle ----
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression

    X, y = load_breast_cancer(return_X_y=True)
    with tempfile.TemporaryDirectory() as tmp:
        pkl = Path(tmp) / "model.pkl"
        pkl.write_bytes(pickle.dumps(LogisticRegression(max_iter=5000).fit(X, y)))
        code = SKLEARN_CHILD.format(pkl=str(pkl), event=str(event))
        run_child(code)
        sk_rows = [run_child(code) for _ in range(N_COLD)]

    # ---- warm path: in-process invocations through the real handler ----
    sys.path.insert(0, str(REPO_ROOT / "lambda_fn"))
    import handler  # noqa: E402

    ev = json.loads(event.read_text())
    handler.lambda_handler(ev, None)  # discard the cold call
    lat = []
    for _ in range(N_WARM):
        t = time.perf_counter_ns()
        handler.lambda_handler(ev, None)
        lat.append((time.perf_counter_ns() - t) / 1000)  # microseconds

    sklearn_mb = sum(dir_size_mb(p) for p in ("numpy", "scipy", "sklearn", "joblib", "threadpoolctl"))
    result = {
        "note": "local machine timings; not AWS cold-start numbers",
        "python": sys.version.split()[0],
        "n_cold_runs": N_COLD,
        "n_warm_invocations": N_WARM,
        "handler_package_zip_bytes": zip_bytes,
        "handler_cold": {
            "import_ms": summarise(handler_rows, "import_ms"),
            "first_invoke_ms": summarise(handler_rows, "first_invoke_ms"),
            "process_wall_ms": summarise(handler_rows, "process_wall_ms"),
        },
        "sklearn_cold": {
            "import_ms": summarise(sk_rows, "import_ms"),
            "first_invoke_ms": summarise(sk_rows, "first_invoke_ms"),
            "process_wall_ms": summarise(sk_rows, "process_wall_ms"),
        },
        "sklearn_stack_installed_mb": round(sklearn_mb, 1),
        "warm_invoke_us": {
            "p50": round(pct(lat, 50), 1),
            "p95": round(pct(lat, 95), 1),
            "p99": round(pct(lat, 99), 1),
            "max": round(max(lat), 1),
        },
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
