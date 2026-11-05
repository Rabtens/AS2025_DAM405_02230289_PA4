"""
### tumor_pipeline

extract -> validate (Great Expectations gate) -> transform -> train -> evaluate
-> package -> deploy.

Design rules followed (Airflow best practice)
* **Idempotent**: every task writes only to its own `ds=<logical date>` partition,
  so re-running a task or a whole run for the same date yields identical output.
* **Atomic**: files are written to a temp name then `os.replace`d; a crash never
  leaves a half-written artefact. One task = one unit of work.
* **No secrets in code**: paths/switches come from environment variables or
  Airflow Variables (`DEPLOY_TARGET`). AWS credentials, when deploying, come from
  the standard provider chain (IAM role / profile), never from this repo.
* **Light top level**: heavy imports (pandas, sklearn, great_expectations) happen
  inside tasks so the scheduler can parse this file quickly.
* **Gates stop the line**: a failed quality gate or a model below the accuracy
  threshold raises `AirflowFailException`, which skips retries and blocks every
  downstream task.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from airflow.models import Variable
from airflow.models.param import Param

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `pipeline` importable

log = logging.getLogger("airflow.task")


@dag(
    dag_id="tumor_pipeline",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 11, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "mlops", "retries": 1, "retry_delay": timedelta(minutes=1)},
    params={
        "inject_fault": Param(
            False,
            type="boolean",
            description="DEMO ONLY: corrupt the extract to show the quality gate blocking the run.",
        )
    },
    tags=["mlops", "dam405", "serverless"],
    doc_md=__doc__,
)
def tumor_pipeline():
    @task
    def extract_data(ds=None, params=None) -> str:
        from pipeline import steps

        return steps.extract(ds, inject_fault=bool(params.get("inject_fault", False)))

    @task
    def validate_data(raw_path: str, ds=None) -> str:
        """Data-quality gate. Failure raises AirflowFailException (no retry)."""
        from pipeline import steps

        result = steps.validate(raw_path, ds)
        log.info(
            "Great Expectations: %s/%s expectations passed",
            result["successful_expectations"],
            result["evaluated_expectations"],
        )
        if not result["success"]:
            log.error("Quality gate FAILED. First failures: %s", json.dumps(result["failures"][:5]))
            raise AirflowFailException(
                f"Data-quality gate failed ({result['unsuccessful_expectations']} expectations); "
                "downstream tasks blocked."
            )
        return raw_path

    @task
    def transform_data(validated_raw_path: str, ds=None) -> str:
        from pipeline import steps

        return steps.transform(validated_raw_path, ds)

    @task
    def train_model(processed_dir: str, ds=None) -> str:
        from pipeline import steps

        return steps.train(processed_dir, ds)

    @task
    def evaluate_model(model_path: str, processed_dir: str, ds=None) -> dict:
        """Model-quality gate: refuse to promote a model below the accuracy floor."""
        from pipeline import steps

        metrics = steps.evaluate(model_path, processed_dir, ds)
        log.info("Test metrics: %s", json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}))
        if not metrics["passed_threshold"]:
            raise AirflowFailException(
                f"accuracy {metrics['accuracy']:.4f} < required {metrics['min_accuracy_required']}"
            )
        return metrics

    @task
    def package_model(model_path: str, processed_dir: str, metrics: dict, ds=None) -> dict:
        # `metrics` is an explicit upstream: packaging only happens for an approved model.
        from pipeline import steps

        return steps.package(model_path, processed_dir, ds)

    @task
    def deploy_function(package_info: dict, ds=None) -> dict:
        """`sam build && sam deploy` when DEPLOY_TARGET=aws; local smoke test otherwise."""
        from pipeline import steps

        target = Variable.get("DEPLOY_TARGET", default_var=os.environ.get("DEPLOY_TARGET", "local"))
        log.info("Deploy target: %s (package %s bytes)", target, package_info["zip_bytes"])
        return steps.deploy(ds, target=target)

    raw = extract_data()
    validated = validate_data(raw)
    processed = transform_data(validated)
    model = train_model(processed)
    metrics = evaluate_model(model, processed)
    package = package_model(model, processed, metrics)
    deploy_function(package)


tumor_pipeline()
