# Serverless Model Deployment & Pipeline Orchestration (DAM405 PA4)

A tumour classifier served from **AWS Lambda** (defined as code with **AWS SAM**) and
produced by an **Apache Airflow** pipeline with a **Great Expectations** data-quality gate.

- **Data:** Wisconsin Diagnostic Breast Cancer (569 rows, 30 numeric features), bundled with scikit-learn - the pipeline runs offline.
- **Model:** logistic regression, exported as a small `model.json` so the Lambda handler needs **no third-party libraries**.
- **Flow:** `extract -> validate (GE gate) -> transform -> train -> evaluate (accuracy gate) -> package -> deploy`

Repository link for the report: **`<INSERT YOUR REPOSITORY URL HERE>`**

## What was actually run (be precise in your report)

| Item | Status in the submitted evidence |
|---|---|
| Airflow DAG (`airflow dags test`), 7/7 tasks | Run, logs in `evidence/` |
| Great Expectations gate passes on clean data / blocks corrupted data | Run, both logs in `evidence/` |
| 36 pytest tests | Run, `evidence/pytest.log` |
| `cfn-lint`, `sam validate --lint`, `sam build` | Run, `evidence/sam_validate_build.log` |
| Handler invoked over HTTP via **local API emulator** (`scripts/local_api.py`) | Run, `evidence/api_invocations.log` |
| Deployment to a real AWS account / `sam local invoke` / LocalStack | **Not run** (no AWS account; no Docker daemon in the build sandbox) |
| Airflow web UI / scheduler | **Not run** - DAG executed with `airflow dags test` |

`scripts/local_api.py` is a small emulator written for this project (API Gateway payload-format-2.0
event -> real `handler.py`). It is **not** SAM local or LocalStack - say so in your report.
Local timings in `evidence/benchmark.json` are **not** AWS cold-start numbers.

## Layout

```
lambda_fn/handler.py      Lambda handler (stdlib only)       lambda_fn/model.json  exported model
template.yaml             SAM / CloudFormation (IaC)         samconfig.toml        SAM defaults (no secrets)
dags/tumor_pipeline_dag.py  Airflow DAG                      pipeline/             step logic, GE gate, atomic IO
tests/                    pytest suite                       events/               sample API events
scripts/                  airflow_env.sh, local_api.py, benchmark.py, collect_evidence.sh
evidence/                 logs, metrics, DAG graph           report/                figure + PDF builders
```

## Setup (Python 3.12)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt graphviz matplotlib reportlab cfn-lint \
  --constraint https://raw.githubusercontent.com/apache/airflow/constraints-2.11.2/constraints-3.12.txt
# SAM CLI in its own environment (its dependency pins differ from Airflow's):
python3.12 -m venv .venv-sam && .venv-sam/bin/pip install aws-sam-cli
source scripts/airflow_env.sh && airflow db migrate
```

Always install with the Airflow **constraints file**; without it numpy/pandas can drift and break Airflow.

## Run

```bash
source .venv/bin/activate && source scripts/airflow_env.sh
python -m pytest tests -v                                        # 36 tests
airflow dags test tumor_pipeline 2026-11-04                      # happy path, all 7 tasks
airflow dags test tumor_pipeline 2026-11-05 --conf '{"inject_fault": true}'   # gate must BLOCK (exit 1)
python scripts/local_api.py --port 3000 &                        # local API Gateway stand-in
curl -s -X POST localhost:3000/predict -d @events/benign_body.json
export AWS_DEFAULT_REGION=ap-south-1; .venv-sam/bin/sam validate --lint && .venv-sam/bin/sam build
bash scripts/collect_evidence.sh                                 # regenerate everything in evidence/
python report/make_figures.py && python report/build_report.py   # figures + report.pdf
```

For the Airflow UI (useful for screenshots): `airflow standalone`, open http://localhost:8080,
unpause `tumor_pipeline`, trigger it (use *Trigger DAG w/ config* to try `inject_fault`).

## Deploy to real AWS (optional)

Credentials are **never** stored in this repo; SAM/boto3 use the standard AWS credential chain
(`aws configure sso`, a profile, or an IAM role). Edit `region` in `samconfig.toml` first.

```bash
.venv-sam/bin/sam build && .venv-sam/bin/sam deploy --guided     # first time; prints PredictUrl
curl -s -X POST "<PredictUrl>" -d @events/benign_body.json
.venv-sam/bin/sam logs -n InferenceFunction --stack-name tumor-classifier --tail
.venv-sam/bin/sam delete                                         # remove everything afterwards
```

To let the DAG deploy: `airflow variables set DEPLOY_TARGET aws` (default `local` = smoke test only).
`sam local invoke -e events/benign.json` also works if you have Docker running.

## Design decisions

- **Stdlib-only handler** - scaler folded into the weights, so inference is one dot product. 3.4 KB package, ~8 ms local import, vs ~1 s and ~158 MB to serve the same model with scikit-learn (`evidence/benchmark.json`).
- **Idempotent + atomic tasks** - each task writes only to its `ds=<date>` partition via temp-file + `os.replace`; re-running a date is byte-identical (`evidence/idempotency.txt`).
- **Two gates** - bad data stops at `validate_data`; a model under 93 % test accuracy stops at `evaluate_model`. Both raise `AirflowFailException` (no pointless retries).
- **Evaluation uses the deployed code** - `evaluate_model` scores the test split through `handler.py` itself.
- **No secrets** - paths/switches from env vars or Airflow Variables; a test scans the repo for credential patterns.
- **File permissions** - artefacts are written `0644`; Lambda runs as a different OS user and fails on `0600` files (a real bug caught and covered by a regression test).

## Known limitations

Small static dataset (86 test rows -> wide confidence interval); educational model, **not** for clinical use
(2 of 32 malignant test cases were predicted benign); no API authorizer; model baked into the package
(redeploy to update); `extract` reads a bundled dataset instead of a real source. See the report for more.

## AI-use declaration

This project was developed with assistance from an AI tool (Claude, Anthropic). **Declare this
in your report as required by your module/institutional policy, review every file, and make sure you
can explain each design decision yourself.**
# AS2025_DAM405_02230289_PA4
