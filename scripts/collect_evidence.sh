#!/usr/bin/env bash
# Regenerates everything in evidence/ from scratch. Run from the repo root:
#   bash scripts/collect_evidence.sh
# Logs are filtered to the task/pipeline lines (library start-up noise removed) and
# the absolute repo path is shortened to "." so logs are portable. Nothing else is edited.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
REPO="$(pwd)"
source scripts/airflow_env.sh
PY="${PYTHON:-python}"
SAM="${SAM_BIN:-.venv-sam/bin/sam}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-ap-south-1}" SAM_CLI_TELEMETRY=0
EV=evidence; mkdir -p "$EV"

clean() {  # keep timestamped task lines; drop import noise and duplicate-format lines
  grep -E '^\[2026-[0-9-]+T[0-9:.]+\+[0-9]{4}\] \{|AirflowFailException:' "$1" \
    | grep -vE '_docs_decorators|plugins\.py|dagbag\.py|settings\.py|configuration\.py|providers_manager' \
    | sed "s#$REPO/#./#g" | awk '!seen[$0]++'
}

echo "== 1/7 unit + integration tests"
$PY -m pytest tests -v -p no:cacheprovider 2>&1 | grep -E "PASSED|FAILED|ERROR|passed|failed" | sed "s#$REPO/#./#g" > $EV/pytest.log
tail -1 $EV/pytest.log

echo "== 2/7 DAG structure"
airflow db migrate >/dev/null 2>&1
airflow tasks list tumor_pipeline --tree 2>&1 | grep -E "<Task" | sed 's/(_PythonDecoratedOperator)//' > $EV/dag_task_tree.txt
airflow dags show tumor_pipeline --save $EV/dag_graph.png >/dev/null 2>&1 && echo "dag_graph.png written (Airflow-rendered, needs graphviz 'dot')"

echo "== 3/7 DAG run (clean data) + idempotent re-run"
rm -rf data build
airflow dags test tumor_pipeline 2026-11-04 > /tmp/dag_ok.log 2>&1; echo "exit=$?"
clean /tmp/dag_ok.log > $EV/dag_run_success.log
( cd data && find . -type f -exec sha256sum {} + | sort -k2 ) > /tmp/h1.txt
sleep 1
airflow dags test tumor_pipeline 2026-11-04 > /tmp/dag_ok2.log 2>&1; echo "re-run exit=$?"
( cd data && find . -type f -exec sha256sum {} + | sort -k2 ) > /tmp/h2.txt
{ echo "Files hashed after run 1: $(wc -l < /tmp/h1.txt)   after re-run of the same date: $(wc -l < /tmp/h2.txt)";
  if diff -q /tmp/h1.txt /tmp/h2.txt >/dev/null; then echo "RESULT: byte-identical (idempotent)"; else echo "RESULT: DIFFERENT"; diff /tmp/h1.txt /tmp/h2.txt; fi
  echo; cat /tmp/h1.txt; } > $EV/idempotency.txt
head -2 $EV/idempotency.txt
cp data/metrics/ds=2026-11-04/metrics.json $EV/metrics.json
cp data/validation/ds=2026-11-04/result.json $EV/validation_pass.json

echo "== 4/7 DAG run (corrupted data -> gate must block)"
airflow dags test tumor_pipeline 2026-11-05 --conf '{"inject_fault": true}' > /tmp/dag_bad.log 2>&1; echo "exit=$? (non-zero expected)"
clean /tmp/dag_bad.log > $EV/dag_run_gate_blocks.log
cp data/validation/ds=2026-11-05/result.json $EV/validation_fail.json
{ echo "Partitions created per stage for the corrupted date (ds=2026-11-05):";
  for d in raw validation processed models metrics; do printf "  %-11s %s\n" "$d" "$(ls data/$d 2>/dev/null | grep -c 'ds=2026-11-05' | sed 's/1/created/;s/0/NOT created/')"; done; } > $EV/gate_blocked_partitions.txt

echo "== 5/7 SAM template: validate + build"
{ echo '$ cfn-lint template.yaml'; .venv/bin/cfn-lint template.yaml && echo "(no findings)";
  echo; echo '$ sam validate --lint'; $SAM validate --lint 2>&1 | grep -E "valid SAM Template|rror";
  echo; echo '$ sam build'; rm -rf .aws-sam; $SAM build 2>&1 | grep -E "Build (Succeeded|Failed)|Built Artifacts";
  echo; echo '$ ls -l .aws-sam/build/InferenceFunction/'; ls -l .aws-sam/build/InferenceFunction/ | tail -n +2;
  echo; echo "build artefact size: $(du -sb .aws-sam/build/InferenceFunction | cut -f1) bytes"; } > $EV/sam_validate_build.log 2>&1
grep -E "valid SAM|Build" $EV/sam_validate_build.log

echo "== 6/7 local API Gateway emulation (curl)"
PORT=3010
$PY scripts/local_api.py --port $PORT > /tmp/api.log 2>&1 &
APIPID=$!
for i in $(seq 1 30); do curl -s -o /dev/null localhost:$PORT/ && break; sleep 0.3; done
call() { echo "\$ $1"; shift; "$@"; echo; echo; }
{
  call "curl -X POST localhost:$PORT/predict -d @events/benign_body.json     # 1st request = cold start" \
       curl -s -w '\nHTTP %{http_code}' -X POST localhost:$PORT/predict -d @events/benign_body.json
  call "curl -X POST localhost:$PORT/predict -d @events/malignant_body.json  # warm" \
       curl -s -w '\nHTTP %{http_code}' -X POST localhost:$PORT/predict -d @events/malignant_body.json
  call "curl -X POST localhost:$PORT/predict -d '{\"features\":[1,2,3]}'      # invalid input" \
       curl -s -w '\nHTTP %{http_code}' -X POST localhost:$PORT/predict -d '{"features":[1,2,3]}'
  call "curl localhost:$PORT/predict                                         # route not defined" \
       curl -s -w '\nHTTP %{http_code}' localhost:$PORT/predict
} > $EV/api_invocations.log 2>&1
kill $APIPID 2>/dev/null; wait $APIPID 2>/dev/null
grep -E "^HTTP" $EV/api_invocations.log | tr '\n' ' '; echo
sed "s#$REPO/#./#g" /tmp/api.log | grep -E '^(INFO|WARNING|ERROR) \{"request_id"' > $EV/lambda_logs.log

echo "== 7/7 benchmark"
$PY scripts/benchmark.py > /dev/null 2>&1 && echo "benchmark.json written"
echo "done -> $EV/"
