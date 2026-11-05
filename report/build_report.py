#!/usr/bin/env python3
"""Build report/report.pdf. Every number comes from evidence/ or is computed here."""

import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (HRFlowable, Image, KeepTogether, PageBreak, Paragraph,
                                Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle)
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parents[1]
EV, FIG = ROOT / "evidence", ROOT / "report" / "figs"
OUT = ROOT / "report" / "report.pdf"

# ------------------------------------------------------------------ evidence
metrics = json.loads((EV / "metrics.json").read_text())
bench = json.loads((EV / "benchmark.json").read_text())
vpass = json.loads((EV / "validation_pass.json").read_text())
vfail = json.loads((EV / "validation_fail.json").read_text())
ok_log = (EV / "dag_run_success.log").read_text().splitlines()
bad_log = (EV / "dag_run_gate_blocks.log").read_text().splitlines()
pytest_log = (EV / "pytest.log").read_text().strip().splitlines()
n_tests = int(re.search(r"(\d+) passed", pytest_log[-1]).group(1))
m = re.search(r"train=(\d+) val=(\d+) test=(\d+)", "\n".join(ok_log))
n_train, n_val, n_test = map(int, m.groups())
y = load_breast_cancer().target
n_mal, n_ben = int((y == 0).sum()), int((y == 1).sum())
chosen_C = re.search(r"chose C=([\d.]+)", "\n".join(ok_log)).group(1)


def ts(line):  # "[2026-11-04T17:39:57.090+0600]" -> seconds of day
    t = re.match(r"\[\d{4}-\d\d-\d\dT(\d\d):(\d\d):([\d.]+)", line)
    return int(t.group(1)) * 3600 + int(t.group(2)) * 60 + float(t.group(3))


run_seconds = ts([l for l in ok_log if "DagRun Finished" in l][0]) - ts([l for l in ok_log if "created dagrun" in l][0])

acc, n = metrics["accuracy"], metrics["n_test"]
k = round(acc * n)
z = 1.96
wilson_lo = (acc + z * z / (2 * n) - z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n))) / (1 + z * z / n)
wilson_hi = (acc + z * z / (2 * n) + z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n))) / (1 + z * z / n)
cmx = metrics["confusion_matrix"]

# AWS Lambda list prices, us-east-1, from aws.amazon.com/lambda/pricing (accessed 4 Nov 2026)
P_REQ, P_GBS_X86, MEM_GB, FREE_REQ, FREE_GBS = 0.20 / 1e6, 0.0000166667, 128 / 1024, 1e6, 400_000


def lambda_cost(reqs, billed_ms=1.0, free=False):
    req_c, gbs = reqs * P_REQ, reqs * (billed_ms / 1000) * MEM_GB
    if free:
        req_c, gbs = max(0, reqs - FREE_REQ) * P_REQ, max(0, gbs - FREE_GBS)
    return req_c + gbs * P_GBS_X86


# ------------------------------------------------------------------ styles
NAVY, BLUE, ORANGE, GREY = "#16213e", "#2a78d6", "#eb6834", "#52514e"
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], keepWithNext=1, fontSize=13.5, spaceBefore=12, spaceAfter=3, textColor=colors.HexColor(NAVY))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], keepWithNext=1, fontSize=10.8, spaceBefore=7, spaceAfter=2, textColor=colors.HexColor("#0f3460"))
BODY = ParagraphStyle("B", parent=ss["Normal"], fontSize=9.6, leading=14, spaceAfter=5, alignment=TA_JUSTIFY)
BUL = ParagraphStyle("BU", parent=BODY, leftIndent=14, firstLineIndent=-9, spaceAfter=2.5, alignment=0)
CAP = ParagraphStyle("C", parent=ss["Normal"], fontSize=8.2, leading=11, alignment=TA_CENTER, spaceAfter=8,
                     textColor=colors.HexColor(GREY), fontName="Helvetica-Oblique")
SM = ParagraphStyle("S", parent=BODY, fontSize=8.6, leading=12, spaceAfter=3, alignment=0)
TC = ParagraphStyle("TC", parent=ss["Normal"], fontSize=8.2, leading=10.5)
TCB = ParagraphStyle("TCB", parent=TC, fontName="Helvetica-Bold", textColor=colors.white)
CODE = ParagraphStyle("CODE", fontName="Courier", fontSize=6.9, leading=8.6, backColor=colors.HexColor("#f4f4f2"),
                      borderPadding=4, leftIndent=4, rightIndent=4, spaceAfter=7, spaceBefore=2)
TITLE = ParagraphStyle("T", parent=ss["Title"], fontSize=19, leading=23, textColor=colors.HexColor(NAVY), spaceAfter=4)
SUB = ParagraphStyle("Sub", parent=ss["Normal"], fontSize=11, alignment=TA_CENTER, textColor=colors.HexColor(GREY), spaceAfter=3)

P = lambda t: Paragraph(t, BODY)
B = lambda t: Paragraph("&bull;&nbsp; " + t, BUL)
SP = lambda h=6: Spacer(1, h)
HR = lambda c="#bbbbbb", t=0.6: HRFlowable(width="100%", thickness=t, color=colors.HexColor(c), spaceBefore=2, spaceAfter=4)


def sec(t):
    rule = HR(NAVY, 0.9)
    rule.keepWithNext = 1
    return [Paragraph(t, H1), rule]


def table(rows, widths, header=True, align_right_from=None):
    data = [[Paragraph(str(c), TCB if (header and i == 0) else TC) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cfcfcb")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)))
    for i in range(1, len(rows)):
        st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#eef3fb") if i % 2 else colors.white))
    t.setStyle(TableStyle(st))
    return t


def img(path, width_cm, cap=None):
    w, h = PILImage.open(path).size
    items = [Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)]
    if cap:
        items.append(Paragraph(cap, CAP))
    return KeepTogether(items)


def nice(line):
    """Compact an Airflow log line for the report (content unchanged)."""
    mm = re.match(r"^\[\d{4}-\d\d-\d\dT([\d:.]+)\+\d{4}\] \{[^}]*\} (\w+) - (.*)$", line)
    if not mm:
        return line.strip()[:300]
    t, lvl, msg = mm.groups()
    s = re.match(r"Marking task as (\w+)\. dag_id=\S+, task_id=(\w+),", msg)
    if s:
        msg = f"task {s.group(2):<16} -> {s.group(1)}"
    msg = msg.replace("./data/", "data/")
    out = f"{t[:12]} {lvl:<7} {msg}"
    return out if len(out) <= 300 else out[:296] + " ..."


def block(lines):
    import textwrap
    out = []
    for l in lines:  # wrap to the page width instead of clipping; content is unchanged
        out += textwrap.wrap(l, width=108, subsequent_indent="    ", break_long_words=True,
                             break_on_hyphens=False, drop_whitespace=False) or [""]
    return Preformatted("\n".join(out), CODE)


# ------------------------------------------------------------------ content
S = []
S += [SP(10), HR(NAVY, 2), SP(4), Paragraph("DAM405 &mdash; Machine Learning Operations", SUB),
      Paragraph("Programming Assignment 4", TITLE),
      Paragraph("Serverless Model Deployment &amp; Pipeline Orchestration on AWS", SUB), SP(2), HR(NAVY, 2), SP(6)]
S.append(table([
    ["Student", "[Your name] &nbsp;&nbsp; [Student ID]"],
    ["Repository", "[INSERT REPOSITORY URL]"],
    ["Environment", "AWS Lambda + API Gateway + CloudWatch defined as code (AWS SAM). <b>Validated and built locally; "
                    "not deployed to a real AWS account.</b> Invocation was emulated with a small local API shim "
                    "(<font name='Courier'>scripts/local_api.py</font>), not SAM local or LocalStack. "
                    "Airflow 2.11.2 ran locally via <font name='Courier'>airflow dags test</font>."],
], [3 * cm, 14 * cm], header=False))
S.append(SP(4))

# 1 ---------------------------------------------------------------
S += sec("1. Introduction")
S.append(P(
    "<b>Problem.</b> Deploying a model is not the end of an ML project: the model must be served cheaply and "
    "reliably, and it must be rebuilt repeatedly from data that may be wrong. This project packages a classifier "
    "as a serverless function and wraps an automated pipeline around it, with two checkpoints that stop bad data or a "
    "weak model before anything is deployed."))
S.append(P(
    f"<b>Dataset.</b> Wisconsin Diagnostic Breast Cancer (569 samples, 30 numeric tumour measurements, "
    f"{n_mal} malignant / {n_ben} benign), bundled with scikit-learn so the pipeline runs offline. It was chosen because it is small, "
    f"well-understood and numeric, which keeps the focus on <i>MLOps</i> rather than modelling. It is an educational dataset; "
    f"the model is not suitable for clinical use."))
S.append(P(
    f"<b>Approach.</b> An Apache Airflow DAG runs <i>extract &rarr; validate &rarr; transform &rarr; train &rarr; evaluate &rarr; "
    f"package &rarr; deploy</i>. A Great Expectations suite gates the data ({vpass['evaluated_expectations']} expectations) and an accuracy "
    f"threshold gates the model. A logistic-regression model is exported to a small JSON file and served by a standard-library-only "
    f"AWS Lambda handler behind an API Gateway HTTP API, all defined in one SAM template."))

# 2 ---------------------------------------------------------------
S += sec("2. Architecture")
S.append(img(FIG / "fig_architecture.png", 17,
             "Figure 1: Workflow. Orange-outlined tasks are the two gates. Dashed boxes are cloud services that were "
             "emulated or stood in for locally (see the legend under the figure)."))
S.append(P(
    "The design separates two concerns that change at different speeds. The <b>pipeline</b> (top) runs on a schedule, "
    "produces a versioned model artefact, and decides whether it is good enough. The <b>serving path</b> (bottom) is a tiny, "
    "stateless function that only needs the artefact and a request. The only coupling is the file <font name='Courier'>model.json</font> "
    "(which carries a version string such as <font name='Courier'>" + metrics["model_version"] + "</font>), and every response "
    "reports the version that produced it, so predictions are traceable to the pipeline run that built the model."))

# 3 ---------------------------------------------------------------
S += sec("3. Serverless Design")
S += [Paragraph("3.1 The handler", H2)]
S.append(P(
    "<font name='Courier'>lambda_fn/handler.py</font> uses only the Python standard library. The training step folds the "
    "scaler into the logistic-regression weights, "
    "<font name='Courier'>z = b - sum(w<sub>i</sub>&middot;mu<sub>i</sub>/s<sub>i</sub>) + sum((w<sub>i</sub>/s<sub>i</sub>)&middot;x<sub>i</sub>)</font>, "
    "so inference on raw feature values is one dot product and a sigmoid. A unit test checks that this matches scikit-learn's "
    "<font name='Courier'>predict_proba</font> to within the 6-decimal rounding of the response. Design points:"))
S += [
    B("<b>Load once.</b> The model is read at import time (Lambda's INIT phase), so warm invocations only compute the prediction."),
    B("<b>Two event shapes.</b> It accepts an API Gateway / function-URL proxy event (JSON in <font name='Courier'>body</font>, optionally base64) "
      "and a direct invocation payload. Single (<font name='Courier'>features</font>) and batch (<font name='Courier'>instances</font>, max 100) requests, "
      "as a list or as named features."),
    B("<b>Defensive.</b> Wrong length, non-numeric, boolean, NaN/Infinity, empty or malformed bodies return HTTP 400 with a reason; unexpected "
      "exceptions return a generic 500 and are logged, never leaked to the caller. The sigmoid is numerically stable for extreme inputs."),
    B("<b>Observable.</b> One structured JSON log line per request (request id, status, latency, cold-start flag), which CloudWatch Logs can query."),
]
S += [Paragraph("3.2 Trigger and deployment as code", H2)]
S.append(P(
    "The function is triggered by <font name='Courier'>POST /predict</font> on an API Gateway <b>HTTP API</b>. "
    "<font name='Courier'>template.yaml</font> (AWS SAM, a CloudFormation transform) declares the function, the route, a log group with "
    "14-day retention, and the endpoint URL as an output. Choices that are deliberate rather than defaults:"))
S += [
    B("<b>arm64 (Graviton), 128 MB, 5 s timeout.</b> The model is a few KB and inference takes microseconds, so more memory buys nothing; "
      "a short timeout fails fast. Memory should be re-tuned if the model grows."),
    B("<b><font name='Courier'>AutoPublishAlias: live</font>.</b> Every deployment publishes an immutable version behind an alias, enabling instant rollback "
      "and, later, canary traffic shifting."),
    B("<b>Explicit log group.</b> Lambda's auto-created log group never expires logs; declaring it makes retention part of the code."),
    B("<b>Parameters</b> (<font name='Courier'>Stage</font>, <font name='Courier'>LogRetentionDays</font>) so one template serves dev/staging/prod. "
      "<b>No credentials anywhere</b>: SAM and boto3 use the standard AWS credential chain, and a test scans the repository for key patterns."),
]
S.append(P(
    "The template passes <font name='Courier'>cfn-lint</font> with no findings and <font name='Courier'>sam validate --lint</font>, and "
    "<font name='Courier'>sam build</font> succeeds (8,282-byte build artefact). Deploying is <font name='Courier'>sam build &amp;&amp; sam deploy</font>; "
    "the DAG's last task does this when <font name='Courier'>DEPLOY_TARGET=aws</font> and otherwise runs a local smoke test."))

# 4 ---------------------------------------------------------------
S += sec("4. Airflow DAG")
S.append(img(EV / "dag_graph.png", 17, "Figure 2: DAG graph rendered by Airflow itself (<font name='Courier'>airflow dags show</font>)."))
S.append(table([
    ["Task", "What it does", "Output (partition <font name='Courier'>ds=&lt;date&gt;</font>)"],
    ["extract_data", "Loads the dataset and lands it as a raw CSV.", "raw/tumor.csv"],
    ["<b>validate_data</b>", f"<b>Gate 1.</b> Great Expectations suite, {vpass['evaluated_expectations']} expectations: exact schema, row count 400&ndash;1000, "
                             "label in {0,1} and not null, label mean 0.4&ndash;0.8, per-feature non-null and non-negative, plausible ranges for radius and area. "
                             "Failure raises <font name='Courier'>AirflowFailException</font>.", "validation/result.json"],
    ["transform_data", "Stratified 70/15/15 split; scaler fitted on the training split only (no leakage).", "processed/*.csv, scaler.json"],
    ["train_model", "Chooses C on the validation split, fits, exports Lambda-ready JSON.", "models/model.json"],
    ["<b>evaluate_model</b>", f"<b>Gate 2.</b> Scores the test split <i>through the real handler code</i>; fails below {metrics['min_accuracy_required']:.0%} accuracy.", "metrics/metrics.json"],
    ["package_model", "Promotes the model into <font name='Courier'>lambda_fn/</font>, builds a byte-reproducible zip, writes sample events.", "build/function.zip, events/*.json"],
    ["deploy_function", "<font name='Courier'>sam build/deploy</font> (aws) or local smoke test.", "-"],
], [3.0 * cm, 10.1 * cm, 3.9 * cm]))
S.append(SP(5))
S.append(P(
    "<b>Why the gate sits where it does.</b> <font name='Courier'>validate_data</font> is the only direct child of "
    "<font name='Courier'>extract_data</font> and an ancestor of every other task (checked by a test), so there is no path to training or "
    "deployment that bypasses it. <font name='Courier'>package_model</font> takes the evaluation result as an input, which makes the accuracy gate a "
    "real dependency, not just an ordering."))
S += [Paragraph("Best-practice compliance", H2)]
S += [
    B("<b>Idempotent.</b> Each task writes only to its own <font name='Courier'>ds=&lt;logical date&gt;</font> partition. Re-running the same date produced "
      "<b>byte-identical output for all 9 artefacts</b> (SHA-256 comparison, <font name='Courier'>evidence/idempotency.txt</font>). "
      "The zip uses fixed timestamps so even the package hash is stable."),
    B("<b>Atomic.</b> Files are written to a temporary name and moved into place with <font name='Courier'>os.replace</font>; a failure leaves no partial file "
      "(tested). One task does one unit of work and passes small values (paths, metrics), not data."),
    B("<b>No secrets in code.</b> Settings come from environment variables or an Airflow Variable (<font name='Courier'>DEPLOY_TARGET</font>)."),
    B("<b>Fast parsing, sensible defaults.</b> Heavy imports happen inside tasks; <font name='Courier'>catchup=False</font>, "
      "<font name='Courier'>max_active_runs=1</font>, retries on every task, but gates use <font name='Courier'>AirflowFailException</font> so a "
      "<i>correct</i> rejection is not pointlessly retried."),
    B(f"<b>Testable.</b> Step logic lives in an importable <font name='Courier'>pipeline/</font> package; the DAG file is thin wrappers. "
      f"{n_tests} automated tests pass, including DAG structure and five separate gate-failure scenarios."),
]

# 5 ---------------------------------------------------------------
S += sec("5. AWS Services and MLOps Roles")
S.append(table([
    ["Service", "MLOps role", "Status in this submission"],
    ["AWS Lambda", "Serverless inference compute", "Handler built (<font name='Courier'>sam build</font>) and exercised locally; <b>not deployed</b>"],
    ["Amazon API Gateway (HTTP API)", "Public, managed trigger and routing for inference", "Defined in the template; <b>emulated</b> by <font name='Courier'>local_api.py</font>"],
    ["AWS CloudFormation (via SAM)", "Infrastructure as Code; repeatable, reviewable deployment", "Template linted, validated and built; stack <b>not created</b>"],
    ["Amazon CloudWatch Logs", "Observability: request logs, latency and error tracking", "Log group defined; JSON log lines <b>shown on stdout</b>"],
    ["Amazon S3", "Artefact and model store (versioned, event-capable)", "<b>Stood in for</b> by the local filesystem; production target"],
    ["IAM", "Least-privilege execution role", "SAM generates the basic execution role; <b>not exercised</b>"],
    ["Amazon MWAA", "Managed Airflow for scheduling and orchestration", "<b>Not used</b>; Airflow ran locally (the DAG is portable)"],
], [4.0 * cm, 6.2 * cm, 6.8 * cm]))
S.append(Paragraph("Table 1: Service-to-role mapping. The right-hand column states honestly what was real versus emulated.", CAP))

# 6 ---------------------------------------------------------------
S += sec("6. Results and Evidence")
S += [Paragraph("6.1 Successful DAG run", H2)]
S.append(P(
    f"The full run completed all 7 tasks in about {run_seconds:.1f} s (Airflow's own timestamps, local sandbox). "
    f"Excerpt from <font name='Courier'>evidence/dag_run_success.log</font>:"))
keep = [l for l in ok_log if re.search(r"validate:|Great Expectations|chose C=|evaluate:|Marking task as SUCCESS|DagRun Finished", l)]
shown = [nice(l) for l in keep if "DagRun Finished" not in l] + ["... DagRun Finished: state=success"]
S.append(block(shown))
S += [Paragraph("6.2 The data-quality gate blocks bad data", H2)]
S.append(P(
    "To prove the gate works, the DAG was triggered with a deliberate fault (5% of <i>mean radius</i> set to null, 10 <i>mean area</i> values set to "
    f"&minus;1). {vfail['unsuccessful_expectations']} of {vfail['evaluated_expectations']} expectations failed. The task failed immediately, "
    "the command exited with code 1, and <b>no downstream partition was created</b> (processed, models and metrics are all missing for that date):"))
bl = [nice(l) for l in bad_log if re.search(r"validate:|Quality gate FAILED|AirflowFailException|Immediate failure", l)]
S.append(block(bl))
rows = [["Failed expectation", "Column", "Offending rows"]]
for f in vfail["failures"]:
    rule = ", ".join(f"{a}={b:g}" for a, b in f["rule"].items()) or "no nulls allowed"
    rows.append([f["expectation"].replace("expect_column_values_to_", "").replace("_", " "), f"{f['column']} ({rule})",
                 f"{f['observed']['unexpected_count']} ({f['observed']['unexpected_percent']:.1f}%)"])
S.append(table(rows, [5.2 * cm, 7.6 * cm, 4.2 * cm]))
S.append(SP(4))

S += [Paragraph("6.3 Serverless invocation", H2)]
S.append(P(
    "The sample events were generated by the pipeline from real held-out test rows. The requests below went over HTTP to the local API shim, which builds an "
    "API Gateway v2 event and calls the real <font name='Courier'>handler.py</font>. The first request is a cold start; the 400 and 404 show that bad input and "
    "undefined routes are handled as API Gateway/Lambda would."))
api = (EV / "api_invocations.log").read_text().strip()
api_lines = []
for l in api.splitlines():
    api_lines.append(l)
S.append(block(api_lines))
lam = [l for l in (EV / "lambda_logs.log").read_text().splitlines() if l.strip()]
S.append(P("Structured log lines emitted by the handler for those three calls (the CloudWatch Logs equivalent):"))
S.append(block(lam))

S += [Paragraph("6.4 Model quality", H2)]
S.append(img(FIG / "fig_results.png", 17,
             "Figure 3: Left: confusion matrix on the held-out test split. Right: local cost of the cold path for the chosen design versus "
             "serving the same model with scikit-learn (20 fresh processes each; local timings, <b>not</b> AWS cold-start numbers)."))
S.append(table([
    ["Metric", "Value", "Note"],
    ["Split (train / val / test)", f"{n_train} / {n_val} / {n_test}", "stratified, seed 42"],
    ["Chosen C (validation accuracy)", f"{chosen_C} (0.9882)", "C=0.1, 1 and 10 tied; the most regularised tied value wins"],
    ["Test accuracy", f"{acc:.4f} ({k}/{n})", f"95% Wilson interval {wilson_lo:.3f} &ndash; {wilson_hi:.3f}"],
    ["Macro precision / recall / F1", f"{metrics['precision_macro']:.4f} / {metrics['recall_macro']:.4f} / {metrics['f1_macro']:.4f}", "measured through the deployed handler code"],
    ["ROC AUC", f"{metrics['roc_auc']:.4f}", ""],
    ["Errors", f"{cmx[0][1]} malignant predicted benign; {cmx[1][0]} benign predicted malignant", "the two false negatives are the costlier error type"],
], [5.0 * cm, 5.6 * cm, 6.4 * cm]))
S.append(SP(4))
S.append(P(
    f"<b>What this shows.</b> The model clears the {metrics['min_accuracy_required']:.0%} gate, but with only {n} test rows the plausible accuracy range "
    f"is wide ({wilson_lo:.0%}&ndash;{wilson_hi:.0%}), so the exact figure should not be over-interpreted. The pipeline's value is not this single number but that it "
    f"is <i>reproducible</i> (identical re-runs), <i>guarded</i> (corrupt data never reaches training) and <i>traceable</i> (every prediction carries the model version)."))

# 7 ---------------------------------------------------------------
S += sec("7. Serverless Trade-offs")
S.append(table([
    ["Dimension", "What the evidence and pricing show", "Mitigation / decision here"],
    ["<b>Cost</b>",
     f"Pay per request and per GB-second; nothing when idle. At the us-east-1 x86 list prices ($0.20 per 1M requests, $0.0000166667 per GB-second) and "
     f"an assumed 1 ms billed at 128 MB, 1M requests cost about ${lambda_cost(1e6):.2f} and 100M about ${lambda_cost(1e8):.2f} "
     f"(compute is only ${lambda_cost(1e8) - 1e8 * P_REQ:.2f} of that: <b>the per-request fee dominates</b>). The monthly free tier (1M requests, "
     f"400,000 GB-s) would cover the first 1M. API Gateway is billed separately and is excluded.",
     "arm64 is priced lower per GB-second (not quantified here). At sustained high, steady load an always-on service can become cheaper; the break-even "
     "depends on utilisation and was not measured."],
    ["<b>Cold starts</b>",
     f"A new execution environment must start the runtime and import code. Locally, importing this handler took a median {bench['handler_cold']['import_ms']['median']:.1f} ms "
     f"(p95 {bench['handler_cold']['import_ms']['p95']:.1f}) versus {bench['sklearn_cold']['import_ms']['median']:,.0f} ms (p95 {bench['sklearn_cold']['import_ms']['p95']:,.0f}) "
     f"for the same model through scikit-learn, with a {bench['handler_package_zip_bytes']/1000:.1f} KB package versus ~{bench['sklearn_stack_installed_mb']:.0f} MB of libraries. "
     f"Warm invocations took a median {bench['warm_invoke_us']['p50']:.0f}  microseconds in-process. AWS adds sandbox creation and code download that cannot be reproduced locally.",
     "Minimise package size and imports (the main lever, applied here). If a latency SLO demanded it: provisioned concurrency. Cold-start time also counts toward "
     "billed duration for managed runtimes, per a third-party summary of AWS pricing."],
    ["<b>Scaling</b>",
     "Concurrency scales automatically, one request per execution environment, subject to account concurrency quotas and a burst rate. No capacity planning; "
     "no fleet to patch.",
     "Quotas must be checked before a traffic spike; throttling returns errors to callers. Not load-tested here."],
    ["<b>Limits and fit</b>",
     "Short-lived, stateless, CPU-only functions with bounded memory and package size. Large models, GPU inference or long jobs do not fit.",
     "Fits because the model is KB-sized. A deep model would need another target (container service, SageMaker endpoint)."],
], [2.3 * cm, 9.2 * cm, 5.5 * cm]))
S.append(Paragraph("Table 2: Trade-offs. Cost figures are an illustration from list prices and an assumed duration, not a bill.", CAP))
S += [Paragraph("Future trend: serverless, scale-to-zero serving for larger and generative models", H2)]
S.append(P(
    "Today's functions suit small CPU models, but the main MLOps direction is serving much larger models without paying for idle accelerators. Scale-to-zero "
    "GPU endpoints and managed model APIs apply the same economics (pay per use, no fleet) to heavy inference, while shifting the hard problems to "
    "cold-start of multi-gigabyte weights, evaluation of non-deterministic output, and cost control. The practices built here (versioned artefacts, quality "
    "gates before deploy, reproducible runs, traceable predictions) carry over unchanged; what changes is that the 'quality gate' becomes continuous "
    "evaluation against a test set rather than a single accuracy threshold."))

# 8 ---------------------------------------------------------------
S += sec("8. Reflection, Limitations and Next Steps")
S.append(P("<b>Limitations (stated plainly).</b>"))
S += [
    B("<b>Not deployed to AWS.</b> No AWS account or Docker daemon was available. Everything up to deployment is verified (lint, validate, build, handler tests), "
      "but cloud behaviour (IAM, API Gateway integration, real cold starts, real cost) is untested. The local API shim is my own emulator and is not equivalent to SAM local or LocalStack."),
    B("<b>Airflow was run with <font name='Courier'>airflow dags test</font></b>, which executes tasks in-process. The scheduler, executor and web UI were not exercised, so there are no UI screenshots."),
    B(f"<b>Small, static dataset.</b> {n} test rows give a wide confidence interval. Data never changes, so the quality gate was proven on injected faults, not natural drift."),
    B("<b>Gate thresholds are judgement calls</b> (0.93 accuracy; feature ranges). For a medical task the false-negative rate should be gated explicitly; here "
      f"{cmx[0][1]} malignant cases were missed."),
    B("<b>No API authorisation, throttling or monitoring of drift</b>; the model is baked into the package, so updating it needs a redeploy."),
    B("<b>Great Expectations</b> was used with an ephemeral context, so validation results are saved as JSON but no browsable data docs are produced."),
]
S.append(P("<b>What I would improve next.</b>"))
S += [
    B("Deploy to a real account with SAM, capture CloudWatch evidence, measure true cold starts and run a load test; delete the stack afterwards."),
    B("Store artefacts and a model registry in S3; let the function fetch the model version from an alias so a model update does not require a code deploy."),
    B("CI/CD (e.g. GitHub Actions with OIDC, no stored keys): run tests and <font name='Courier'>cfn-lint</font> on every push; use CodeDeploy canary traffic shifting on the "
      "<font name='Courier'>live</font> alias with automatic rollback on error alarms."),
    B("Run Airflow on MWAA or a container with a real scheduler; add alerting on gate failures."),
    B("Add monitoring: input-distribution drift against the training statistics, plus a false-negative-rate gate and a larger evaluation set."),
]

# 9 ---------------------------------------------------------------
S += sec("9. Declarations and References")
S.append(P(
    "<b>AI-use declaration.</b> This project was developed with the assistance of an AI tool (Claude, by Anthropic), which helped write and test the code and draft this report. "
    "All code was run and all figures in this report were generated from the logs in <font name='Courier'>evidence/</font>. [Edit this statement to match your module's policy.]"))
refs = [
    "AWS. <i>AWS Lambda Developer Guide</i>. https://docs.aws.amazon.com/lambda",
    "AWS. <i>AWS Serverless Application Model (SAM) Developer Guide</i>. https://docs.aws.amazon.com/serverless-application-model",
    "AWS. <i>AWS Lambda pricing</i>, us-east-1 list prices. https://aws.amazon.com/lambda/pricing/ (accessed 4 Nov 2026).",
    "Wiz. <i>AWS Lambda cost breakdown for 2026</i> (INIT-phase billing summary). https://www.wiz.io/academy/cloud-cost/aws-lambda-cost-breakdown",
    "Apache Software Foundation. <i>Apache Airflow documentation</i> (2.11). https://airflow.apache.org/docs",
    "Great Expectations. <i>GX Core documentation</i>. https://docs.greatexpectations.io",
    "Street, W. N., Wolberg, W. H. &amp; Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. <i>SPIE</i> 1905. "
    "(Dataset, via scikit-learn: Pedregosa et al. (2011), <i>JMLR</i> 12.)",
    "Gift, N. &amp; Deza, A. (2021). <i>Practical MLOps</i>. O'Reilly.",
]
for i, r in enumerate(refs, 1):
    S.append(Paragraph(f"[{i}] {r}", SM))


def footer(canvas, doc_):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(2 * cm, 1.1 * cm, "DAM405 PA4 - Serverless model deployment and pipeline orchestration")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc_.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.9 * cm,
                        title="DAM405 PA4 - Serverless Model Deployment & Pipeline Orchestration", author="[Your name]")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
