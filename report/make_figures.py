#!/usr/bin/env python3
"""Build report figures from the evidence files (no hand-typed numbers)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
EV, OUT = ROOT / "evidence", ROOT / "report" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# Validated categorical slots 1-2 (scripts/validate_palette.js: ALL CHECKS PASS, light surface)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": MUTED})


def box(ax, x, y, w, h, text, color, dashed=False, bold=False, fs=7.6, fill=None):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
            fc=fill or SURFACE, ec=color, lw=2.2 if bold else 1.4,
            ls=(0, (4, 3)) if dashed else "-",
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=INK, linespacing=1.25)


def arrow(ax, p, q, color=INK2, style="-|>", ls="-", rad=0.0, lw=1.3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=10, color=color,
                                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}"))


def architecture():
    fig, ax = plt.subplots(figsize=(11, 6.1))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 11); ax.set_ylim(-0.4, 6.1); ax.axis("off")

    # ---- lane 1: Airflow -------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.1, 3.75), 10.8, 2.2, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#f2f7fd", ec=BLUE, lw=1.0))
    ax.text(0.28, 5.72, "ORCHESTRATION  -  Apache Airflow DAG  tumor_pipeline  (daily, idempotent, atomic)",
            fontsize=8.6, fontweight="bold", color=BLUE, va="center")
    names = [
        ("extract_data", "sklearn WDBC\n569 x 31 -> raw/"),
        ("validate_data", "Great Expectations\n67 checks\nGATE 1: data"),
        ("transform_data", "70/15/15 split\nscaler fit on\ntrain only"),
        ("train_model", "LogisticRegression\nC chosen on val\n-> model.json"),
        ("evaluate_model", "test metrics via\nthe real handler\nGATE 2: acc >= 0.93"),
        ("package_model", "deterministic zip\nsample events"),
        ("deploy_function", "sam build/deploy\n(aws) or local\nsmoke test"),
    ]
    w, gap, x0, y0, h = 1.27, 0.2, 0.3, 3.95, 1.45  # 7 boxes must fit inside the 10.8-wide lane
    xs = [x0 + i * (w + gap) for i in range(len(names))]
    for i, (n, d) in enumerate(names):
        gate = n in ("validate_data", "evaluate_model")
        box(ax, xs[i], y0, w, h, f"{n}\n\n{d}", ORANGE if gate else BLUE, bold=gate, fs=6.3 if gate else 6.9)
        if i:
            arrow(ax, (xs[i - 1] + w + 0.02, y0 + h / 2), (xs[i] - 0.02, y0 + h / 2), color=BLUE)

    # ---- artifact store --------------------------------------------------
    box(ax, 0.1, 2.62, 10.8, 0.62,
        "ARTIFACT STORE  -  one partition per logical date:  raw/ds=...   validation/ds=...   processed/ds=...   "
        "models/ds=...   metrics/ds=...\n(local filesystem in this submission; Amazon S3 in production)",
        MUTED, dashed=True, fs=7.4, fill="#f6f6f4")
    for i in (0, 1, 2, 3, 4):
        arrow(ax, (xs[i] + w / 2, y0 - 0.02), (xs[i] + w / 2, 3.26), color=MUTED, style="<|-|>", lw=0.9, ls=":")

    # ---- lane 2: serverless --------------------------------------------
    ax.add_patch(FancyBboxPatch((0.1, 0.12), 10.8, 2.18, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#fdf5f1", ec=ORANGE, lw=1.0))
    ax.text(0.28, 2.07, "SERVERLESS INFERENCE  -  as code: template.yaml (AWS SAM)",
            fontsize=8.6, fontweight="bold", color=ORANGE, va="center")
    cy, ch = 0.42, 1.28
    box(ax, 0.3, cy, 1.55, ch, "Client\n\nPOST /predict\nJSON body:\n30 features", INK2, fs=7.2)
    box(ax, 2.5, cy, 2.05, ch, "API Gateway\nHTTP API\n\nroute POST /predict\npayload format 2.0", ORANGE, dashed=True, fs=7.2)
    box(ax, 5.2, cy, 2.75, ch, "AWS Lambda\ntumor-classifier-<stage>\n\npython3.12 - arm64 - 128 MB - 5 s\nhandler.py + model.json (3.4 KB zip)\nstdlib only, model loaded at INIT",
        ORANGE, dashed=True, bold=True, fs=6.6)
    box(ax, 8.6, cy, 2.1, ch, "CloudWatch Logs\n\nJSON log line / request\nretention 14 days", ORANGE, dashed=True, fs=7.2)
    arrow(ax, (1.87, cy + ch / 2), (2.48, cy + ch / 2), color=INK2)
    arrow(ax, (4.57, cy + ch / 2), (5.18, cy + ch / 2), color=INK2)
    arrow(ax, (7.97, cy + ch / 2), (8.58, cy + ch / 2), color=INK2)
    # deploy -> lambda (IaC)
    arrow(ax, (xs[6] + w / 2, y0 - 0.02), (7.55, cy + ch + 0.03), color=ORANGE, rad=-0.12, lw=1.8)
    ax.text(8.15, 3.5, "sam build && sam deploy\n(Infrastructure as Code)", fontsize=7.2, color=ORANGE,
            ha="center", va="center", fontweight="bold", bbox=dict(fc="white", ec="none", pad=1.5))

    ax.text(0.1, -0.22, "Solid box = executed in this submission.   Dashed box = emulated or stood in for locally "
            "(local API shim for API Gateway + Lambda runtime; stdout for CloudWatch; filesystem for S3).",
            fontsize=7.1, color=INK2, va="bottom")
    fig.savefig(OUT / "fig_architecture.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def results():
    m = json.loads((EV / "metrics.json").read_text())
    b = json.loads((EV / "benchmark.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})

    # (a) confusion matrix: sequential single hue, direct labels
    cm = np.array(m["confusion_matrix"])
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#eaf2fc", BLUE])
    a1.imshow(cm, cmap=ramp, vmin=0, vmax=cm.max())
    for (i, j), v in np.ndenumerate(cm):
        a1.text(j, i, str(v), ha="center", va="center", fontsize=18, fontweight="bold",
                color="white" if v > cm.max() * 0.7 else INK)  # dark ink on light/mid blue, white only on the darkest
    a1.set_xticks([0, 1], ["malignant", "benign"]); a1.set_yticks([0, 1], ["malignant", "benign"])
    a1.set_xlabel("predicted", color=INK2); a1.set_ylabel("true", color=INK2)
    a1.set_title(f"Test confusion matrix (n={m['n_test']}, accuracy {m['accuracy']:.1%})", fontsize=9.5, loc="left")
    for s in a1.spines.values():
        s.set_visible(False)
    a1.tick_params(length=0)

    # (b) cold-path import time, log axis, direct labels (two series -> legend by axis labels)
    hc, sc = b["handler_cold"]["import_ms"]["median"], b["sklearn_cold"]["import_ms"]["median"]
    vals, labels, cols = [hc, sc], ["stdlib-only handler\n(this project)", "same model via\nscikit-learn + pickle"], [BLUE, ORANGE]
    y = np.arange(2)[::-1]
    a2.barh(y, vals, color=cols, height=0.5)
    a2.set_xscale("log"); a2.set_xlim(1, 4000)
    a2.set_yticks(y, labels, fontsize=8.5)
    for yi, v, mem in zip(y, vals, [f"{b['handler_package_zip_bytes']/1000:.1f} KB zipped package",
                                    f"{b['sklearn_stack_installed_mb']:.0f} MB installed libraries"]):
        a2.text(v * 1.15, yi + 0.06, f"{v:,.1f} ms", va="center", fontsize=10, fontweight="bold", color=INK)
        a2.text(v * 1.15, yi - 0.2, mem, va="center", fontsize=7.8, color=INK2)
    a2.set_xlabel("median import / INIT time, log scale (ms)", color=INK2)
    a2.set_title(f"Local cold-path cost, {b['n_cold_runs']} fresh processes each", fontsize=9.5, loc="left")
    a2.grid(axis="x", color=GRID, lw=0.8); a2.set_axisbelow(True)
    for s in ("top", "right", "left"):
        a2.spines[s].set_visible(False)
    a2.tick_params(length=0)
    fig.text(0.99, 0.005, "local machine timings - not AWS cold-start numbers", ha="right", fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "fig_results.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    results()
    print("figures written to", OUT)
