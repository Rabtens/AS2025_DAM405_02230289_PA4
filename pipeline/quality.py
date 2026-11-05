"""Data-quality gate built on Great Expectations (GX Core 1.x).

``run_quality_gate`` validates a DataFrame against an expectation suite and
returns a plain dict. The DAG task raises when ``success`` is False, which stops
every downstream task (transform, train, evaluate, deploy).
"""

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd

from pipeline.config import TARGET_COLUMN


def build_suite(feature_columns, n_min=400, n_max=1000) -> "gx.ExpectationSuite":
    expectations = [
        # Schema: exactly the columns we trained on, nothing missing or extra.
        gxe.ExpectTableColumnsToMatchSet(column_set=list(feature_columns) + [TARGET_COLUMN]),
        # Volume: guards against empty / truncated extracts.
        gxe.ExpectTableRowCountToBeBetween(min_value=n_min, max_value=n_max),
        # Label sanity.
        gxe.ExpectColumnValuesToNotBeNull(column=TARGET_COLUMN),
        gxe.ExpectColumnValuesToBeInSet(column=TARGET_COLUMN, value_set=[0, 1]),
        gxe.ExpectColumnMeanToBeBetween(column=TARGET_COLUMN, min_value=0.4, max_value=0.8),
        # Domain checks on key physical measurements.
        gxe.ExpectColumnValuesToBeBetween(column="mean radius", min_value=3, max_value=40),
        gxe.ExpectColumnValuesToBeBetween(column="mean area", min_value=50, max_value=3000),
    ]
    # Completeness + non-negativity for every feature (all are physical measurements).
    for col in feature_columns:
        expectations.append(gxe.ExpectColumnValuesToNotBeNull(column=col))
        expectations.append(gxe.ExpectColumnValuesToBeBetween(column=col, min_value=0))
    return gx.ExpectationSuite(name="tumor_raw_data_suite", expectations=expectations)


def run_quality_gate(df: pd.DataFrame, feature_columns) -> dict:
    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas("raw_source")
    asset = source.add_dataframe_asset("raw_asset")
    batch_def = asset.add_batch_definition_whole_dataframe("whole_df")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})
    suite = build_suite(feature_columns)
    result = batch.validate(suite)

    failures = []
    for r in result.results:
        if not r.success:
            cfg = r.expectation_config
            failures.append(
                {
                    "expectation": cfg.type,
                    "column": cfg.kwargs.get("column"),
                    "rule": {
                        k: v
                        for k, v in cfg.kwargs.items()
                        if k in ("min_value", "max_value", "value_set", "column_set")
                    },
                    "observed": {
                        k: v
                        for k, v in (r.result or {}).items()
                        if k in ("observed_value", "unexpected_count", "unexpected_percent")
                    },
                }
            )
    stats = result.statistics
    return {
        "success": bool(result.success),
        "evaluated_expectations": int(stats["evaluated_expectations"]),
        "successful_expectations": int(stats["successful_expectations"]),
        "unsuccessful_expectations": int(stats["unsuccessful_expectations"]),
        "n_rows": int(len(df)),
        "failures": failures,
    }
