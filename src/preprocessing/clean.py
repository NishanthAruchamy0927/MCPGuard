"""Preprocessing stage: validation, repair, de-duplication and quality report.

Run: ``python -m src.data.preprocess`` (alias) or ``python -m src.preprocessing.clean``

Decisions
---------
* **Text** – NFKC + whitespace normalisation; empty descriptions are kept and
  imputed as ``""`` (the absence of a description is itself a risk signal and
  is captured by ``desc_token_count = 0``).
* **Schemas** – parsed with :func:`parse_schema`; malformed schemas are *kept*
  and flagged (``schema_valid = 0``) rather than dropped.
* **Duplicates** – exact duplicates of (name, description, schema) are removed;
  near-duplicates (same name + description) share a ``lineage_id`` and are
  therefore always on the same side of the split.
* **Numeric outliers** – clipped at the 99.5th percentile *only* for reporting;
  tree models are unaffected and linear models use robust scaling.
* **Inconsistent types** – numeric columns are coerced with ``errors='coerce'``
  and imputed with the training-set median inside the feature pipeline.
"""
from __future__ import annotations

import json

import pandas as pd

from src.preprocessing.schema import parse_schema
from src.preprocessing.text import normalize_text
from src.utils.config import load_config, resolve
from src.utils.io import read_jsonl, write_json, write_jsonl
from src.utils.logging_utils import get_logger, log_event

log = get_logger("preprocessing.clean")

REQUIRED_COLUMNS = ["tool_id", "tool_name", "tool_description", "input_schema", "error_occurred", "lineage_id"]


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict = {"input_records": int(len(df))}
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"dataset is missing required columns: {missing_cols}")
    report["missing_values_before"] = {c: int(v) for c, v in df.isna().sum().items() if v}
    df = df.copy()
    df["tool_name"] = df["tool_name"].map(normalize_text)
    df["tool_description"] = df["tool_description"].map(normalize_text)
    df["server_name"] = df.get("server_name", pd.Series(["unknown"] * len(df))).fillna("unknown").map(normalize_text)
    report["empty_descriptions"] = int((df["tool_description"] == "").sum())
    issues = df["input_schema"].map(lambda s: parse_schema(s)[1])
    df["schema_issues"] = issues.map(lambda x: "|".join(x))
    report["schema_issue_counts"] = pd.Series([i for x in issues for i in x]).value_counts().to_dict() if len(issues) else {}
    report["malformed_schemas"] = int(issues.map(lambda x: any(i in ("invalid_json", "schema_not_object", "properties_not_mapping") for i in x)).sum())
    before = len(df)
    df["_dupkey"] = df["tool_name"] + "\x1f" + df["tool_description"] + "\x1f" + df["input_schema"].astype(str)
    df = df.drop_duplicates("_dupkey").drop(columns="_dupkey")
    report["exact_duplicates_removed"] = int(before - len(df))
    near = df.groupby(df["tool_name"].str.lower() + "|" + df["tool_description"].str.lower()).size()
    report["near_duplicate_groups"] = int((near > 1).sum())
    for col in ("error_occurred", "failure_rate", "agent_success_rate", "execution_count"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    n_bad_label = int(df["error_occurred"].isna().sum())
    df = df[df["error_occurred"].notna()].copy()
    df["error_occurred"] = df["error_occurred"].astype(int)
    report["dropped_missing_label"] = n_bad_label
    report["output_records"] = int(len(df))
    report["class_balance"] = df["error_occurred"].value_counts().sort_index().to_dict()
    report["imbalance_ratio"] = round(float((df["error_occurred"] == 0).sum() / max(1, (df["error_occurred"] == 1).sum())), 3)
    return df.reset_index(drop=True), report


def run() -> pd.DataFrame:
    cfg = load_config()
    rows = read_jsonl(resolve(cfg["data"]["dataset_file"]))
    df = pd.DataFrame(rows)
    df["input_schema"] = df["input_schema"].map(lambda s: s if isinstance(s, str) else json.dumps(s))
    clean_df, report = clean(df)
    write_jsonl(clean_df.to_dict(orient="records"), resolve(cfg["data"]["clean_file"]))
    write_json(report, cfg.path("metadata") / "data_quality_report.json")
    log_event(log, "preprocessing complete", records=len(clean_df), duplicates=report["exact_duplicates_removed"],
              malformed=report["malformed_schemas"])
    return clean_df


if __name__ == "__main__":
    run()
