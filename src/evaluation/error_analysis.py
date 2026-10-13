"""Error analysis of the production model on the held-out test split.

* TP / TN / FP / FN example tables with SHAP drivers and an automatic
  "possible reason" derived from the benchmark failure mode and the features.
* Slice analysis (source, observed failure mode, severity, description length,
  parameter count, BFCL category) with per-slice error / FN / FP rates.
* An automated plain-language summary of the dominant sources of mistakes.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

REASONS_FN = {
    "WRONG_TOOL_SELECTION": "Failure came from routing among co-loaded sibling tools; the tool's own documentation looks adequate, so the risk is only partly observable from a single description.",
    "INVALID_ARG_VALUE": "Gold argument values were not stated literally in user requests; documentation gave too little guidance on accepted values, which the model under-weighted.",
    "MISSING_REQUIRED_ARG": "A required argument was omitted; the schema marks it weakly or the description does not mention it.",
    "INVALID_ARG_TYPE": "Untyped or loosely typed parameters led to type errors that the structural features only partly capture.",
    "INVALID_ARG_FORMAT": "Enum/format violations on parameters whose accepted values are not visible in the schema.",
    "HALLUCINATED_PARAMETER": "Agent added parameters on a strict server; this depends on server strictness, which is not visible in the description.",
    "SCHEMA_PARSE_ERROR": "The schema could not be parsed by the client.",
    "NONE": "Borderline case: failure rate just above the error threshold.",
}


def _bucket(s: pd.Series, bins, labels) -> pd.Series:
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True).astype(str)


def _reason_fp(row: pd.Series) -> str:
    bits = []
    if row.get("desc_token_count", 99) < 10:
        bits.append("very short description")
    if row.get("param_described_ratio", 1) < 0.8:
        bits.append("undocumented parameters")
    if row.get("nested_schema_depth", 0) >= 2:
        bits.append("nested schema")
    if row.get("parameter_count", 0) >= 6:
        bits.append("many parameters")
    if row.get("hinted_parameter_ratio", 1) < 0.4:
        bits.append("few format hints")
    base = "Model reacted to " + (", ".join(bits) if bits else "a risky text/semantic profile")
    return base + "; in the benchmark the requests supplied literal values, so the agent still succeeded."


def analyse(df: pd.DataFrame, preds: pd.DataFrame, local: dict, structured: pd.DataFrame | None = None) -> dict:
    d = df.drop(columns=["split"], errors="ignore").merge(preds, on="tool_id", how="inner")
    if structured is not None:
        d = d.merge(structured, on="tool_id", how="left", suffixes=("", "_st"))
    d = d[d["split"] == "test"].copy()
    d["outcome"] = np.select(
        [(d.y_true == 1) & (d.y_pred == 1), (d.y_true == 0) & (d.y_pred == 0), (d.y_true == 0) & (d.y_pred == 1), (d.y_true == 1) & (d.y_pred == 0)],
        ["TP", "TN", "FP", "FN"], default="?")
    d["correct"] = (d.y_true == d.y_pred).astype(int)

    def examples(kind: str, n: int = 25) -> list[dict]:
        sub = d[d.outcome == kind].copy()
        # most confident mistakes / most confident hits first
        sub["_conf"] = np.where(sub.y_pred == 1, sub.proba, 1 - sub.proba)
        sub = sub.sort_values("_conf", ascending=False).head(n)
        rows = []
        for _, r in sub.iterrows():
            loc = local.get(r.tool_id, {})
            if kind == "FN":
                reason = REASONS_FN.get(r.error_type, "")
            elif kind == "FP":
                reason = _reason_fp(r)
            elif kind == "TP":
                reason = f"Correctly flagged; dominant benchmark failure: {r.error_type}."
            else:
                reason = "Correctly predicted as low risk."
            rows.append({"tool_id": r.tool_id, "tool_name": r.tool_name, "server_name": r.server_name, "source": r.source,
                         "description": r.tool_description, "input_schema": r.input_schema, "actual": int(r.y_true),
                         "predicted": int(r.y_pred), "probability": round(float(r.proba), 4), "error_type": r.error_type,
                         "error_severity": r.error_severity, "failure_rate": float(r.failure_rate),
                         "error_type_counts": json.loads(r.error_type_counts) if isinstance(r.error_type_counts, str) else r.error_type_counts,
                         "important_features": loc.get("top", [])[:5], "possible_reason": reason})
        return rows

    d["desc_len_bucket"] = _bucket(d["description_length"], [0, 60, 120, 250, 1e9], ["<60 chars", "60-120", "120-250", ">250"])
    d["param_bucket"] = _bucket(d["parameter_count"], [-1, 1, 3, 6, 1e9], ["0-1", "2-3", "4-6", "7+"])
    slices = {}
    for col in ("source", "error_type", "error_severity", "desc_len_bucket", "param_bucket", "bfcl_category"):
        g = d.groupby(d[col].fillna("mcp_reference").astype(str))
        tab = pd.DataFrame({
            "n": g.size(), "error_rate": 1 - g["correct"].mean(), "positive_rate": g["y_true"].mean(),
            "fn": g.apply(lambda x: int(((x.y_true == 1) & (x.y_pred == 0)).sum()), include_groups=False),
            "fp": g.apply(lambda x: int(((x.y_true == 0) & (x.y_pred == 1)).sum()), include_groups=False),
        }).reset_index().rename(columns={col if col in d else "index": "slice"})
        tab.columns = ["slice", "n", "error_rate", "positive_rate", "fn", "fp"]
        slices[col] = tab.sort_values("n", ascending=False).round(4).to_dict(orient="records")

    counts = d["outcome"].value_counts().to_dict()
    fn = d[d.outcome == "FN"]
    fp = d[d.outcome == "FP"]
    fn_modes = fn["error_type"].value_counts(normalize=True).round(3).to_dict()
    top_fn_mode = next(iter(fn_modes), "NONE")
    worst = max(slices["error_type"], key=lambda r: (r["error_rate"] if r["n"] >= 10 else -1))
    fp_profile = {}
    for c in ("desc_token_count", "param_described_ratio", "parameter_count", "nested_schema_depth"):
        if c in d:
            fp_profile[c] = {"fp_mean": round(float(fp[c].mean()), 3) if len(fp) else None,
                             "tn_mean": round(float(d[d.outcome == "TN"][c].mean()), 3)}
    ctx_share = float(sum(fn_modes.get(m, 0) for m in ("INVALID_ARG_VALUE", "WRONG_TOOL_SELECTION", "HALLUCINATED_PARAMETER")))
    summary = {
        "headline": (f"Most common source of model mistakes: missed failures whose dominant benchmark failure mode is "
                     f"{top_fn_mode.replace('_', ' ').lower()} ({fn_modes.get(top_fn_mode, 0):.0%} of false negatives)."),
        "points": [
            f"Test outcomes — TP {counts.get('TP', 0)}, TN {counts.get('TN', 0)}, FP {counts.get('FP', 0)}, FN {counts.get('FN', 0)}.",
            f"Hardest failure-mode slice: {worst['slice']} (error rate {worst['error_rate']:.0%}, n={worst['n']}).",
            (f"{ctx_share:.0%} of false negatives are failure modes that depend on context outside the tool definition "
             f"(argument values in the user request, sibling-tool routing, server strictness)."),
            (f"False positives vs. true negatives: mean description length {fp_profile.get('desc_token_count', {}).get('fp_mean')} vs "
             f"{fp_profile.get('desc_token_count', {}).get('tn_mean')} tokens; mean parameter count "
             f"{fp_profile.get('parameter_count', {}).get('fp_mean')} vs {fp_profile.get('parameter_count', {}).get('tn_mean')}."),
        ],
        "fn_failure_modes": fn_modes,
        "fp_vs_tn_profile": fp_profile,
    }
    return {"counts": counts, "summary": summary, "slices": slices,
            "examples": {k: examples(k) for k in ("FN", "FP", "TP", "TN")}}
