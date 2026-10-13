"""Final evaluation of the production model.

Run: ``python -m src.evaluation.evaluate``

* test metrics with 95 % bootstrap confidence intervals
* paired bootstrap significance tests vs. baselines (PR-AUC, ROC-AUC)
* reliability (calibration) data, probability distributions, confusion matrix
* full error analysis (src.evaluation.error_analysis)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from src.evaluation.error_analysis import analyse
from src.evaluation.metrics import bootstrap_ci, compute_metrics, paired_bootstrap_delta
from src.features.build import structured_frame
from src.training.data import load_with_split
from src.utils.config import load_config
from src.utils.io import read_json, write_json
from src.utils.logging_utils import get_logger, log_event

log = get_logger("evaluation")


def run() -> dict:
    cfg = load_config()
    mdir = cfg.path("metrics")
    meta = read_json(cfg.path("models") / "model_metadata.json")
    bench = read_json(mdir / "benchmark.json")
    preds = pd.read_csv(mdir / "predictions.csv")
    tp = pd.read_csv(mdir / "test_probabilities.csv")
    df = load_with_split()
    thr = meta["threshold"]
    test = preds[preds.split == "test"]
    y, p = test.y_true.to_numpy(), test.proba.to_numpy()
    metrics = compute_metrics(y, p, thr)
    ci = bootstrap_ci(y, p, thr, n_boot=1000, seed=cfg["seed"])

    ytp = df.set_index("tool_id").loc[tp.tool_id, "error_occurred"].to_numpy()
    best_id = meta["model_id"]
    comparisons = []
    for r in bench["results"]:
        if r["id"] == best_id or r["model_key"] == "dummy":
            continue
        comparisons.append({"vs": r["id"], "vs_name": r["name"],
                            "pr_auc": paired_bootstrap_delta(ytp, tp[best_id], tp[r["id"]], "pr_auc", 1000, cfg["seed"]),
                            "roc_auc": paired_bootstrap_delta(ytp, tp[best_id], tp[r["id"]], "roc_auc", 1000, cfg["seed"])})
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    hist_bins = np.linspace(0, 1, 21)
    dist = {"bins": hist_bins.round(3).tolist(),
            "positive": np.histogram(p[y == 1], hist_bins)[0].tolist(), "negative": np.histogram(p[y == 0], hist_bins)[0].tolist()}
    all_p = preds.proba.to_numpy()
    levels = cfg["risk_levels"]
    bands = sorted(levels.items(), key=lambda kv: kv[1])
    risk_counts = {}
    for i, (name, lo) in enumerate(bands):
        hi = bands[i + 1][1] if i + 1 < len(bands) else 101
        risk_counts[name] = int(((all_p * 100 >= lo) & (all_p * 100 < hi)).sum())

    local = read_json(mdir / "shap_local.json") if (mdir / "shap_local.json").exists() else {}
    st = structured_frame(df)
    st.insert(0, "tool_id", df["tool_id"].values)
    st = st.drop(columns=[c for c in ("parameter_count", "nested_schema_depth") if c in df.columns and c in st.columns])
    ea = analyse(df, preds, local, st)

    out = {"model": meta["model_name"], "model_id": best_id, "threshold": thr, "test_metrics": metrics, "bootstrap_ci_95": ci,
           "paired_tests": comparisons, "calibration": {"mean_predicted": mean_pred.round(4).tolist(), "fraction_positive": frac_pos.round(4).tolist()},
           "probability_distribution": dist, "risk_level_counts_all_tools": risk_counts,
           "confusion_matrix": [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]}
    write_json(out, mdir / "evaluation.json")
    write_json(ea, mdir / "error_analysis.json")
    log_event(log, "evaluation complete", test_f1=round(metrics["f1"], 4), test_pr_auc=round(metrics["pr_auc"], 4),
              fn=metrics["fn"], fp=metrics["fp"])
    return out


if __name__ == "__main__":
    run()
