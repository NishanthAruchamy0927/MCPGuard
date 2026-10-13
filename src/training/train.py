"""Training orchestrator: experiment matrix, benchmark, ablation, model selection.

Run: ``python -m src.training.train``

Outputs (all regenerated from real runs – nothing is hard-coded):
    models/best_model.joblib            fitted Pipeline(features -> model)
    models/preprocessor.joblib          fitted feature builder of the best model
    models/tfidf_vectorizer.joblib      fitted TF-IDF union
    models/model_metadata.json, feature_schema.json, embedding_config.json, shap_config.json
    artifacts/metrics/benchmark.json    every model x {train, val, test}
    artifacts/metrics/experiment_matrix.json, ablation.json, curves.json
    artifacts/metrics/predictions.csv   best-model probabilities for every tool
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import best_f1_threshold, compute_metrics, curves
from src.features.build import ABLATION_SETS, FAMILY_LABELS, MCPFeatureBuilder
from src.models.registry import DISPLAY_NAMES, TREE_MODELS, make_model
from src.training.data import load_with_split
from src.training.tracking import ExperimentTracker
from src.utils.config import load_config
from src.utils.io import read_json, set_seed, write_json
from src.utils.logging_utils import get_logger, log_event

log = get_logger("training.train")

TFIDF = ("tfidf",)
EMB = ("emb",)
ALL = ABLATION_SETS["F_all_features"]

# Experiment matrix (project brief §56) + full benchmark on fused features (§19)
EXPERIMENTS = [
    # id,  name,                                       model,            families, group
    ("E1", "TF-IDF + Logistic Regression",             "logreg",         TFIDF, "matrix"),
    ("E2", "TF-IDF + Random Forest",                   "random_forest",  TFIDF, "matrix"),
    ("E3", "Semantic Embedding + Random Forest",       "random_forest",  EMB,   "matrix"),
    ("E4", "Semantic Embedding + XGBoost",             "xgboost",        EMB,   "matrix"),
    ("E5", "Combined Features + XGBoost",              "xgboost",        ALL,   "matrix"),
    ("E6", "Optimized Combined Features + XGBoost",    "xgboost_optuna", ALL,   "matrix"),
    ("B0", "Majority / prior baseline",                "dummy",          ALL,   "benchmark"),
    ("B1", "Doc-quality heuristic (rule)",             "doc_quality_rule", ALL, "benchmark"),
    ("B2", "Combined Features + Logistic Regression",  "logreg",         ALL,   "benchmark"),
    ("B3", "Combined Features + Random Forest",        "random_forest",  ALL,   "benchmark"),
    ("B4", "Combined Features + LightGBM",             "lightgbm",       ALL,   "benchmark"),
    ("B5", "Combined Features + Hist. Gradient Boosting", "hist_gb",     ALL,   "benchmark"),
    ("B6", "Combined Features + MLP Neural Network",   "mlp",            ALL,   "benchmark"),
    ("B7", "Optimized Combined Features + LightGBM",   "lightgbm_optuna", ALL,  "benchmark"),
]


def feature_set_label(fams: tuple) -> str:
    if fams == TFIDF:
        return "TF-IDF (word+char)"
    if fams == EMB:
        return "Semantic embedding (MiniLM->PCA)"
    if fams == ALL:
        return "All fused features"
    return " + ".join(FAMILY_LABELS[f] for f in fams)


def tuned_params(model: str) -> dict:
    cfg = load_config()
    base = model.replace("_optuna", "")
    path = cfg.path("models") / f"best_params_{base}.json"
    if not path.exists():
        from src.training.optimize import optimize
        optimize(base)
    return read_json(path)["best_params"]


class _RuleModel:
    """Unsupervised documentation-quality heuristic: risk = 1 - description_quality_score.

    Operationalises the hypothesis of tool-documentation studies (Hsieh et al. 2023;
    Yuan et al. 2024) that poorer documentation lowers tool-use success."""

    def __init__(self, col: int):
        self.col = col

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        q = np.asarray(X[:, self.col]).ravel()
        p = np.clip(1 - q, 0, 1)
        return np.c_[1 - p, p]


def run() -> dict:
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    tracker = ExperimentTracker()
    df = load_with_split()
    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    y = {s: parts[s]["error_occurred"].to_numpy() for s in parts}
    pos_weight = float((y["train"] == 0).sum() / max(1, (y["train"] == 1).sum()))
    log_event(log, "data", **{s: len(p) for s, p in parts.items()}, pos_weight=round(pos_weight, 3))

    builders: dict[tuple, MCPFeatureBuilder] = {}
    mats: dict[tuple, dict] = {}

    def features(fams: tuple):
        if fams not in builders:
            t0 = time.perf_counter()
            fb = MCPFeatureBuilder(families=fams, random_state=seed).fit(parts["train"])
            builders[fams] = fb
            mats[fams] = {s: fb.transform(parts[s]) for s in parts}
            log_event(log, "features built", families="+".join(fams), dim=len(fb.feature_names_), seconds=round(time.perf_counter() - t0, 1))
        return builders[fams], mats[fams]

    results, fitted, probas = [], {}, {}
    for eid, name, model_key, fams, group in EXPERIMENTS:
        fb, X = features(fams)
        params = tuned_params(model_key) if model_key.endswith("_optuna") else {}
        if model_key == "doc_quality_rule":
            clf = _RuleModel(fb.feature_names_.index("risk:description_quality_score"))
        else:
            clf = make_model(model_key, pos_weight=pos_weight, sparse_input=("tfidf" in fams), params=params)
        t0 = time.perf_counter()
        clf.fit(X["train"], y["train"])
        train_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = {s: clf.predict_proba(X[s])[:, 1] for s in parts}
        infer_ms = (time.perf_counter() - t0) / sum(len(v) for v in parts.values()) * 1000
        thr = best_f1_threshold(y["val"], p["val"])
        m = {s: compute_metrics(y[s], p[s], thr) for s in parts}
        row = {"id": eid, "name": name, "model_key": model_key, "model": DISPLAY_NAMES[model_key], "group": group,
               "feature_set": feature_set_label(fams), "families": list(fams), "n_features": len(fb.feature_names_),
               "threshold": thr, "train_seconds": round(train_s, 3), "inference_ms_per_tool": round(infer_ms, 4),
               "params": params, "train": m["train"], "val": m["val"], "test": m["test"]}
        results.append(row)
        fitted[eid] = (fams, clf)
        probas[eid] = p
        tracker.log_run(f"{eid} {name}", DISPLAY_NAMES[model_key], feature_set_label(fams),
                        {**(params or {}), "threshold": round(thr, 4), "n_features": len(fb.feature_names_)},
                        {**{f"val_{k}": v for k, v in m["val"].items() if isinstance(v, float)},
                         **{f"test_{k}": v for k, v in m["test"].items() if isinstance(v, float)}},
                        stage=group, extra={"training_time_s": round(train_s, 3), "inference_ms_per_tool": round(infer_ms, 4)})
        log_event(log, "experiment", id=eid, model=model_key, val_pr_auc=round(m["val"]["pr_auc"], 4),
                  test_pr_auc=round(m["test"]["pr_auc"], 4), test_f1=round(m["test"]["f1"], 4))

    # ------------------------------------------------------- model selection
    eligible = [r for r in results if r["model_key"] not in ("dummy", "doc_quality_rule")]
    key = lambda r: (round(r["val"]["pr_auc"], 3), round(r["val"]["f1"], 3), round(r["val"]["roc_auc"], 3))
    best = max(eligible, key=key)
    ranking = [r["id"] for r in sorted(eligible, key=key, reverse=True)]
    fams, clf = fitted[best["id"]]
    fb = builders[fams]
    pipeline = Pipeline([("features", fb), ("model", clf)])
    models_dir = cfg.path("models")
    joblib.dump(pipeline, models_dir / "best_model.joblib", compress=3)
    joblib.dump(fb, models_dir / "preprocessor.joblib", compress=3)
    if getattr(fb, "tfidf_", None) is not None:
        joblib.dump(fb.tfidf_, models_dir / "tfidf_vectorizer.joblib", compress=3)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    selection = {
        "rule": "max validation PR-AUC, then F1, then ROC-AUC (rounded to 3 d.p.)",
        "selected_id": best["id"], "selected_name": best["name"], "ranking": ranking,
        "reason": (f"{best['name']} achieved the highest validation PR-AUC ({best['val']['pr_auc']:.3f}); "
                   f"F1 {best['val']['f1']:.3f}, ROC-AUC {best['val']['roc_auc']:.3f}. PR-AUC is primary because "
                   f"agent failures are the minority, costly class."),
    }
    metadata = {
        "project": cfg["project"]["name"], "version": cfg["project"]["version"], "trained_at": now,
        "dataset_version": cfg["data"]["dataset_version"], "model_id": best["id"], "model_name": best["model"],
        "model_key": best["model_key"], "experiment_name": best["name"], "feature_set": best["feature_set"],
        "families": list(fams), "n_features": len(fb.feature_names_), "threshold": best["threshold"],
        "is_tree_model": best["model_key"] in TREE_MODELS, "params": best["params"],
        "metrics": {"train": best["train"], "val": best["val"], "test": best["test"]}, "selection": selection,
        "embedding_backend": getattr(fb, "embedding_backend_", None), "positive_class": "agent error (error_occurred = 1)",
        "split_sizes": {s: len(p) for s, p in parts.items()}, "pos_weight": pos_weight,
    }
    write_json(metadata, models_dir / "model_metadata.json")
    write_json({"feature_names": fb.feature_names_, "feature_families": fb.feature_families_,
                "input_fields": {"tool_name": "string (required)", "description": "string (required)",
                                 "input_schema": "JSON object (MCP inputSchema)"}}, models_dir / "feature_schema.json")
    write_json({"backend": getattr(fb, "embedding_backend_", None), "model_name": cfg["features"]["embedding"]["model_name"],
                "pca_components": int(getattr(getattr(fb, "pca_", None), "n_components_", 0) or 0),
                "pca_explained_variance": float(np.sum(fb.pca_.explained_variance_ratio_)) if getattr(fb, "pca_", None) is not None else None,
                "text_svd_components": int(getattr(getattr(fb, "svd_", None), "n_components", 0) or 0),
                "text_svd_explained_variance": float(np.sum(fb.svd_.explained_variance_ratio_)) if getattr(fb, "svd_", None) is not None else None},
               models_dir / "embedding_config.json")
    write_json({"explainer": "tree" if best["model_key"] in TREE_MODELS else "permutation",
                "background_size": cfg["explainability"]["shap_background_size"], "top_k": cfg["explainability"]["top_k_features"]},
               models_dir / "shap_config.json")

    # -------------------------------------------------------------- outputs
    mdir = cfg.path("metrics")
    write_json({"generated_at": now, "selection": selection, "results": results}, mdir / "benchmark.json")
    write_json({"generated_at": now, "rows": [r for r in results if r["group"] == "matrix"]}, mdir / "experiment_matrix.json")
    write_json({"generated_at": now, "y_test": y["test"].tolist(),
                "models": {r["id"]: {"name": r["name"], "model": r["model"], **curves(y["test"], probas[r["id"]]["test"]),
                                      "pr_auc": r["test"]["pr_auc"], "roc_auc": r["test"]["roc_auc"]}
                           for r in results if r["model_key"] != "dummy"}}, mdir / "curves.json")
    pred_rows = []
    for s in parts:
        p = probas[best["id"]][s]
        for i, row in parts[s].iterrows():
            pred_rows.append({"tool_id": row["tool_id"], "split": s, "y_true": int(row["error_occurred"]), "proba": float(p[i]),
                              "y_pred": int(p[i] >= best["threshold"])})
    pd.DataFrame(pred_rows).to_csv(mdir / "predictions.csv", index=False)
    # baseline probabilities on test for paired significance tests
    pd.DataFrame({"tool_id": parts["test"]["tool_id"], **{r["id"]: probas[r["id"]]["test"] for r in results}}).to_csv(
        mdir / "test_probabilities.csv", index=False)

    ablation = run_ablation(df, parts, y, pos_weight, builders, mats, seed)
    write_json({"generated_at": now, **ablation}, mdir / "ablation.json")
    log_event(log, "training complete", best=best["id"], model=best["model"], test_pr_auc=round(best["test"]["pr_auc"], 4),
              test_f1=round(best["test"]["f1"], 4), test_roc_auc=round(best["test"]["roc_auc"], 4))
    return metadata


def run_ablation(df, parts, y, pos_weight, builders, mats, seed) -> dict:
    """Feature-family ablation with XGBoost (fixed config params).

    * ``cv``: 5-fold stratified grouped CV on train+val (builder re-fit per fold)
    * ``test``: trained on train, evaluated on the held-out test split
    """
    cfg = load_config()
    trval = df[df["split"] != "test"].reset_index(drop=True)
    yv = trval["error_occurred"].to_numpy()
    g = trval["lineage_id"].to_numpy()
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(trval, yv, g))
    cv_scores = {k: {"pr_auc": [], "roc_auc": [], "f1": []} for k in ABLATION_SETS}
    for tr, va in folds:
        fb = MCPFeatureBuilder(families=ALL, random_state=seed).fit(trval.iloc[tr])
        Xtr, Xva = fb.transform(trval.iloc[tr]), fb.transform(trval.iloc[va])
        fam = np.array(fb.feature_families_)
        for k, fams in ABLATION_SETS.items():
            cols = np.isin(fam, fams)
            clf = make_model("xgboost", pos_weight=pos_weight).fit(Xtr[:, cols], yv[tr])
            pv = clf.predict_proba(Xva[:, cols])[:, 1]
            m = compute_metrics(yv[va], pv, 0.5)
            for met in cv_scores[k]:
                cv_scores[k][met].append(m[met])
    fb_all = builders[ALL]
    fam_all = np.array(fb_all.feature_families_)
    rows = []
    for k, fams in ABLATION_SETS.items():
        cols = np.isin(fam_all, fams)
        clf = make_model("xgboost", pos_weight=pos_weight).fit(mats[ALL]["train"][:, cols], y["train"])
        pval = clf.predict_proba(mats[ALL]["val"][:, cols])[:, 1]
        thr = best_f1_threshold(y["val"], pval)
        pt = clf.predict_proba(mats[ALL]["test"][:, cols])[:, 1]
        mt = compute_metrics(y["test"], pt, thr)
        rows.append({"set": k, "families": list(fams), "label": " + ".join(FAMILY_LABELS[f] for f in fams), "n_features": int(cols.sum()),
                     "cv": {met: {"mean": float(np.mean(v)), "std": float(np.std(v))} for met, v in cv_scores[k].items()},
                     "test": mt})
        log_event(log, "ablation", set=k, cv_pr_auc=round(float(np.mean(cv_scores[k]["pr_auc"])), 4), test_pr_auc=round(mt["pr_auc"], 4))
    best = max(rows, key=lambda r: r["cv"]["pr_auc"]["mean"])
    singles = [r for r in rows if len(r["families"]) <= 2 and r["set"][0] in "ABC"]
    best_single = max(singles, key=lambda r: r["cv"]["pr_auc"]["mean"])
    return {"model": "XGBoost (config defaults)", "cv_folds": 5, "rows": rows, "best_set": best["set"],
            "fusion_gain_pr_auc_vs_best_single": float(rows[-1]["cv"]["pr_auc"]["mean"] - best_single["cv"]["pr_auc"]["mean"]),
            "best_single_set": best_single["set"]}


def main() -> None:
    run()


if __name__ == "__main__":
    main()
