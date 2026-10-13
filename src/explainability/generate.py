"""Generate SHAP artefacts for the selected production model.

Run: ``python -m src.explainability.generate``

Writes
    artifacts/metrics/shap_global.json   global importance, family totals, beeswarm data
    artifacts/metrics/shap_local.json    top contributions for every tool (by tool_id)
"""
from __future__ import annotations

import joblib
import numpy as np

from src.explainability.recommendations import recommend
from src.explainability.shap_explain import PipelineExplainer
from src.features.build import FAMILY_LABELS
from src.training.data import load_with_split
from src.utils.config import load_config
from src.utils.io import write_json
from src.utils.logging_utils import get_logger, log_event

log = get_logger("explainability.generate")


def run() -> dict:
    cfg = load_config()
    ecfg = cfg["explainability"]
    pipe = joblib.load(cfg.path("models") / "best_model.joblib")
    df = load_with_split()
    train = df[df["split"] == "train"]
    exp = PipelineExplainer(pipe, background_df=train, background_size=ecfg["shap_background_size"], seed=cfg["seed"])
    X, sv, base = exp.explain_frame(df)
    proba = pipe.named_steps["model"].predict_proba(X)[:, 1]

    test = (df["split"] == "test").to_numpy()
    mean_abs = np.abs(sv[test]).mean(axis=0)
    order = np.argsort(-mean_abs)
    global_rows = [{"feature": exp.names[i], "display_name": exp.display[i], "family": exp.families[i],
                    "family_label": FAMILY_LABELS[exp.families[i]], "mean_abs_shap": float(mean_abs[i]),
                    "mean_shap": float(sv[test, i].mean())} for i in order]
    fam_tot: dict[str, float] = {}
    for r in global_rows:
        fam_tot[r["family"]] = fam_tot.get(r["family"], 0.0) + r["mean_abs_shap"]
    total = sum(fam_tot.values()) or 1.0
    families = [{"family": f, "label": FAMILY_LABELS[f], "mean_abs_shap": v, "share": v / total}
                for f, v in sorted(fam_tot.items(), key=lambda kv: -kv[1])]

    # beeswarm payload: top 15 features, test rows, min-max normalised feature values
    rng = np.random.default_rng(cfg["seed"])
    idx_test = np.where(test)[0]
    sample = rng.choice(idx_test, size=min(len(idx_test), ecfg["shap_sample_size"]), replace=False)
    bees = []
    for i in order[:15]:
        col = X[sample, i]
        lo, hi = np.percentile(X[:, i], [2, 98])
        norm = np.clip((col - lo) / (hi - lo), 0, 1) if hi > lo else np.full_like(col, 0.5)
        bees.append({"feature": exp.names[i], "display_name": exp.display[i], "shap": sv[sample, i].round(4).tolist(),
                     "value_norm": norm.round(3).tolist()})

    local = {}
    struct_cols = [j for j, f in enumerate(exp.families) if f in ("lexical", "schema", "risk")]
    for r, (tid, x_row, s_row) in enumerate(zip(df["tool_id"], X, sv)):
        top = exp.top_contributions(x_row, s_row, ecfg["top_k_features"])
        feats = {exp.names[j].split(":", 1)[1]: float(x_row[j]) for j in struct_cols}
        local[tid] = {"proba": float(proba[r]), "base_value": base, "top": top,
                      "families": exp.family_contributions(s_row), "recommendations": recommend(feats, top, 4)}
    out = {"model": type(exp.model).__name__, "explainer": exp.kind, "base_value": base, "n_explained": int(len(df)),
           "global": global_rows, "families": families, "beeswarm": bees, "lsa_topics": {str(k): v for k, v in exp.topics.items()}}
    mdir = cfg.path("metrics")
    write_json(out, mdir / "shap_global.json")
    write_json(local, mdir / "shap_local.json", indent=None)
    np.savez_compressed(mdir / "shap_values_test.npz", shap=sv[test], X=X[test], names=np.array(exp.names))
    log_event(log, "SHAP generated", explainer=exp.kind, tools=len(df), top_feature=global_rows[0]["display_name"])
    return out


if __name__ == "__main__":
    run()
