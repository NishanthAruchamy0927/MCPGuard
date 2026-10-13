"""Static report figures (PNG) + projection data for the web dashboard.

Run: ``python -m src.visualization.figures``

Every figure is drawn from artefacts produced by real runs (benchmark.json,
curves.json, evaluation.json, shap_global.json, ablation.json, Optuna trials,
the dataset itself).  Palette: validated categorical order (blue, orange, aqua,
...) and reserved status colours for risk levels (always shown with labels).
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.features.build import documents  # noqa: E402
from src.features.embeddings import EmbeddingEncoder  # noqa: E402
from src.training.data import load_with_split  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import read_json, write_json  # noqa: E402
from src.utils.logging_utils import get_logger, log_event  # noqa: E402

log = get_logger("visualization")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = {"LOW": "#0ca30c", "MEDIUM": "#fab219", "HIGH": "#ec835a", "CRITICAL": "#d03b3b"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f0efec", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGE = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.titlelocation": "left", "axes.labelsize": 10, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9,
        "font.family": "DejaVu Sans", "lines.linewidth": 2, "figure.dpi": 110,
    })


def _save(fig, name: str) -> str:
    cfg = load_config()
    path = cfg.path("figures") / name
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return name


def confusion(ev: dict) -> str:
    cm = np.array(ev["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap=BLUES)
    ax.grid(False)
    labels = ["Success (0)", "Agent error (1)"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    tot = cm.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{cm[i, j] / tot:.1%}", ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > cm.max() * 0.55 else INK, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {ev['model']} (test)")
    return _save(fig, "01_confusion_matrix.png")


def roc_pr(curves: dict, bench: dict) -> list[str]:
    sel = bench["selection"]["selected_id"]
    show = [sel] + [k for k in ("E1", "E2", "E4", "B2", "B6", "B1") if k in curves["models"] and k != sel]
    out = []
    for kind, fname in (("roc", "02_roc_curves.png"), ("pr", "03_pr_curves.png")):
        fig, ax = plt.subplots(figsize=(6, 4.8))
        for i, k in enumerate(show):
            m = curves["models"][k]
            x, yv = (m["roc"]["fpr"], m["roc"]["tpr"]) if kind == "roc" else (m["pr"]["recall"], m["pr"]["precision"])
            auc = m["roc_auc"] if kind == "roc" else m["pr_auc"]
            ax.plot(x, yv, color=SERIES[i], lw=2.6 if k == sel else 1.6, alpha=1 if k == sel else 0.85,
                    label=f"{m['name']} ({auc:.3f})" + ("  ★ selected" if k == sel else ""))
        if kind == "roc":
            ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("ROC curves (test split)")
        else:
            base = float(np.mean(curves["y_test"]))
            ax.axhline(base, ls="--", color=MUTED, lw=1)
            ax.text(0.99, base + 0.015, f"base rate {base:.2f}", color=MUTED, ha="right", fontsize=8)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision–recall curves (test split)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right" if kind == "roc" else "upper right", fontsize=7.5)
        out.append(_save(fig, fname))
    return out


def shap_bar(sg: dict) -> str:
    rows = sg["global"][:15][::-1]
    fam_color = {"text": SERIES[0], "emb": SERIES[1], "schema": SERIES[2], "lexical": SERIES[3], "risk": SERIES[6]}
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.barh([r["display_name"] for r in rows], [r["mean_abs_shap"] for r in rows], height=0.62,
            color=[fam_color[r["family"]] for r in rows])
    ax.set_xlabel("mean |SHAP value| (log-odds), test split")
    ax.set_title("Global feature importance (SHAP)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in fam_color.values()]
    ax.legend(handles, ["Text (LSA)", "Semantic embedding", "Schema", "Lexical / naming", "Risk scores"], loc="lower right")
    return _save(fig, "04_shap_importance.png")


def shap_beeswarm(sg: dict) -> str:
    bees = sg["beeswarm"][:12][::-1]
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    rng = np.random.default_rng(0)
    for i, b in enumerate(bees):
        s = np.array(b["shap"])
        v = np.array(b["value_norm"])
        jitter = rng.normal(0, 0.12, len(s))
        sc = ax.scatter(s, i + jitter, c=v, cmap=DIVERGE, s=9, alpha=0.8, linewidths=0)
    ax.set_yticks(range(len(bees)), [b["display_name"] for b in bees])
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("SHAP value (impact on log-odds of agent error)")
    ax.set_title("SHAP summary — feature value vs. impact")
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("feature value (low → high)", color=INK2)
    cb.outline.set_visible(False)
    ax.grid(axis="y", visible=False)
    return _save(fig, "05_shap_beeswarm.png")


def projections(df: pd.DataFrame, preds: pd.DataFrame) -> tuple[list[str], dict]:
    import umap
    from sklearn.decomposition import PCA
    cfg = load_config()
    enc = EmbeddingEncoder()
    E = enc.encode(documents(df))
    enc.save()
    pca = PCA(n_components=2, random_state=cfg["seed"]).fit(E)
    P = pca.transform(E)
    ucfg = cfg["features"]["umap"]
    U = umap.UMAP(n_neighbors=ucfg["n_neighbors"], min_dist=ucfg["min_dist"], metric=ucfg["metric"],
                  random_state=cfg["seed"]).fit_transform(E)
    pmap = dict(zip(preds.tool_id, preds.proba))
    proba = df["tool_id"].map(pmap).fillna(0).to_numpy()
    y = df["error_occurred"].to_numpy()
    names = []
    for coords, title, fname in ((U, "UMAP", "06_umap_projection.png"),
                                 (P, f"PCA ({pca.explained_variance_ratio_.sum():.1%} var.)", "07_pca_projection.png")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        for lab, color, name in ((0, SERIES[0], "Success"), (1, SERIES[1], "Agent error")):
            m = y == lab
            axes[0].scatter(coords[m, 0], coords[m, 1], s=7, alpha=0.65, color=color, label=name, linewidths=0)
        axes[0].legend(markerscale=2.5)
        axes[0].set_title(f"{title} of tool embeddings — observed outcome")
        sc = axes[1].scatter(coords[:, 0], coords[:, 1], c=proba, cmap=DIVERGE, s=7, alpha=0.75, linewidths=0, vmin=0, vmax=1)
        axes[1].set_title(f"{title} — predicted risk")
        cb = fig.colorbar(sc, ax=axes[1], pad=0.01)
        cb.outline.set_visible(False)
        for a in axes:
            a.set_xticks([])
            a.set_yticks([])
        names.append(_save(fig, fname))
    proj = {"pca_explained_variance": pca.explained_variance_ratio_.round(4).tolist(),
            "points": [{"tool_id": t, "tool_name": n, "server": s, "source": src, "y": int(yy), "proba": round(float(pp), 4),
                        "error_type": et, "umap": [round(float(u[0]), 3), round(float(u[1]), 3)],
                        "pca": [round(float(p[0]), 4), round(float(p[1]), 4)]}
                       for t, n, s, src, yy, pp, et, u, p in zip(df.tool_id, df.tool_name, df.server_name, df.source, y, proba,
                                                                  df.error_type, U, P)]}
    return names, proj


def model_comparison(bench: dict) -> str:
    rows = [r for r in bench["results"] if r["model_key"] != "dummy"]
    rows = sorted(rows, key=lambda r: r["test"]["pr_auc"])
    sel = bench["selection"]["selected_id"]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    yv = np.arange(len(rows))
    h = 0.26
    for j, (met, lab) in enumerate((("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("f1", "F1"))):
        ax.barh(yv + (j - 1) * h, [r["test"][met] for r in rows], height=h - 0.03, color=SERIES[j], label=lab)
    ax.set_yticks(yv, [("★ " if r["id"] == sel else "") + f"{r['id']} {r['name']}" for r in rows], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("score on held-out test split")
    ax.set_title("Model comparison (test)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    return _save(fig, "08_model_comparison.png")


def error_types(df: pd.DataFrame) -> str:
    counts = df["error_type"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(counts.index[::-1].str.replace("_", " ").str.title(), counts.values[::-1], color=SERIES[0], height=0.6)
    for i, v in enumerate(counts.values[::-1]):
        ax.text(v + counts.max() * 0.01, i, str(v), va="center", fontsize=8, color=INK2)
    ax.set_xlabel("tools (dominant failure mode in the controlled benchmark)")
    ax.set_title("Error-type distribution")
    ax.grid(axis="y", visible=False)
    return _save(fig, "09_error_type_distribution.png")


def proba_distribution(ev: dict) -> str:
    d = ev["probability_distribution"]
    bins = np.array(d["bins"])
    centers = (bins[:-1] + bins[1:]) / 2
    w = (bins[1] - bins[0]) * 0.42
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(centers - w / 2, d["negative"], width=w, color=SERIES[0], label="Actual success")
    ax.bar(centers + w / 2, d["positive"], width=w, color=SERIES[1], label="Actual agent error")
    ax.axvline(ev["threshold"], color=INK, ls="--", lw=1)
    ax.text(ev["threshold"] + 0.01, max(d["negative"] + d["positive"]) * 0.95, f"threshold {ev['threshold']:.2f}", fontsize=8, color=INK2)
    ax.set_xlabel("predicted probability of agent error")
    ax.set_ylabel("tools (test)")
    ax.set_title("Prediction probability distribution")
    ax.legend()
    return _save(fig, "10_probability_distribution.png")


def risk_distribution(ev: dict) -> str:
    rc = ev["risk_level_counts_all_tools"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    names = list(rc)
    ax.bar(names, [rc[n] for n in names], color=[STATUS[n] for n in names], width=0.6)
    for i, n in enumerate(names):
        ax.text(i, rc[n], f"{rc[n]}", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_ylabel("tools")
    ax.set_title("Tool-description risk distribution (all tools)")
    ax.grid(axis="x", visible=False)
    return _save(fig, "11_risk_distribution.png")


def ablation_chart(ab: dict) -> str:
    rows = ab["rows"]
    fig, ax = plt.subplots(figsize=(7.4, 4))
    labels = [r["set"].split("_", 1)[0] + " · " + r["label"] for r in rows]
    means = [r["cv"]["pr_auc"]["mean"] for r in rows]
    stds = [r["cv"]["pr_auc"]["std"] for r in rows]
    colors = [SERIES[1] if r["set"].startswith("F") else SERIES[0] for r in rows]
    ax.barh(labels[::-1], means[::-1], xerr=stds[::-1], color=colors[::-1], height=0.6, error_kw={"ecolor": INK2, "lw": 1, "capsize": 3})
    for i, m in enumerate(means[::-1]):
        ax.text(0.01, i, f"{m:.3f}", va="center", color="white", fontsize=8, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_xlabel("PR-AUC (5-fold grouped CV, mean ± sd)")
    ax.set_title("Ablation study — feature families (XGBoost)")
    ax.grid(axis="y", visible=False)
    return _save(fig, "12_ablation_study.png")


def optuna_history() -> str | None:
    cfg = load_config()
    path = cfg.path("experiments") / "optuna_xgboost_trials.csv"
    if not path.exists():
        return None
    t = pd.read_csv(path)
    t = t[t.state == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.scatter(t.trial, t.value, s=22, color=SERIES[0], label="trial CV PR-AUC", zorder=3)
    ax.plot(t.trial, t.value.cummax(), color=SERIES[1], label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel("mean grouped-CV PR-AUC")
    ax.set_title("Optuna optimisation history — XGBoost (TPE)")
    ax.legend()
    return _save(fig, "13_optuna_history.png")


def calibration(ev: dict) -> str:
    c = ev["calibration"]
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1, label="perfect calibration")
    ax.plot(c["mean_predicted"], c["fraction_positive"], marker="o", ms=6, color=SERIES[0], label=ev["model"])
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed error frequency")
    ax.set_title("Reliability diagram (test)")
    ax.legend(loc="upper left")
    return _save(fig, "14_calibration.png")


def run() -> dict:
    cfg = load_config()
    _style()
    mdir = cfg.path("metrics")
    bench = read_json(mdir / "benchmark.json")
    curves = read_json(mdir / "curves.json")
    ev = read_json(mdir / "evaluation.json")
    sg = read_json(mdir / "shap_global.json")
    ab = read_json(mdir / "ablation.json")
    preds = pd.read_csv(mdir / "predictions.csv")
    df = load_with_split()
    made = [confusion(ev), *roc_pr(curves, bench), shap_bar(sg), shap_beeswarm(sg)]
    proj_figs, proj = projections(df, preds)
    made += proj_figs
    made += [model_comparison(bench), error_types(df), proba_distribution(ev), risk_distribution(ev), ablation_chart(ab), calibration(ev)]
    oh = optuna_history()
    if oh:
        made.append(oh)
    write_json(proj, mdir / "projections.json", indent=None)
    write_json({"figures": sorted(made)}, cfg.path("figures") / "index.json")
    log_event(log, "figures generated", n=len(made))
    return {"figures": made}


if __name__ == "__main__":
    run()
