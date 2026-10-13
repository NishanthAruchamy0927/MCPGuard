"""Report-support generator.

Run: ``python -m src.reporting.report``

Builds markdown/LaTeX tables from the experiment artefacts and fills the IEEE
report templates in ``docs/report/`` (``{{PLACEHOLDER}}`` tokens), so every
number in the report is regenerated whenever the pipeline re-runs.
"""
from __future__ import annotations

import re

import pandas as pd

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.io import read_json, write_json
from src.utils.logging_utils import get_logger, log_event

log = get_logger("reporting")


def _f(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}"


def md_table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def tex_table(rows: list[list], header: list[str], bold_row: int | None = None) -> str:
    def esc(s):
        return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
    lines = [" & ".join(rf"\textbf{{{esc(h)}}}" for h in header) + r" \\ \midrule"]
    for i, r in enumerate(rows):
        cells = [esc(c) for c in r]
        if bold_row is not None and i == bold_row:
            cells = [rf"\textbf{{{c}}}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines)


def run() -> dict:
    cfg = load_config()
    mdir = cfg.path("metrics")
    bench = read_json(mdir / "benchmark.json")
    ab = read_json(mdir / "ablation.json")
    ev = read_json(mdir / "evaluation.json")
    meta = read_json(cfg.path("models") / "model_metadata.json")
    ds = read_json(cfg.path("metadata") / "dataset_summary.json")
    dq = read_json(cfg.path("metadata") / "data_quality_report.json")
    sp = read_json(cfg.path("metadata") / "split_report.json")
    ea = read_json(mdir / "error_analysis.json")
    sg = read_json(mdir / "shap_global.json")
    opt = {m: read_json(cfg.path("models") / f"best_params_{m}.json") for m in ("xgboost", "lightgbm")
           if (cfg.path("models") / f"best_params_{m}.json").exists()}

    sel = bench["selection"]["selected_id"]
    order = sorted(bench["results"], key=lambda r: -r["test"]["pr_auc"])
    header = ["ID", "Model / features", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    brow = [[r["id"] + (" *" if r["id"] == sel else ""), r["name"], *(_f(r["test"][k]) for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"))]
            for r in order]
    mat = [r for r in bench["results"] if r["group"] == "matrix"]
    mrow = [[r["id"], r["name"], _f(r["val"]["pr_auc"]), _f(r["test"]["f1"]), _f(r["test"]["roc_auc"]), _f(r["test"]["pr_auc"]),
             f"{r['train_seconds']:.2f}"] for r in mat]
    arow = [[r["set"].split("_")[0], r["label"], r["n_features"], f"{r['cv']['pr_auc']['mean']:.3f} ± {r['cv']['pr_auc']['std']:.3f}",
             f"{r['cv']['roc_auc']['mean']:.3f}", _f(r["test"]["pr_auc"])] for r in ab["rows"]]
    prow = [[c["vs"], c["vs_name"], f"{c['pr_auc']['mean_delta']:+.3f}", f"[{c['pr_auc']['ci_low']:+.3f}, {c['pr_auc']['ci_high']:+.3f}]",
             f"{c['pr_auc']['p_value_one_sided']:.3f}"] for c in ev["paired_tests"]]

    md = ["# MCPGuard — results tables (auto-generated)", "",
          f"Selected production model: **{meta['experiment_name']}** ({meta['model_name']}), threshold {meta['threshold']:.3f}.", "",
          "## Benchmark (held-out test split)", md_table(brow, header), "",
          "## Experiment matrix", md_table(mrow, ["ID", "Experiment", "Val PR-AUC", "Test F1", "Test ROC-AUC", "Test PR-AUC", "Train s"]), "",
          "## Ablation (XGBoost, 5-fold grouped CV on train+val)", md_table(arow, ["Set", "Families", "#feat", "CV PR-AUC", "CV ROC-AUC", "Test PR-AUC"]), "",
          "## Paired bootstrap: selected model minus baseline (test PR-AUC)", md_table(prow, ["vs", "Baseline", "ΔPR-AUC", "95% CI", "p (one-sided)"]), "",
          "## Error analysis", ea["summary"]["headline"], "", *[f"- {p}" for p in ea["summary"]["points"]]]
    rdir = cfg.path("reports")
    (rdir / "results_tables.md").write_text("\n".join(md), encoding="utf-8")

    tm = ev["test_metrics"]
    ci = ev["bootstrap_ci_95"]
    top_feats = ", ".join(r["display_name"] for r in sg["global"][:5])
    fams = ", ".join(f"{f['label']} {f['share']:.0%}" for f in sg["families"])
    best_single = next(r for r in ab["rows"] if r["set"] == ab["best_single_set"])
    numbers = {
        "N_RECORDS": dq["output_records"], "N_RAW": ds["records"], "N_BFCL": ds["by_source"].get("bfcl_v3", 0),
        "N_MCP": ds["by_source"].get("mcp_reference_servers", 0), "N_AUG": ds["by_source"].get("augmented", 0),
        "N_SERVERS": ds["servers"], "N_LINEAGES": ds["lineages"], "POS_RATE": f"{ds['positive_rate']:.1%}",
        "MEAN_FAIL": f"{ds['mean_failure_rate']:.1%}", "N_DUP": dq["exact_duplicates_removed"], "N_MALFORMED": dq["malformed_schemas"],
        "N_TRAIN": sp["train"]["n"], "N_VAL": sp["val"]["n"], "N_TEST": sp["test"]["n"],
        "MODEL": meta["model_name"], "EXPERIMENT": meta["experiment_name"], "N_FEATURES": meta["n_features"],
        "THRESHOLD": _f(meta["threshold"]), "F1": _f(tm["f1"]), "ROC_AUC": _f(tm["roc_auc"]), "PR_AUC": _f(tm["pr_auc"]),
        "ACC": _f(tm["accuracy"]), "PREC": _f(tm["precision"]), "REC": _f(tm["recall"]), "MCC": _f(tm["mcc"]),
        "PR_CI": f"[{ci['pr_auc']['low']:.3f}, {ci['pr_auc']['high']:.3f}]", "F1_CI": f"[{ci['f1']['low']:.3f}, {ci['f1']['high']:.3f}]",
        "ROC_CI": f"[{ci['roc_auc']['low']:.3f}, {ci['roc_auc']['high']:.3f}]",
        "TP": tm["tp"], "TN": tm["tn"], "FP": tm["fp"], "FN": tm["fn"],
        "ABL_ALL": _f(ab["rows"][-1]["cv"]["pr_auc"]["mean"]), "ABL_BEST_SINGLE": _f(best_single["cv"]["pr_auc"]["mean"]),
        "ABL_BEST_SINGLE_NAME": best_single["label"], "ABL_GAIN": f"{ab['fusion_gain_pr_auc_vs_best_single']:+.3f}",
        "TOP_FEATURES": top_feats, "FAMILY_SHARES": fams, "EA_HEADLINE": ea["summary"]["headline"],
        "OPT_XGB_TRIALS": opt.get("xgboost", {}).get("n_trials", "n/a"), "OPT_LGB_TRIALS": opt.get("lightgbm", {}).get("n_trials", "n/a"),
        "OPT_XGB_CV": _f(opt["xgboost"]["best_value_cv_pr_auc"]) if "xgboost" in opt else "n/a",
        "RULE_PR_AUC": _f(next(r for r in bench["results"] if r["id"] == "B1")["test"]["pr_auc"]),
        "LR_TFIDF_PR_AUC": _f(next(r for r in bench["results"] if r["id"] == "E1")["test"]["pr_auc"]),
        "DUMMY_PR_AUC": _f(next(r for r in bench["results"] if r["id"] == "B0")["test"]["pr_auc"]),
        "TABLE_BENCHMARK_TEX": tex_table([r[1:] for r in brow], header[1:], bold_row=[r[0] for r in brow].index(sel + " *")),
        "TABLE_ABLATION_TEX": tex_table(arow, ["Set", "Families", "#feat", "CV PR-AUC", "CV ROC-AUC", "Test PR-AUC"]),
        "TABLE_BENCHMARK_MD": md_table(brow, header), "TABLE_ABLATION_MD": md_table(arow, ["Set", "Families", "#feat", "CV PR-AUC", "CV ROC-AUC", "Test PR-AUC"]),
        "TABLE_PAIRED_MD": md_table(prow, ["vs", "Baseline", "ΔPR-AUC", "95% CI", "p"]),
    }
    # extra values / tables for the full-length LaTeX report
    pos_src = ds.get("positive_rate_by_source", {})
    fnm = ea["summary"].get("fn_failure_modes", {})
    pretty = lambda k: k.replace("_", " ").lower().capitalize()  # noqa: E731
    xgbp = opt.get("xgboost", {}).get("best_params", {})
    numbers.update({
        "LOG_LOSS": _f(tm["log_loss"]), "BRIER": _f(tm["brier"]), "SPEC": _f(tm["specificity"]),
        "N_NEARDUP": dq.get("near_duplicate_groups", "n/a"), "IMBALANCE": dq.get("imbalance_ratio", "n/a"),
        "POS_BFCL": f"{pos_src.get('bfcl_v3', 0):.1%}", "POS_MCP": f"{pos_src.get('mcp_reference_servers', 0):.1%}",
        "POS_AUG": f"{pos_src.get('augmented', 0):.1%}", "N_REALQ": ds.get("by_episode_origin", {}).get("bfcl_real", "n/a"),
        "G_TRAIN": sp["train"].get("groups", "n/a"), "G_VAL": sp["val"].get("groups", "n/a"), "G_TEST": sp["test"].get("groups", "n/a"),
        "OPT_LGB_CV": _f(opt["lightgbm"]["best_value_cv_pr_auc"]) if "lightgbm" in opt else "n/a",
        "OPT_XGB_PRUNED": opt.get("xgboost", {}).get("n_pruned", "n/a"), "OPT_LGB_PRUNED": opt.get("lightgbm", {}).get("n_pruned", "n/a"),
        "OPT_XGB_BEST": ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in xgbp.items()),
        "POS_WEIGHT": f"{meta.get('pos_weight', 1):.2f}",
        "FN_MODES": ", ".join(f"{pretty(k)} {v:.0%}" for k, v in fnm.items()),
        "EA_POINTS": " ".join(p.rstrip(".") + "." for p in ea["summary"]["points"]),
        "TABLE_PAIRED_TEX": tex_table([[c["vs"], c["vs_name"], f"{c['pr_auc']['mean_delta']:+.3f}",
                                        f"[{c['pr_auc']['ci_low']:+.3f}, {c['pr_auc']['ci_high']:+.3f}]", f"{c['pr_auc']['p_value_one_sided']:.3f}"]
                                       for c in sorted(ev["paired_tests"], key=lambda c: -c["pr_auc"]["mean_delta"])],
                                      ["ID", "Baseline", r"$\Delta$PR-AUC", "95% CI", "p"]),
        "TABLE_SLICE_TEX": tex_table([[pretty(r["slice"]), r["n"], f"{r['error_rate']:.1%}", f"{r['positive_rate']:.0%}", r["fn"], r["fp"]]
                                      for r in ea["slices"].get("error_type", [])],
                                     ["Failure mode", "n", "Misclass.", "Pos. rate", "FN", "FP"]),
    })
    write_json({k: v for k, v in numbers.items() if not k.startswith("TABLE")}, rdir / "report_numbers.json")

    tdir = PROJECT_ROOT / "docs" / "report"
    for tpl, outname in (("IEEE_REPORT.template.md", "IEEE_REPORT.md"), ("ieee_report.template.tex", "ieee_report.tex")):
        p = tdir / tpl
        if p.exists():
            text = p.read_text(encoding="utf-8")
            is_tex = outname.endswith(".tex")

            def fill(m, is_tex=is_tex):
                key = m.group(1)
                if key not in numbers:
                    return m.group(0)
                val = str(numbers[key])
                # values land in running LaTeX text: escape %, #, &, _ (pre-built *_TEX tables are already escaped)
                if is_tex and not key.endswith("_TEX"):
                    val = val.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
                return val

            text = re.sub(r"\{\{([A-Z0-9_]+)\}\}", fill, text)
            (tdir / outname).write_text(text, encoding="utf-8")
    readme = PROJECT_ROOT / "README.md"
    if readme.exists():
        block = chr(10).join([
            "<!-- RESULTS:START -->",
            f"*Auto-generated from the latest pipeline run ({bench['generated_at'][:10]}). Labels: controlled agent benchmark.*", "",
            f"**Selected production model:** {meta['experiment_name']}: test **PR-AUC {numbers['PR_AUC']}** (95% CI {numbers['PR_CI']}), "
            f"ROC-AUC {numbers['ROC_AUC']}, F1 {numbers['F1']}, MCC {numbers['MCC']} (threshold {numbers['THRESHOLD']}). "
            f"No-skill PR-AUC = {numbers['DUMMY_PR_AUC']}; documentation-quality heuristic = {numbers['RULE_PR_AUC']}.", "",
            "**Benchmark (held-out test split, `*` = selected)**", "", numbers["TABLE_BENCHMARK_MD"], "",
            "**Ablation (XGBoost, 5-fold grouped CV)**", "", numbers["TABLE_ABLATION_MD"], "",
            f"Feature fusion vs. best single family ({numbers['ABL_BEST_SINGLE_NAME']}): {numbers['ABL_GAIN']} PR-AUC.", "",
            f"**Error analysis.** {numbers['EA_HEADLINE']}", "",
            "Paired bootstrap tests, calibration and all figures: `artifacts/reports/results_tables.md`, `artifacts/figures/`.",
            "<!-- RESULTS:END -->"])
        text = readme.read_text(encoding="utf-8")
        text = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda m: block, text, flags=re.S)
        readme.write_text(text, encoding="utf-8")
    log_event(log, "report artefacts written", tables=str(rdir / "results_tables.md"))
    return numbers


if __name__ == "__main__":
    run()
