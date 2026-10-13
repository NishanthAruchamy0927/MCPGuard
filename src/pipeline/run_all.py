"""End-to-end pipeline.

Run: ``python -m src.pipeline.run_all [--skip-download] [--fast]``

    download -> build dataset (controlled benchmark) -> preprocess -> features
    -> Optuna -> train/benchmark/ablation -> SHAP -> evaluate/error analysis
    -> figures -> report tables

``--fast`` uses 6 Optuna trials per model (smoke test); the default uses the
trial budgets from configs/config.yaml.
"""
from __future__ import annotations

import argparse
import time

from src.utils.logging_utils import get_logger, log_event

log = get_logger("pipeline")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()

    from src.data import build_dataset, download
    from src.evaluation import evaluate
    from src.explainability import generate
    from src.features import build as features
    from src.preprocessing import clean
    from src.reporting import report
    from src.training import optimize, train
    from src.training.data import load_with_split
    from src.visualization import figures

    stages = []

    def stage(name, fn):
        s = time.perf_counter()
        log_event(log, f"stage start: {name}")
        fn()
        stages.append((name, round(time.perf_counter() - s, 1)))
        log_event(log, f"stage done: {name}", seconds=stages[-1][1])

    if not a.skip_download:
        stage("download", download.download)
    stage("build_dataset", build_dataset.build)
    stage("preprocess", clean.run)
    stage("split", lambda: load_with_split(refresh=True))
    stage("features", features.main)

    def _opt():
        from src.training.tracking import ExperimentTracker
        from src.utils.config import load_config
        ExperimentTracker().reset()  # fresh pipeline run -> fresh run log
        cfg = load_config()
        df = load_with_split()
        tr = df[df["split"] == "train"].reset_index(drop=True)
        folds = optimize.fold_matrices(tr, optimize.ABLATION_SETS["F_all_features"], cfg["optuna"]["cv_folds"], cfg["seed"])
        for m in ("xgboost", "lightgbm"):
            optimize.optimize(m, 6 if a.fast else None, folds)

    stage("optimize", _opt)
    stage("train", train.run)
    stage("explain", generate.run)
    stage("evaluate", evaluate.run)
    stage("figures", figures.run)
    stage("report", report.run)
    log_event(log, "pipeline complete", seconds=round(time.perf_counter() - t0, 1), stages=dict(stages))


if __name__ == "__main__":
    main()
