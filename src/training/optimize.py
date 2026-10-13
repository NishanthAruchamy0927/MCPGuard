"""Bayesian hyper-parameter optimisation with Optuna (TPE + median pruning).

Objective: mean average precision (PR-AUC) over a stratified *grouped* k-fold
CV **inside the training split** – the validation split stays untouched for
model selection / threshold tuning and the test split for final reporting.
Feature builders are re-fitted inside every fold (no unsupervised leakage).

Run: ``python -m src.training.optimize [--model xgboost|lightgbm|all] [--trials N]``
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from src.features.build import ABLATION_SETS, MCPFeatureBuilder
from src.models.registry import make_model
from src.training.data import load_with_split
from src.training.tracking import ExperimentTracker
from src.utils.config import load_config
from src.utils.io import set_seed, write_json
from src.utils.logging_utils import get_logger, log_event

log = get_logger("training.optimize")


def fold_matrices(df: pd.DataFrame, families: tuple, n_folds: int, seed: int) -> list[tuple]:
    y = df["error_occurred"].to_numpy()
    g = df["lineage_id"].to_numpy()
    out = []
    for tr, va in StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(df, y, g):
        fb = MCPFeatureBuilder(families=families, random_state=seed).fit(df.iloc[tr])
        out.append((fb.transform(df.iloc[tr]), y[tr], fb.transform(df.iloc[va]), y[va]))
    return out


def suggest(trial: optuna.Trial, model: str) -> dict:
    if model == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 900, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 12.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        }
    if model == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 900, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 8, 96),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        }
    raise ValueError(model)


def optimize(model: str, n_trials: int | None = None, folds: list | None = None) -> dict:
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    ocfg = cfg["optuna"]
    n_trials = n_trials or ocfg[f"n_trials_{model}"]
    df = load_with_split()
    train = df[df["split"] == "train"].reset_index(drop=True)
    pos_weight = float((train["error_occurred"] == 0).sum() / max(1, (train["error_occurred"] == 1).sum()))
    if folds is None:
        folds = fold_matrices(train, ABLATION_SETS["F_all_features"], ocfg["cv_folds"], seed)
    tracker = ExperimentTracker()
    rows = []

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, model)
        t0 = time.perf_counter()
        scores = []
        for i, (Xtr, ytr, Xva, yva) in enumerate(folds):
            clf = make_model(model, pos_weight=pos_weight, params=params)
            clf.fit(Xtr, ytr)
            scores.append(average_precision_score(yva, clf.predict_proba(Xva)[:, 1]))
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        score = float(np.mean(scores))
        rows.append({"trial": trial.number, **params, "cv_pr_auc": score, "cv_pr_auc_std": float(np.std(scores)),
                     "train_seconds": round(time.perf_counter() - t0, 3), "state": "COMPLETE"})
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name=f"mcpguard_{model}",
                                sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=ocfg["timeout_seconds"])
    elapsed = time.perf_counter() - t0
    trials = pd.DataFrame([{"trial": t.number, "state": t.state.name, "value": t.value, **t.params,
                            "duration_s": t.duration.total_seconds() if t.duration else None} for t in study.trials])
    trials.to_csv(cfg.path("experiments") / f"optuna_{model}_trials.csv", index=False)
    result = {"model": model, "best_value_cv_pr_auc": float(study.best_value), "best_params": study.best_params,
              "best_trial": int(study.best_trial.number), "n_trials": len(study.trials),
              "n_pruned": int(sum(t.state.name == "PRUNED" for t in study.trials)), "cv_folds": ocfg["cv_folds"],
              "search_seconds": round(elapsed, 1), "sampler": "TPE", "pruner": "MedianPruner",
              "objective": "mean grouped-CV average precision (train split)"}
    write_json(result, cfg.path("models") / f"best_params_{model}.json")
    tracker.log_run(f"optuna_{model}", model, "F_all_features", study.best_params,
                    {"cv_pr_auc": study.best_value, "n_trials": len(study.trials)}, stage="optimization",
                    extra={"training_time_s": round(elapsed, 2)})
    log_event(log, "optimisation complete", model=model, best_cv_pr_auc=round(study.best_value, 4), trials=len(study.trials))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=["all", "xgboost", "lightgbm"])
    ap.add_argument("--trials", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config()
    df = load_with_split()
    train = df[df["split"] == "train"].reset_index(drop=True)
    folds = fold_matrices(train, ABLATION_SETS["F_all_features"], cfg["optuna"]["cv_folds"], cfg["seed"])
    for m in (["xgboost", "lightgbm"] if a.model == "all" else [a.model]):
        optimize(m, a.trials, folds)


if __name__ == "__main__":
    main()
