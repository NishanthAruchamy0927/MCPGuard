"""Model factory. Every model is a scikit-learn compatible estimator.

Class imbalance strategy (docs/METHODOLOGY.md): cost-sensitive learning –
``class_weight='balanced'`` / ``scale_pos_weight = n_neg / n_pos`` – plus
decision-threshold tuning on the validation split.  SMOTE is deliberately not
applied: interpolating between sentence-embedding / LSA vectors of different
tools produces points that correspond to no real tool description.
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

from src.utils.config import load_config

DISPLAY_NAMES = {
    "dummy": "Majority/Prior baseline",
    "doc_quality_rule": "Doc-quality heuristic (rule)",
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "hist_gb": "Hist. Gradient Boosting",
    "mlp": "MLP Neural Network",
    "xgboost_optuna": "XGBoost (Optuna-tuned)",
    "lightgbm_optuna": "LightGBM (Optuna-tuned)",
}
TREE_MODELS = {"random_forest", "xgboost", "lightgbm", "hist_gb", "xgboost_optuna", "lightgbm_optuna"}


def make_model(name: str, pos_weight: float = 1.0, sparse_input: bool = False, params: dict | None = None, seed: int | None = None):
    cfg = load_config()
    m = cfg["models"]
    seed = cfg["seed"] if seed is None else seed
    params = dict(params or {})
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        p = {**m["logistic_regression"], **params}
        scaler = MaxAbsScaler() if sparse_input else StandardScaler()
        return make_pipeline(scaler, LogisticRegression(C=p["C"], max_iter=p["max_iter"], class_weight="balanced", random_state=seed))
    if name == "random_forest":
        p = {**m["random_forest"], **params}
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if name in ("xgboost", "xgboost_optuna"):
        from xgboost import XGBClassifier
        p = {**m["xgboost"], **params}
        return XGBClassifier(**p, scale_pos_weight=pos_weight, eval_metric="aucpr", tree_method="hist", n_jobs=-1,
                             random_state=seed, verbosity=0)
    if name in ("lightgbm", "lightgbm_optuna"):
        from lightgbm import LGBMClassifier
        p = {**m["lightgbm"], **params}
        return LGBMClassifier(**p, subsample_freq=1, scale_pos_weight=pos_weight, random_state=seed, n_jobs=-1, verbose=-1)
    if name == "hist_gb":
        p = {**m["hist_gradient_boosting"], **params}
        return HistGradientBoostingClassifier(**p, class_weight="balanced", random_state=seed)
    if name == "mlp":
        p = {**m["mlp"], **params}
        scaler = MaxAbsScaler() if sparse_input else StandardScaler()
        return make_pipeline(scaler, MLPClassifier(hidden_layer_sizes=tuple(p["hidden_layer_sizes"]), alpha=p["alpha"],
                                                   learning_rate_init=p["learning_rate_init"], max_iter=p["max_iter"],
                                                   early_stopping=p["early_stopping"], random_state=seed))
    raise ValueError(f"unknown model {name}")
