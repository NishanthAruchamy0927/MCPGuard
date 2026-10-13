import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import best_f1_threshold, bootstrap_ci, compute_metrics
from src.models.registry import make_model
from src.training.split import grouped_split
from src.utils.config import PROJECT_ROOT
from tests.conftest import needs_model


@pytest.mark.parametrize("name", ["logreg", "random_forest", "xgboost", "lightgbm", "hist_gb", "mlp", "dummy"])
def test_every_registered_model_trains_and_predicts(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6))
    y = (X[:, 0] + 0.3 * rng.normal(size=120) > 0).astype(int)
    m = make_model(name, pos_weight=1.0, params={"n_estimators": 30} if name in ("random_forest", "xgboost", "lightgbm") else None)
    m.fit(X, y)
    p = m.predict_proba(X)[:, 1]
    assert p.shape == (120,) and ((p >= 0) & (p <= 1)).all()


def test_metrics_and_threshold():
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.1, 0.4, 0.35, 0.8, 0.9, 0.2])
    m = compute_metrics(y, p, 0.3)
    assert m["tp"] + m["fn"] == 3 and m["n"] == 6
    assert 0 <= best_f1_threshold(y, p) <= 1
    ci = bootstrap_ci(y, p, 0.5, n_boot=50)
    assert ci["roc_auc"]["low"] <= ci["roc_auc"]["high"]


def test_grouped_split_has_no_group_leakage():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"lineage_id": [f"g{i // 3}" for i in range(600)], "error_occurred": rng.integers(0, 2, 600)})
    s = grouped_split(df)
    groups = {k: set(df.lineage_id[s == k]) for k in ("train", "val", "test")}
    assert not groups["train"] & groups["test"] and not groups["train"] & groups["val"] and not groups["val"] & groups["test"]
    assert 0.6 < (s == "train").mean() < 0.8


@needs_model
def test_trained_artifacts_are_consistent():
    meta = json.loads((PROJECT_ROOT / "models" / "model_metadata.json").read_text())
    schema = json.loads((PROJECT_ROOT / "models" / "feature_schema.json").read_text())
    assert meta["n_features"] == len(schema["feature_names"])
    assert 0 < meta["threshold"] < 1
    bench = json.loads((PROJECT_ROOT / "artifacts" / "metrics" / "benchmark.json").read_text())
    best = next(r for r in bench["results"] if r["id"] == meta["model_id"])
    # metrics shown in the app are exactly the ones produced by training
    assert best["test"]["pr_auc"] == meta["metrics"]["test"]["pr_auc"]
    # selected model must beat the no-skill baseline on validation PR-AUC
    dummy = next(r for r in bench["results"] if r["model_key"] == "dummy")
    assert best["val"]["pr_auc"] > dummy["val"]["pr_auc"]
