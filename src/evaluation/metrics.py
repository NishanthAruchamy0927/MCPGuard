"""Classification metrics, threshold selection and bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-7, 1 - 1e-7)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, proba)) if both else float("nan"),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, proba)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn), "n": int(len(y_true)),
    }


def best_f1_threshold(y_true, proba) -> float:
    """Threshold maximising F1 (validation split only)."""
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return float(t[i]) if len(t) else 0.5


def curves(y_true, proba) -> dict:
    fpr, tpr, _ = roc_curve(y_true, proba)
    p, r, _ = precision_recall_curve(y_true, proba)
    idx_roc = np.linspace(0, len(fpr) - 1, min(len(fpr), 200)).astype(int)
    idx_pr = np.linspace(0, len(p) - 1, min(len(p), 200)).astype(int)
    return {"roc": {"fpr": fpr[idx_roc].round(4).tolist(), "tpr": tpr[idx_roc].round(4).tolist()},
            "pr": {"precision": p[idx_pr].round(4).tolist(), "recall": r[idx_pr].round(4).tolist()}}


def bootstrap_ci(y_true, proba, threshold: float, n_boot: int = 1000, seed: int = 42, metrics=("f1", "roc_auc", "pr_auc")) -> dict:
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    vals = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = compute_metrics(y_true[idx], proba[idx], threshold)
        for k in metrics:
            vals[k].append(m[k])
    return {k: {"low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5))} for k, v in vals.items()}


def paired_bootstrap_delta(y_true, p_a, p_b, metric: str = "pr_auc", n_boot: int = 1000, seed: int = 42) -> dict:
    """Bootstrap distribution of metric(A) - metric(B) on the same resamples."""
    fn = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score}[metric]
    rng = np.random.default_rng(seed)
    y_true, p_a, p_b = map(np.asarray, (y_true, p_a, p_b))
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        deltas.append(fn(y_true[idx], p_a[idx]) - fn(y_true[idx], p_b[idx]))
    d = np.array(deltas)
    return {"metric": metric, "mean_delta": float(d.mean()), "ci_low": float(np.percentile(d, 2.5)),
            "ci_high": float(np.percentile(d, 97.5)), "p_value_one_sided": float((d <= 0).mean())}
