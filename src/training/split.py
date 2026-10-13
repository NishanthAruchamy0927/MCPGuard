"""Leakage-safe, stratified *grouped* train/validation/test split.

Groups = ``lineage_id``: an augmented variant and its seed tool, and near-
duplicate definitions of the same function, always land in the same split.
StratifiedGroupKFold keeps the positive rate approximately equal across splits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils.config import load_config


def grouped_split(df: pd.DataFrame, seed: int | None = None) -> pd.Series:
    cfg = load_config()
    sc = cfg["split"]
    seed = cfg["seed"] if seed is None else seed
    y = df[sc["stratify_column"]].to_numpy()
    g = df[sc["group_column"]].to_numpy()
    split = pd.Series("train", index=df.index)
    n_test = max(2, round(1 / sc["test"]))
    test_idx = next(StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed).split(df, y, g))[1]
    split.iloc[test_idx] = "test"
    rest = np.where(split.to_numpy() != "test")[0]
    n_val = max(2, round((1 - sc["test"]) / sc["val"]))
    val_rel = next(StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed + 1).split(rest, y[rest], g[rest]))[1]
    split.iloc[rest[val_rel]] = "val"
    return split


def split_report(df: pd.DataFrame, split: pd.Series) -> dict:
    out = {}
    for s in ("train", "val", "test"):
        m = split == s
        out[s] = {"n": int(m.sum()), "share": round(float(m.mean()), 4), "positive_rate": round(float(df.loc[m, "error_occurred"].mean()), 4),
                  "groups": int(df.loc[m, "lineage_id"].nunique())}
    groups = {s: set(df.loc[split == s, "lineage_id"]) for s in ("train", "val", "test")}
    out["group_overlap"] = {"train_val": len(groups["train"] & groups["val"]), "train_test": len(groups["train"] & groups["test"]),
                            "val_test": len(groups["val"] & groups["test"])}
    return out
