"""Shared data access for training / evaluation stages."""
from __future__ import annotations

import pandas as pd

from src.training.split import grouped_split, split_report
from src.utils.config import load_config, resolve
from src.utils.io import read_jsonl, write_json


def load_clean() -> pd.DataFrame:
    cfg = load_config()
    path = resolve(cfg["data"]["clean_file"])
    if not path.exists():
        from src.preprocessing.clean import run
        return run()
    return pd.DataFrame(read_jsonl(path))


def load_with_split(refresh: bool = False) -> pd.DataFrame:
    """Clean dataset with a persisted ``split`` column (train / val / test)."""
    cfg = load_config()
    df = load_clean()
    path = cfg.path("processed") / "splits.csv"
    if path.exists() and not refresh:
        s = pd.read_csv(path)
        mapping = dict(zip(s["tool_id"], s["split"]))
        if set(df["tool_id"]) <= set(mapping):
            df["split"] = df["tool_id"].map(mapping)
            return df
    df["split"] = grouped_split(df).values
    df[["tool_id", "lineage_id", "split"]].to_csv(path, index=False)
    write_json(split_report(df, df["split"]), cfg.path("metadata") / "split_report.json")
    return df
