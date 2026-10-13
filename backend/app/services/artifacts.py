"""Cached, read-only access to pipeline artefacts (re-read automatically when files change)."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.io import read_jsonl


class Artifacts:
    def __init__(self):
        cfg = load_config()
        self.metrics_dir: Path = cfg.path("metrics")
        self.meta_dir: Path = cfg.path("metadata")
        self.models_dir: Path = cfg.path("models")
        self.exp_dir: Path = cfg.path("experiments")
        self.figures_dir: Path = cfg.path("figures")
        self.clean_file = Path(cfg.path("processed")) / Path(cfg["data"]["clean_file"]).name
        self.split_file = Path(cfg.path("processed")) / "splits.csv"
        self.exp_log = self.exp_dir / "experiments.jsonl"
        self._cache: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    def _load(self, path: Path, loader):
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        key = str(path)
        with self._lock:
            hit = self._cache.get(key)
            if hit and hit[0] == mtime:
                return hit[1]
        value = loader(path)
        with self._lock:
            self._cache[key] = (mtime, value)
        return value

    @staticmethod
    def _json(p: Path):
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def get(self, name: str):
        return self._load(self.metrics_dir / name, self._json)

    def get_meta(self, name: str):
        return self._load(self.meta_dir / name, self._json)

    def model_meta(self):
        return self._load(self.models_dir / "model_metadata.json", self._json)

    def best_params(self, model: str):
        return self._load(self.models_dir / f"best_params_{model}.json", self._json)

    def experiments(self) -> list[dict]:
        rows = self._load(self.exp_log, read_jsonl)
        return list(rows or [])

    def optuna_trials(self, model: str) -> list[dict]:
        df = self._load(self.exp_dir / f"optuna_{model}_trials.csv", pd.read_csv)
        return [] if df is None else df.where(pd.notna(df), None).to_dict(orient="records")

    def tools_frame(self) -> pd.DataFrame | None:
        def build(_):
            df = pd.DataFrame(read_jsonl(self.clean_file))
            if self.split_file.exists():
                s = pd.read_csv(self.split_file)
                df["split"] = df["tool_id"].map(dict(zip(s.tool_id, s.split)))
            preds = self.metrics_dir / "predictions.csv"
            if preds.exists():
                p = pd.read_csv(preds)
                df["proba"] = df["tool_id"].map(dict(zip(p.tool_id, p.proba)))
                df["y_pred"] = df["tool_id"].map(dict(zip(p.tool_id, p.y_pred)))
            return df
        if not self.clean_file.exists():
            return None
        # invalidate when predictions change too
        key_path = self.metrics_dir / "predictions.csv"
        stamp = (self.clean_file.stat().st_mtime, key_path.stat().st_mtime if key_path.exists() else 0)
        with self._lock:
            hit = self._cache.get("tools_frame")
            if hit and hit[0] == stamp:
                return hit[1]
        df = build(None)
        with self._lock:
            self._cache["tools_frame"] = (stamp, df)
        return df


artifacts = Artifacts()
