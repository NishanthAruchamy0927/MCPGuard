"""Experiment tracking: MLflow (SQLite backend) + always-on JSONL logger.

The JSONL log (``artifacts/experiments/experiments.jsonl``) is the source of
truth for the dashboard's Experiments page, so tracking keeps working even
when MLflow is unavailable (e.g. on a slim cloud container).
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from src.utils.config import load_config, resolve
from src.utils.logging_utils import get_logger, log_event

log = get_logger("training.tracking")


class ExperimentTracker:
    def __init__(self, run_group: str | None = None):
        cfg = load_config()
        self.cfg = cfg
        self.log_path = resolve(cfg["tracking"]["custom_log_file"])
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_group = run_group or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.mlflow = None
        if cfg["tracking"]["mlflow_enabled"]:
            try:
                import logging as _logging
                import os as _os
                _os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
                import mlflow
                _logging.getLogger("mlflow").setLevel(_logging.ERROR)
                db = resolve("artifacts/mlflow.db")
                mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
                mlflow.set_experiment(cfg["tracking"]["experiment_name"])
                self.mlflow = mlflow
            except Exception as exc:  # pragma: no cover - environment dependent
                log_event(log, "MLflow unavailable; custom logger only", error=str(exc)[:200])

    def reset(self) -> None:
        if self.log_path.exists():
            self.log_path.unlink()

    def log_run(self, name: str, model: str, feature_set: str, params: dict, metrics: dict, extra: dict | None = None,
                stage: str = "benchmark") -> str:
        run_id = uuid.uuid4().hex[:10]
        record = {
            "experiment_id": run_id, "run_group": self.run_group, "name": name, "stage": stage, "model": model,
            "feature_set": feature_set, "dataset_version": self.cfg["data"]["dataset_version"],
            "params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in (params or {}).items()},
            "metrics": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()},
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(extra or {}),
        }
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        if self.mlflow is not None:
            try:
                with self.mlflow.start_run(run_name=name):
                    self.mlflow.set_tags({"stage": stage, "model": model, "feature_set": feature_set,
                                          "dataset_version": record["dataset_version"], "run_group": self.run_group})
                    self.mlflow.log_params({k: str(v)[:250] for k, v in record["params"].items()})
                    self.mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v == v})
            except Exception as exc:  # pragma: no cover
                log_event(log, "MLflow logging failed", error=str(exc)[:200])
        return run_id

    @contextmanager
    def timer(self):
        t = {"start": time.perf_counter()}
        yield t
        t["seconds"] = time.perf_counter() - t["start"]
