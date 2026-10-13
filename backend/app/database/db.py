"""SQLAlchemy persistence (SQLite locally; any SQLAlchemy URL, e.g. PostgreSQL, via DATABASE_URL).

Only non-sensitive data is stored: tool names, a SHA-1 digest of the
description (never the text itself), scores and model versions.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.utils.config import PROJECT_ROOT, load_config

_log = logging.getLogger("mcpguard.db")


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Tool(Base):
    __tablename__ = "tools"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    tool_name: Mapped[str] = mapped_column(String(200))
    server_name: Mapped[str] = mapped_column(String(200))
    source: Mapped[str] = mapped_column(String(40))
    split: Mapped[str] = mapped_column(String(10))
    error_occurred: Mapped[int] = mapped_column(Integer)
    error_type: Mapped[str] = mapped_column(String(40))
    failure_rate: Mapped[float] = mapped_column(Float)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    tool_name: Mapped[str] = mapped_column(String(200))
    description_sha1: Mapped[str] = mapped_column(String(40))
    probability: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(12))
    model_version: Mapped[str] = mapped_column(String(40))
    channel: Mapped[str] = mapped_column(String(12), default="single")


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    stage: Mapped[str] = mapped_column(String(40))
    model: Mapped[str] = mapped_column(String(100))
    feature_set: Mapped[str] = mapped_column(String(100))
    params: Mapped[dict] = mapped_column(JSON)
    metrics: Mapped[dict] = mapped_column(JSON)
    timestamp: Mapped[str] = mapped_column(String(40))


class ModelMetric(Base):
    __tablename__ = "model_metrics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(10))
    model_name: Mapped[str] = mapped_column(String(200))
    split: Mapped[str] = mapped_column(String(10))
    metric: Mapped[str] = mapped_column(String(30))
    value: Mapped[float] = mapped_column(Float)


class ErrorAnalysisRow(Base):
    __tablename__ = "error_analysis"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tool_id: Mapped[str] = mapped_column(String(32))
    outcome: Mapped[str] = mapped_column(String(4))
    probability: Mapped[float] = mapped_column(Float)
    error_type: Mapped[str] = mapped_column(String(40))


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    version: Mapped[str] = mapped_column(String(20), primary_key=True)
    built_at: Mapped[str] = mapped_column(String(40))
    records: Mapped[int] = mapped_column(Integer)
    summary: Mapped[dict] = mapped_column(JSON)


def database_url() -> str:
    url = os.environ.get("DATABASE_URL") or load_config()["database"]["url"]
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////") and ":" not in url[10:12]:
        path = PROJECT_ROOT / url[len("sqlite:///"):]
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{path.as_posix()}"
    return url


_engine = None
_Session = None


def _open(url: str):
    engine = create_engine(url, future=True, pool_pre_ping=True)
    Base.metadata.create_all(engine)  # raises if the location is not writable
    return engine


def get_session() -> Session:
    """Open the configured database; if it is not writable (e.g. a root-owned
    Docker volume), fall back to a temp-dir SQLite file instead of failing."""
    global _engine, _Session
    if _Session is None:
        try:
            engine = _open(database_url())
        except Exception as exc:  # noqa: BLE001 - any driver/permission error
            fallback = Path(tempfile.gettempdir()) / "mcpguard.db"
            _log.warning("database %s unavailable (%s); using %s", database_url(), type(exc).__name__, fallback)
            engine = _open(f"sqlite:///{fallback.as_posix()}")
        _engine, _Session = engine, sessionmaker(engine, expire_on_commit=False)
    return _Session()


def digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def log_predictions(rows: list[dict], descriptions: list[str], channel: str) -> None:
    with get_session() as s:
        for r, d in zip(rows, descriptions):
            s.add(Prediction(tool_name=r["tool_name"][:200], description_sha1=digest(d), probability=r["probability"],
                             risk_level=r["risk_level"], model_version=r["model_version"], channel=channel))
        s.commit()


def prediction_stats() -> dict:
    with get_session() as s:
        total = s.scalar(select(func.count(Prediction.id))) or 0
        high = s.scalar(select(func.count(Prediction.id)).where(Prediction.risk_level.in_(["HIGH", "CRITICAL"]))) or 0
    return {"predictions_served": int(total), "high_risk_detected": int(high)}


def sync_artifacts(artifacts) -> dict:
    """Mirror experiment / metric artefacts into the database (idempotent)."""
    counts = {}
    with get_session() as s:
        for model in (Tool, Experiment, ModelMetric, ErrorAnalysisRow, DatasetVersion):
            s.query(model).delete()
        tools = artifacts.tools_frame()
        if tools is not None:
            for r in tools.itertuples():
                s.add(Tool(id=r.tool_id, tool_name=str(r.tool_name)[:200], server_name=str(r.server_name)[:200], source=r.source,
                           split=r.split, error_occurred=int(r.error_occurred), error_type=r.error_type, failure_rate=float(r.failure_rate)))
            counts["tools"] = len(tools)
        exps = artifacts.experiments()
        for e in exps:
            s.add(Experiment(id=e["experiment_id"], name=e["name"], stage=e["stage"], model=e["model"], feature_set=e["feature_set"],
                             params=e.get("params", {}), metrics=e.get("metrics", {}), timestamp=e["timestamp"]))
        counts["experiments"] = len(exps)
        bench = artifacts.get("benchmark.json")
        if bench:
            for r in bench["results"]:
                for split in ("train", "val", "test"):
                    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
                        s.add(ModelMetric(model_id=r["id"], model_name=r["name"], split=split, metric=k, value=float(r[split][k])))
        ea = artifacts.get("error_analysis.json")
        if ea:
            for kind, rows in ea["examples"].items():
                for r in rows:
                    s.add(ErrorAnalysisRow(tool_id=r["tool_id"], outcome=kind, probability=r["probability"], error_type=r["error_type"]))
        ds = artifacts.get_meta("dataset_summary.json")
        if ds:
            s.add(DatasetVersion(version=ds["dataset_version"], built_at=ds["built_at"], records=ds["records"], summary=json.loads(json.dumps(ds))))
        s.commit()
    return counts
