"""REST API routes."""
from __future__ import annotations

import io
import json
import math
import os
import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from backend.app.database.db import log_predictions, prediction_stats, sync_artifacts
from backend.app.schemas.models import BatchRequest, BatchResponse, PredictionResponse, PredictRequest, ToolInput
from backend.app.services.artifacts import artifacts
from src.inference.predictor import get_predictor, risk_level
from src.utils.config import load_config
from src.utils.logging_utils import get_logger, log_event

log = get_logger("api")


def _log_predictions(rows: list[dict], descriptions: list[str], channel: str) -> None:
    """Prediction logging is best-effort: a database problem must never fail inference."""
    try:
        log_predictions(rows, descriptions, channel)
    except Exception as exc:  # noqa: BLE001
        log_event(log, "prediction logging skipped", error=type(exc).__name__)


def _prediction_stats() -> dict:
    try:
        return prediction_stats()
    except Exception as exc:  # noqa: BLE001
        log_event(log, "prediction stats unavailable", error=type(exc).__name__)
        return {"predictions_served": None, "high_risk_detected": None}
router = APIRouter()
STARTED = time.time()


def _clean(o):
    """JSON-safe: NaN/inf -> None, numpy -> python."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    return o


def _require(obj, what: str):
    if obj is None:
        raise HTTPException(status_code=503, detail=f"{what} not available yet — run the training pipeline (python -m src.pipeline.run_all).")
    return obj


def _predictor():
    try:
        return get_predictor()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model artefacts not found — run the training pipeline first.") from None


# ------------------------------------------------------------------ status
@router.get("/health", tags=["status"])
def health():
    meta = artifacts.model_meta()
    try:
        p = get_predictor()
        loaded = True
    except Exception:  # noqa: BLE001
        p, loaded = None, False
    return {"status": "ok" if loaded else "degraded", "model_loaded": loaded,
            "model": meta["model_name"] if meta else None, "model_version": meta["version"] if meta else None,
            "uptime_s": round(time.time() - STARTED, 1), "loaded_at": p.loaded_at if p else None}


@router.get("/api", tags=["status"])
def api_index():
    cfg = load_config()
    return {"name": cfg["project"]["name"], "title": cfg["project"]["title"], "tagline": cfg["project"]["tagline"],
            "docs": "/docs", "redoc": "/redoc",
            "endpoints": ["/health", "/predict", "/batch-predict", "/batch-predict/file", "/model-metrics", "/feature-importance",
                          "/dataset-summary", "/dataset/tools", "/tools/{tool_id}", "/error-analysis", "/experiments", "/model-info",
                          "/projections", "/stats", "/reload-model"]}


@router.get("/stats", tags=["status"])
def stats():
    meta = artifacts.model_meta()
    ds = artifacts.get_meta("dataset_summary.json")
    ev = artifacts.get("evaluation.json")
    df = artifacts.tools_frame()
    high = None
    if df is not None and "proba" in df:
        lv = df["proba"].dropna().map(risk_level)
        high = int(lv.isin(["HIGH", "CRITICAL"]).sum())
    return _clean({"tools_analyzed": int(len(df)) if df is not None else 0, "dataset_size": ds["records"] if ds else 0,
                   "model": meta["model_name"] if meta else None, "experiment": meta["experiment_name"] if meta else None,
                   "test_f1": ev["test_metrics"]["f1"] if ev else None, "test_roc_auc": ev["test_metrics"]["roc_auc"] if ev else None,
                   "test_pr_auc": ev["test_metrics"]["pr_auc"] if ev else None, "high_risk_tools": high,
                   "servers": ds["servers"] if ds else None, "positive_rate": ds["positive_rate"] if ds else None,
                   "risk_levels": load_config()["risk_levels"], **_prediction_stats()})


# --------------------------------------------------------------- inference
@router.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(req: PredictRequest):
    p = _predictor()
    tool = req.model_dump()
    res = p.predict([tool], explain=req.explain)[0]
    _log_predictions([res], [req.description], "single")
    log_event(log, "prediction", tool=req.tool_name[:60], risk=res["risk_level"], latency_ms=res["latency_ms"])
    return _clean(res)


def _batch(tools: list[dict], explain: bool, channel: str) -> dict:
    cfg = load_config()["api"]
    if len(tools) > cfg["max_batch_size"]:
        raise HTTPException(status_code=413, detail=f"Batch too large (max {cfg['max_batch_size']} tools).")
    valid, rows = [], []
    for i, raw in enumerate(tools):
        try:
            t = ToolInput.model_validate(raw)
            valid.append((i, t))
        except ValidationError as exc:
            err = exc.errors()[0]
            rows.append({"row": i + 1, "tool_name": str((raw or {}).get("tool_name", ""))[:200] if isinstance(raw, dict) else "",
                         "error": f"{'.'.join(str(x) for x in err['loc'])}: {err['msg'].replace('Value error, ', '')}"})
    if valid:
        res = _predictor().predict([t.model_dump() for _, t in valid], explain=True)
        for (i, t), r in zip(valid, res):
            rows.append({"row": i + 1, "tool_name": t.tool_name, "risk_score": r["risk_score"], "risk_level": r["risk_level"],
                         "predicted_error": r["prediction"], "confidence": r["confidence"],
                         "top_risk_factor": r["risk_factors"][0] if r["risk_factors"] else None})
        _log_predictions(res, [t.description for _, t in valid], channel)
    rows.sort(key=lambda r: r["row"])
    summary = {k: sum(1 for r in rows if r.get("risk_level") == k) for k in ("LOW", "MEDIUM", "HIGH", "CRITICAL")}
    log_event(log, "batch inference", n=len(tools), ok=len(valid), channel=channel)
    return {"n": len(tools), "n_ok": len(valid), "n_failed": len(tools) - len(valid), "results": rows, "summary": summary}


@router.post("/batch-predict", response_model=BatchResponse, tags=["inference"])
def batch_predict(req: BatchRequest):
    return _batch([t.model_dump() for t in req.tools], req.explain, "batch")


@router.post("/batch-predict/file", tags=["inference"])
async def batch_predict_file(file: UploadFile = File(...), format: str = Query("json", pattern="^(json|csv)$")):
    """Upload a CSV (columns: tool_name, description|tool_description, input_schema) or JSON (list of tools)."""
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB).")
    name = (file.filename or "").lower()
    try:
        if name.endswith(".json") or (file.content_type or "").endswith("json"):
            payload = json.loads(data.decode("utf-8"))
            tools = payload.get("tools", payload) if isinstance(payload, dict) else payload
            if not isinstance(tools, list):
                raise ValueError("JSON must be a list of tools or {\"tools\": [...]}")
        elif name.endswith(".csv") or "csv" in (file.content_type or ""):
            df = pd.read_csv(io.BytesIO(data), dtype=str).fillna("")
            if "description" not in df.columns and "tool_description" in df.columns:
                df = df.rename(columns={"tool_description": "description"})
            missing = {"tool_name", "description"} - set(df.columns)
            if missing:
                raise ValueError(f"CSV is missing columns: {sorted(missing)}")
            tools = df.to_dict(orient="records")
        else:
            raise ValueError("Unsupported file type — upload .csv or .json")
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}") from None
    result = _batch(tools, False, "file")
    if format == "csv":
        out = pd.DataFrame(result["results"])
        cols = ["tool_name", "risk_score", "risk_level", "predicted_error", "confidence", "top_risk_factor", "error"]
        out = out.reindex(columns=cols)
        buf = io.StringIO()
        out.to_csv(buf, index=False)
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=mcpguard_predictions.csv"})
    return result


@router.post("/reload-model", tags=["status"])
def reload_model(x_admin_token: str | None = Header(None)):
    token = os.environ.get("MCPGUARD_ADMIN_TOKEN")
    if token and x_admin_token != token:
        raise HTTPException(status_code=401, detail="Invalid admin token.")
    try:
        p = get_predictor(reload=True)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model artefacts not found.") from None
    counts = sync_artifacts(artifacts)
    return {"status": "reloaded", "model": p.metadata["model_name"], "version": p.metadata["version"], "synced": counts}


# --------------------------------------------------------------- research
@router.get("/model-info", tags=["research"])
def model_info():
    meta = _require(artifacts.model_meta(), "Model metadata")
    emb = None
    p = artifacts.models_dir / "embedding_config.json"
    if p.exists():
        emb = json.loads(p.read_text(encoding="utf-8"))
    return _clean({**meta, "embedding": emb, "optuna": {m: artifacts.best_params(m) for m in ("xgboost", "lightgbm")},
                   "risk_levels": load_config()["risk_levels"]})


@router.get("/model-metrics", tags=["research"])
def model_metrics():
    bench = _require(artifacts.get("benchmark.json"), "Benchmark")
    return _clean({"selection": bench["selection"], "results": bench["results"], "evaluation": artifacts.get("evaluation.json"),
                   "curves": artifacts.get("curves.json"), "ablation": artifacts.get("ablation.json"),
                   "model": artifacts.model_meta()})


@router.get("/feature-importance", tags=["research"])
def feature_importance(top: int = Query(25, ge=1, le=200)):
    sg = _require(artifacts.get("shap_global.json"), "SHAP artefacts")
    return _clean({**sg, "global": sg["global"][:top]})


@router.get("/dataset-summary", tags=["research"])
def dataset_summary():
    ds = _require(artifacts.get_meta("dataset_summary.json"), "Dataset summary")
    df = artifacts.tools_frame()
    out = {"summary": ds, "quality": artifacts.get_meta("data_quality_report.json"), "split": artifacts.get_meta("split_report.json"),
           "manifest": artifacts.get_meta("download_manifest.json")}
    if df is not None:
        out["servers_top"] = df["server_name"].value_counts().head(15).to_dict()
        out["severity_by_source"] = pd.crosstab(df["source"], df["error_severity"]).to_dict(orient="index")
        out["error_type_by_label"] = pd.crosstab(df["error_type"], df["error_occurred"]).rename(columns=str).to_dict(orient="index")
        num = ["description_length", "parameter_count", "required_parameter_count", "nested_schema_depth", "enum_count",
               "failure_rate", "agent_success_rate"]
        desc = df[num].describe().T.round(3)
        out["feature_statistics"] = desc.reset_index().rename(columns={"index": "feature"}).to_dict(orient="records")
        out["missing_values"] = {c: int(v) for c, v in df.isna().sum().items() if v}
        out["histograms"] = {
            c: dict(zip(["counts", "edges"], [h.tolist() for h in np.histogram(df[c].clip(upper=df[c].quantile(0.98)), bins=24)]))
            for c in ("description_length", "parameter_count", "failure_rate")}
        sfile = artifacts.metrics_dir.parent.parent / "data" / "processed" / "structured_features.csv"
        if sfile.exists():
            st = pd.read_csv(sfile)
            risk_cols = ["ambiguity_score", "missing_constraint_score", "parameter_complexity_score", "schema_complexity_score",
                         "description_quality_score", "instruction_clarity_score"]
            out["risk_feature_means"] = st.groupby("error_occurred")[risk_cols].mean().round(4).to_dict(orient="index")
    return _clean(out)


@router.get("/dataset/tools", tags=["research"])
def dataset_tools(q: str | None = None, risk_level_filter: str | None = Query(None, alias="risk_level"),
                  error_type: str | None = None, server: str | None = None, source: str | None = None, split: str | None = None,
                  actual: int | None = Query(None, ge=0, le=1), predicted: int | None = Query(None, ge=0, le=1),
                  sort: str = Query("proba", pattern="^(proba|failure_rate|tool_name|description_length|parameter_count)$"),
                  desc: bool = True, page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=200)):
    df = _require(artifacts.tools_frame(), "Dataset")
    d = df
    if q:
        ql = q.lower()
        d = d[d["tool_name"].str.lower().str.contains(ql, regex=False) | d["tool_description"].str.lower().str.contains(ql, regex=False)]
    if error_type:
        d = d[d["error_type"] == error_type]
    if server:
        d = d[d["server_name"] == server]
    if source:
        d = d[d["source"] == source]
    if split:
        d = d[d["split"] == split]
    if actual is not None:
        d = d[d["error_occurred"] == actual]
    if predicted is not None and "y_pred" in d:
        d = d[d["y_pred"] == predicted]
    if "proba" in d:
        d = d.assign(risk_level=d["proba"].map(lambda p: risk_level(p) if pd.notna(p) else None))
        if risk_level_filter:
            d = d[d["risk_level"] == risk_level_filter]
    d = d.sort_values(sort, ascending=not desc, na_position="last")
    total = len(d)
    page_df = d.iloc[(page - 1) * page_size: page * page_size]
    cols = ["tool_id", "tool_name", "server_name", "source", "split", "tool_description", "parameter_count", "description_length",
            "error_occurred", "error_type", "error_severity", "failure_rate", "proba", "y_pred", "risk_level"]
    rows = page_df[[c for c in cols if c in page_df]].to_dict(orient="records")
    return _clean({"total": total, "page": page, "page_size": page_size, "rows": rows,
                   "facets": {"error_type": sorted(df["error_type"].unique().tolist()), "source": sorted(df["source"].unique().tolist()),
                              "server": df["server_name"].value_counts().head(60).index.tolist()}})


@router.get("/tools/{tool_id}", tags=["research"])
def tool_detail(tool_id: str):
    df = _require(artifacts.tools_frame(), "Dataset")
    row = df[df["tool_id"] == tool_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Tool not found.")
    r = row.iloc[0].to_dict()
    try:
        r["input_schema"] = json.loads(r["input_schema"])
    except (TypeError, json.JSONDecodeError):
        pass
    if isinstance(r.get("error_type_counts"), str):
        r["error_type_counts"] = json.loads(r["error_type_counts"])
    local = artifacts.get("shap_local.json") or {}
    r["explanation"] = local.get(tool_id)
    if pd.notna(r.get("proba")):
        r["risk_level"] = risk_level(float(r["proba"]))
    return _clean(r)


@router.get("/error-analysis", tags=["research"])
def error_analysis():
    return _clean(_require(artifacts.get("error_analysis.json"), "Error analysis"))


@router.get("/experiments", tags=["research"])
def experiments():
    exps = artifacts.experiments()
    return _clean({"runs": exps, "optuna": {m: {"best": artifacts.best_params(m), "trials": artifacts.optuna_trials(m)} for m in ("xgboost", "lightgbm")},
                   "mlflow": {"enabled": load_config()["tracking"]["mlflow_enabled"], "backend": "sqlite:///artifacts/mlflow.db",
                              "ui_command": "mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db"}})


@router.get("/projections", tags=["research"])
def projections():
    return _clean(_require(artifacts.get("projections.json"), "Projections"))


@router.get("/figures", tags=["research"])
def figures(request: Request):
    idx = artifacts.figures_dir / "index.json"
    names = json.loads(idx.read_text(encoding="utf-8"))["figures"] if idx.exists() else []
    return {"figures": [{"name": n, "url": str(request.url_for("figures_static", path=n))} for n in names]}
