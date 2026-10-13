"""Real-time inference: validate -> features -> model -> risk -> SHAP -> advice.

The trained pipeline is loaded once (``get_predictor``) and reused; nothing is
retrained at serving time.
"""
from __future__ import annotations

import threading
import time

import joblib
import numpy as np
import pandas as pd

from src.explainability.recommendations import recommend, risk_factors
from src.explainability.shap_explain import PipelineExplainer
from src.features.risk import structured_features
from src.preprocessing.schema import parse_schema
from src.utils.config import load_config
from src.utils.io import read_json
from src.utils.logging_utils import get_logger, log_event

log = get_logger("inference")


def risk_level(prob: float, levels: dict | None = None) -> str:
    """Map probability (0-1) to a configurable risk band."""
    levels = levels or load_config()["risk_levels"]
    score = prob * 100
    current = "LOW"
    for name, lower in sorted(levels.items(), key=lambda kv: kv[1]):
        if score >= lower:
            current = name
    return current


class MCPGuardPredictor:
    def __init__(self):
        cfg = load_config()
        self.cfg = cfg
        mdir = cfg.path("models")
        path = mdir / "best_model.joblib"
        if not path.exists():
            raise FileNotFoundError("Trained model not found. Run `python -m src.pipeline.run_all` first.")
        self.pipeline = joblib.load(path)
        self.metadata = read_json(mdir / "model_metadata.json")
        self.threshold = float(self.metadata["threshold"])
        self.explainer = PipelineExplainer(self.pipeline) if self.metadata.get("is_tree_model") else None
        self._lock = threading.Lock()
        self.loaded_at = time.time()
        log_event(log, "model loaded", model=self.metadata["model_name"], version=self.metadata["version"],
                  explainer="tree" if self.explainer else "none")

    def _frame(self, tools: list[dict]) -> pd.DataFrame:
        return pd.DataFrame([{"tool_name": t.get("tool_name", ""), "tool_description": t.get("description", ""),
                              "input_schema": t.get("input_schema", {})} for t in tools])

    def predict(self, tools: list[dict], explain: bool = True, top_k: int | None = None) -> list[dict]:
        top_k = top_k or self.cfg["explainability"]["top_k_features"]
        df = self._frame(tools)
        t0 = time.perf_counter()
        with self._lock:
            X = self.pipeline.named_steps["features"].transform(df)
            proba = self.pipeline.named_steps["model"].predict_proba(X)[:, 1]
            sv, base = (self.explainer.shap_values(X) if (explain and self.explainer) else (None, None))
        latency = (time.perf_counter() - t0) * 1000 / max(1, len(df))
        out = []
        for i, t in enumerate(tools):
            p = float(proba[i])
            pred = int(p >= self.threshold)
            feats = structured_features(t.get("tool_name", ""), t.get("description", ""), t.get("input_schema", {}))
            top = self.explainer.top_contributions(X[i], sv[i], top_k) if sv is not None else []
            fam = self.explainer.family_contributions(sv[i]) if sv is not None else {}
            _, issues = parse_schema(t.get("input_schema"))
            out.append({
                "tool_name": t.get("tool_name", ""),
                "prediction": pred,
                "prediction_label": "LIKELY AGENT ERROR" if pred else "LIKELY SUCCESS",
                "probability": round(p, 4),
                "risk_score": round(p * 100, 1),
                "risk_level": risk_level(p),
                "confidence": round(float(max(p, 1 - p)), 4),
                "decision_threshold": round(self.threshold, 4),
                "model": self.metadata["model_name"],
                "model_version": self.metadata["version"],
                "base_value": base,
                "top_features": top,
                "family_contributions": fam,
                "risk_factors": risk_factors(feats, top),
                "recommendations": recommend(feats, top),
                "feature_values": {k: round(v, 4) for k, v in feats.items()},
                "schema_issues": issues,
                "latency_ms": round(latency, 2),
            })
        return out


_predictor: MCPGuardPredictor | None = None
_plock = threading.Lock()


def get_predictor(reload: bool = False) -> MCPGuardPredictor:
    global _predictor
    with _plock:
        if _predictor is None or reload:
            _predictor = MCPGuardPredictor()
    return _predictor


def predict_one(tool: dict) -> dict:
    return get_predictor().predict([tool])[0]


if __name__ == "__main__":
    demo = {"tool_name": "search_customer", "description": "Searches for a customer using customer ID.",
            "input_schema": {"type": "object", "properties": {"customer_id": {"type": "string", "description": "Unique customer identifier"}},
                             "required": ["customer_id"]}}
    import json
    r = predict_one(demo)
    print(json.dumps({k: r[k] for k in ("probability", "risk_score", "risk_level", "prediction_label")}, indent=2))
    print(np.round([c["shap"] for c in r["top_features"]], 3))
