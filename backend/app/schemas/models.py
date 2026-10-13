"""Pydantic request / response models with strict input validation."""
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.utils.config import load_config

_API = load_config()["api"]


class ToolInput(BaseModel):
    """One MCP tool definition to analyse."""

    model_config = ConfigDict(extra="ignore", json_schema_extra={"example": {
        "tool_name": "search_customer",
        "description": "Searches for a customer using customer ID.",
        "input_schema": {"type": "object", "properties": {"customer_id": {"type": "string", "description": "Unique customer identifier"}},
                         "required": ["customer_id"]},
    }})

    tool_name: str = Field(..., min_length=1, max_length=200, description="MCP tool name")
    description: str = Field(..., min_length=1, max_length=_API["max_description_chars"], description="Tool description shown to the agent")
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}},
                                         description="MCP inputSchema (JSON-schema object). A JSON string is also accepted.")
    server_name: str | None = Field(None, max_length=200)
    tool_category: str | None = Field(None, max_length=100)
    execution_context: str | None = Field(None, max_length=500)

    @field_validator("tool_name", "description")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v.strip()

    @field_validator("input_schema", mode="before")
    @classmethod
    def _schema(cls, v: Any) -> dict:
        if v is None or v == "":
            return {"type": "object", "properties": {}}
        if isinstance(v, str):
            if len(v) > _API["max_schema_chars"]:
                raise ValueError("input_schema is too large")
            try:
                v = json.loads(v)
            except json.JSONDecodeError as exc:
                raise ValueError(f"input_schema is not valid JSON (line {exc.lineno}, column {exc.colno})") from None
        if not isinstance(v, dict):
            raise ValueError("input_schema must be a JSON object")
        if "type" in v and v["type"] not in ("object",):
            raise ValueError("input_schema.type must be 'object' for MCP tools")
        if "properties" in v and not isinstance(v["properties"], dict):
            raise ValueError("input_schema.properties must be an object mapping parameter names to schemas")
        if "required" in v and (not isinstance(v["required"], list) or not all(isinstance(x, str) for x in v["required"])):
            raise ValueError("input_schema.required must be a list of parameter names")
        if len(json.dumps(v)) > _API["max_schema_chars"]:
            raise ValueError("input_schema is too large")
        return v


class PredictRequest(ToolInput):
    explain: bool = Field(True, description="Return SHAP explanation and recommendations")


class BatchRequest(BaseModel):
    tools: list[ToolInput] = Field(..., min_length=1, max_length=_API["max_batch_size"])
    explain: bool = False


class FeatureContribution(BaseModel):
    feature: str
    display_name: str
    family: str
    family_label: str
    value: float
    shap: float
    direction: str


class Recommendation(BaseModel):
    title: str
    detail: str
    priority: Literal["HIGH", "MEDIUM", "LOW"]
    feature: str | None = None
    model_driver: bool = False


class PredictionResponse(BaseModel):
    tool_name: str
    prediction: int
    prediction_label: str
    probability: float
    risk_score: float
    risk_level: str
    confidence: float
    decision_threshold: float
    model: str
    model_version: str
    base_value: float | None = None
    top_features: list[FeatureContribution] = []
    family_contributions: dict[str, float] = {}
    risk_factors: list[str] = []
    recommendations: list[Recommendation] = []
    feature_values: dict[str, float] = {}
    schema_issues: list[str] = []
    latency_ms: float


class BatchRow(BaseModel):
    row: int
    tool_name: str
    risk_score: float | None = None
    risk_level: str | None = None
    predicted_error: int | None = None
    confidence: float | None = None
    top_risk_factor: str | None = None
    error: str | None = None


class BatchResponse(BaseModel):
    n: int
    n_ok: int
    n_failed: int
    results: list[BatchRow]
    summary: dict[str, int]
