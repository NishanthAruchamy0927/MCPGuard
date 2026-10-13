from src.inference.predictor import get_predictor, risk_level
from tests.conftest import GOOD_TOOL, VAGUE_TOOL, needs_model


def test_risk_level_bands_follow_config():
    levels = {"LOW": 0, "MEDIUM": 25, "HIGH": 50, "CRITICAL": 75}
    assert risk_level(0.10, levels) == "LOW"
    assert risk_level(0.25, levels) == "MEDIUM"
    assert risk_level(0.60, levels) == "HIGH"
    assert risk_level(0.99, levels) == "CRITICAL"
    assert risk_level(0.30, {"LOW": 0, "HIGH": 20}) == "HIGH"


@needs_model
def test_prediction_output_contract():
    r = get_predictor().predict([GOOD_TOOL])[0]
    for k in ("prediction", "probability", "risk_score", "risk_level", "model", "top_features", "recommendations"):
        assert k in r
    assert 0.0 <= r["probability"] <= 1.0
    assert abs(r["risk_score"] - r["probability"] * 100) < 0.11
    assert r["prediction"] == int(r["probability"] >= r["decision_threshold"])
    assert r["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert len(r["top_features"]) > 0 and len(r["recommendations"]) > 0


@needs_model
def test_prediction_is_deterministic_and_batch_consistent():
    p = get_predictor()
    a = p.predict([GOOD_TOOL])[0]["probability"]
    b = p.predict([VAGUE_TOOL, GOOD_TOOL])[1]["probability"]
    assert abs(a - b) < 1e-6


@needs_model
def test_shap_contributions_sum_to_model_logit():
    import math
    p = get_predictor()
    X = p.pipeline.named_steps["features"].transform(p._frame([VAGUE_TOOL]))
    sv, base = p.explainer.shap_values(X)
    prob = p.pipeline.named_steps["model"].predict_proba(X)[0, 1]
    assert abs((base + sv[0].sum()) - math.log(prob / (1 - prob))) < 1e-3


@needs_model
def test_vague_tool_gets_documentation_recommendations():
    r = get_predictor().predict([VAGUE_TOOL])[0]
    titles = {x["title"] for x in r["recommendations"]}
    assert titles & {"Expand the description", "Remove vague wording", "Document every parameter", "Declare parameter types"}
