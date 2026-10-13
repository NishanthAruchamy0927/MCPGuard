import random

import numpy as np
import pandas as pd

from src.data.augment import OPERATORS, augment
from src.data.harness import CallValidator, SimulatedAgent, value_matches
from src.features.build import LEXICAL_FEATURES, RISK_FEATURES, SCHEMA_FEATURES, MCPFeatureBuilder
from src.features.risk import structured_features, tool_document
from tests.conftest import GOOD_TOOL, VAGUE_TOOL


def _sf(t):
    return structured_features(t["tool_name"], t["description"], t["input_schema"])


def test_structured_features_complete_and_bounded():
    f = _sf(GOOD_TOOL)
    for k in LEXICAL_FEATURES + SCHEMA_FEATURES + RISK_FEATURES:
        assert k in f, k
        assert np.isfinite(f[k])
    for k in RISK_FEATURES:
        assert 0.0 <= f[k] <= 1.0


def test_risk_scores_rank_vague_tool_worse():
    good, vague = _sf(GOOD_TOOL), _sf(VAGUE_TOOL)
    assert vague["ambiguity_score"] > good["ambiguity_score"]
    assert vague["description_quality_score"] < good["description_quality_score"]
    assert vague["untyped_parameter_count"] == 2 and good["untyped_parameter_count"] == 0


def test_tool_document_contains_params():
    doc = tool_document(GOOD_TOOL["tool_name"], GOOD_TOOL["description"], GOOD_TOOL["input_schema"])
    assert "customer id (string, required)" in doc


def test_feature_builder_fit_transform_shapes():
    rows = []
    rng = random.Random(0)
    for i in range(40):
        t = dict(GOOD_TOOL if i % 2 else VAGUE_TOOL)
        rows.append({"tool_name": f"{t['tool_name']}_{i}", "tool_description": t["description"] + f" variant {i} " + rng.choice(["alpha", "beta", "gamma"]),
                     "input_schema": t["input_schema"]})
    df = pd.DataFrame(rows)
    fb = MCPFeatureBuilder(families=("text", "lexical", "schema", "risk"), text_svd_components=8).fit(df)
    X = fb.transform(df)
    assert X.shape == (40, len(fb.feature_names_))
    assert len(fb.feature_names_) == len(fb.feature_families_)
    assert np.isfinite(X).all()
    X2 = fb.transform([{"tool_name": "x", "description": "y", "input_schema": "{bad"}])
    assert X2.shape[1] == X.shape[1]


def test_augmentation_operators_do_not_touch_original():
    import copy
    base = {"tool_name": "get_weather", "description": "Get the weather for a city in the format 'City, State'.",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string", "description": "City"}}, "required": ["city"]}}
    snapshot = copy.deepcopy(base)
    for name, op in OPERATORS.items():
        op(copy.deepcopy(base), random.Random(1))
    new, ops = augment(base, {k: 1.0 for k in OPERATORS}, random.Random(3))
    assert base == snapshot and 1 <= len(ops) <= 2


def test_validator_and_value_matching():
    gold = {"city": ["Paris", "paris, france"], "unit": ["", "celsius"]}
    hidden = {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}
    v = CallValidator()
    assert v.check({"city": "Paris"}, gold, hidden, strict=True) == "NONE"
    assert v.check({}, gold, hidden, strict=True) == "MISSING_REQUIRED_ARG"
    assert v.check({"city": 3}, gold, hidden, strict=True) == "INVALID_ARG_TYPE"
    assert v.check({"city": "Paris", "unit": "kelvin"}, gold, hidden, strict=True) == "INVALID_ARG_FORMAT"
    assert v.check({"city": "Paris", "extra": 1}, gold, hidden, strict=True) == "HALLUCINATED_PARAMETER"
    assert v.check({"city": "Paris", "extra": 1}, gold, hidden, strict=False) == "NONE"
    assert value_matches({"a": 1}, [{"a": [1], "b": ["", 2]}])


def test_simulated_agent_literal_values_are_easier():
    agent = SimulatedAgent({"literal_value_ok": 0.97, "enum_visible_ok": 0.95, "value_mentioned_ok": 0.9, "format_hint_ok": 0.86,
                            "described_ok": 0.76, "undocumented_ok": 0.55, "boolean_described_ok": 0.9, "boolean_undocumented_ok": 0.72,
                            "nested_depth_decay": 0.93}, 0.02)
    q = "what is the weather in paris"
    assert agent.value_prob("Paris", {}, "", q) > agent.value_prob("Lyon", {}, "", q)
    assert agent.value_prob("celsius", {"enum": ["celsius"]}, "", q) > agent.value_prob("celsius", {}, "", q)
