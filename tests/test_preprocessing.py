import json

import pandas as pd
import pytest

from src.preprocessing.clean import clean
from src.preprocessing.schema import iter_params, normalize_types, parse_schema, schema_stats
from src.preprocessing.text import lexical_stats, normalize_text, split_identifier, tokenize


def test_normalize_text_unicode_and_whitespace():
    assert normalize_text("  Ｓearch​   the\n\tweb  ") == "Search the web"
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""


def test_split_identifier_handles_camel_snake_dots():
    assert split_identifier("getUserProfile") == "get user profile"
    assert split_identifier("math.factorial_v2") == "math factorial v2"


def test_tokenize_and_lexical_stats():
    toks = tokenize("Get the weather, e.g. 'Paris'. Must be ISO format.")
    assert "weather" in toks and "paris" in toks
    st = lexical_stats("Handles various data operations etc.")
    assert st["vague_term_count"] >= 3
    assert st["desc_sentence_count"] == 1
    assert 0 <= st["desc_lexical_diversity"] <= 1


@pytest.mark.parametrize("raw,issue", [
    ("{not json", "invalid_json"), ("", "empty_schema"), (None, "missing_schema"), ([1, 2], "schema_not_object"),
    ({"type": "object", "properties": ["a"]}, "properties_not_mapping"),
    ({"type": "object", "properties": {"a": {}}, "required": ["b"]}, "required_unknown_param"),
])
def test_parse_schema_flags_but_never_raises(raw, issue):
    schema, issues = parse_schema(raw)
    assert isinstance(schema, dict) and isinstance(schema["properties"], dict)
    assert issue in issues


def test_type_alias_normalisation():
    s, n = normalize_types({"type": "dict", "properties": {"x": {"type": "float"}, "y": {"type": "tuple", "items": {"type": "int"}}, "z": {"type": "any"}}})
    assert s["type"] == "object" and s["properties"]["x"]["type"] == "number"
    assert s["properties"]["y"]["items"]["type"] == "integer"
    assert "type" not in s["properties"]["z"]
    assert n == 5


def test_schema_stats_nested_depth_and_counts():
    schema = {"type": "object", "required": ["a"], "properties": {
        "a": {"type": "string", "enum": ["x", "y"], "description": "A"},
        "b": {"type": "object", "properties": {"c": {"type": "array", "items": {"type": "object", "properties": {"d": {"type": "integer"}}}}}},
        "e": {"type": ["string", "null"]}}}
    st = schema_stats(schema)
    assert st["parameter_count"] == 3
    assert st["required_parameter_count"] == 1
    assert st["nested_schema_depth"] >= 3
    assert st["enum_count"] == 1
    assert st["nullable_parameter_count"] == 1
    assert len(list(iter_params(schema))) == 5


def test_clean_removes_duplicates_and_keeps_malformed():
    rows = [
        {"tool_id": "1", "tool_name": "a", "tool_description": "x", "input_schema": "{}", "error_occurred": 1, "lineage_id": "L1"},
        {"tool_id": "2", "tool_name": "a", "tool_description": "x", "input_schema": "{}", "error_occurred": 1, "lineage_id": "L1"},
        {"tool_id": "3", "tool_name": "b", "tool_description": " y ", "input_schema": "{broken", "error_occurred": 0, "lineage_id": "L2"},
        {"tool_id": "4", "tool_name": "c", "tool_description": "z", "input_schema": json.dumps({"type": "object"}), "error_occurred": None, "lineage_id": "L3"},
    ]
    df, rep = clean(pd.DataFrame(rows))
    assert rep["exact_duplicates_removed"] == 1
    assert rep["malformed_schemas"] == 1
    assert rep["dropped_missing_label"] == 1
    assert set(df["tool_id"]) == {"1", "3"}
    assert df.loc[df.tool_id == "3", "tool_description"].item() == "y"


def test_clean_requires_columns():
    with pytest.raises(ValueError):
        clean(pd.DataFrame([{"tool_name": "x"}]))
