"""JSON-schema parsing, repair and structural analysis for MCP ``inputSchema``.

Design decisions (documented in docs/PREPROCESSING.md):

* Schemas may arrive as dicts or JSON strings; unparsable strings are kept and
  flagged (``schema_valid = 0``) instead of dropping the record, because a
  malformed schema is itself an important risk signal.
* Non-standard type aliases found in real tool catalogues (``dict``, ``float``,
  ``tuple``, ``int``, ``bool``, ``str``, ``list``, ``any``) are mapped to JSON-
  schema types and counted as ``type_alias_count``.
* Missing ``properties`` / ``required`` are treated as empty (count = 0) and
  recorded through the ``schema_issue_count`` feature.
"""
from __future__ import annotations

import copy
import json
import math
from collections import Counter
from typing import Any, Iterator

TYPE_ALIASES = {
    "dict": "object", "map": "object", "float": "number", "double": "number", "int": "integer", "long": "integer",
    "bool": "boolean", "str": "string", "tuple": "array", "list": "array", "any": None, "String": "string",
    "Integer": "integer", "Boolean": "boolean",
}
JSON_TYPES = {"string", "number", "integer", "boolean", "object", "array", "null"}


def parse_schema(raw: Any) -> tuple[dict, list[str]]:
    """Return (schema_dict, issues). Never raises."""
    issues: list[str] = []
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return {"type": "object", "properties": {}}, ["missing_schema"]
    if isinstance(raw, str):
        txt = raw.strip()
        if not txt:
            return {"type": "object", "properties": {}}, ["empty_schema"]
        try:
            raw = json.loads(txt)
        except json.JSONDecodeError:
            return {"type": "object", "properties": {}}, ["invalid_json"]
    if not isinstance(raw, dict):
        return {"type": "object", "properties": {}}, ["schema_not_object"]
    schema = copy.deepcopy(raw)
    if schema.get("type") not in (None, "object"):
        if schema.get("type") in TYPE_ALIASES and TYPE_ALIASES[schema["type"]] == "object":
            issues.append("type_alias_root")
        else:
            issues.append("root_not_object")
    props = schema.get("properties")
    if props is None:
        if schema.get("type") in ("object", "dict", None) and "properties" not in schema:
            issues.append("missing_properties")
        schema["properties"] = {}
    elif not isinstance(props, dict):
        issues.append("properties_not_mapping")
        schema["properties"] = {}
    req = schema.get("required", [])
    if not isinstance(req, list):
        issues.append("required_not_list")
        schema["required"] = []
    else:
        unknown = [r for r in req if r not in schema["properties"]]
        if unknown:
            issues.append("required_unknown_param")
    return schema, issues


def normalize_types(schema: dict) -> tuple[dict, int]:
    """Map type aliases to JSON-schema types (recursive). Returns (schema, n_aliases)."""
    count = 0

    def _walk(node: Any) -> Any:
        nonlocal count
        if isinstance(node, dict):
            node = dict(node)
            t = node.get("type")
            if isinstance(t, str) and t not in JSON_TYPES:
                count += 1
                mapped = TYPE_ALIASES.get(t, TYPE_ALIASES.get(t.lower()))
                if mapped is None:
                    node.pop("type", None)
                else:
                    node["type"] = mapped
            for key in ("properties",):
                if isinstance(node.get(key), dict):
                    node[key] = {k: _walk(v) for k, v in node[key].items()}
            if "items" in node:
                node["items"] = _walk(node["items"])
            for key in ("oneOf", "anyOf", "allOf"):
                if isinstance(node.get(key), list):
                    node[key] = [_walk(v) for v in node[key]]
            return node
        return node

    return _walk(schema), count


def iter_params(schema: dict, depth: int = 1, prefix: str = "") -> Iterator[tuple[str, dict, int, bool]]:
    """Yield (path, spec, depth, is_required) for every (nested) parameter."""
    props = schema.get("properties") or {}
    required = set(schema.get("required") or []) if isinstance(schema.get("required"), list) else set()
    if not isinstance(props, dict):
        return
    for name, spec in props.items():
        spec = spec if isinstance(spec, dict) else {}
        path = f"{prefix}{name}"
        yield path, spec, depth, name in required
        if spec.get("type") == "object" or isinstance(spec.get("properties"), dict):
            yield from iter_params(spec, depth + 1, path + ".")
        items = spec.get("items")
        if isinstance(items, dict) and (items.get("type") == "object" or isinstance(items.get("properties"), dict)):
            yield from iter_params(items, depth + 1, path + "[].")


def max_depth(node: Any, d: int = 0) -> int:
    if isinstance(node, dict):
        best = d
        if isinstance(node.get("properties"), dict) and node["properties"]:
            best = max(best, max(max_depth(v, d + 1) for v in node["properties"].values()))
        if isinstance(node.get("items"), dict):
            best = max(best, max_depth(node["items"], d + 1))
        return best
    return d


def schema_stats(schema: dict, issues: list[str] | None = None, alias_count: int = 0) -> dict[str, float]:
    """Structural statistics of an (already parsed) input schema."""
    props = schema.get("properties") or {}
    top_required = [r for r in (schema.get("required") or []) if r in props]
    params = list(iter_params(schema))
    def _kind(spec: dict) -> str:
        t = spec.get("type")
        if t is None:
            return "union" if any(k in spec for k in ("oneOf", "anyOf", "allOf")) else "untyped"
        return t if isinstance(t, str) else "union"

    types = Counter(_kind(spec) for _, spec, _, _ in params)
    n_params = len(props)
    n_all = len(params)
    total = sum(types.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in types.values()) if total else 0.0
    described = sum(bool(str(spec.get("description", "")).strip()) for _, spec, _, _ in params)
    enum_params = [spec for _, spec, _, _ in params if isinstance(spec.get("enum"), list)]
    nullable = sum(
        1 for _, spec, _, _ in params
        if spec.get("nullable") is True or (isinstance(spec.get("type"), list) and "null" in spec["type"])
        or any(isinstance(o, dict) and o.get("type") == "null" for o in (spec.get("anyOf") or []) + (spec.get("oneOf") or []))
    )
    constrained = sum(
        1 for _, spec, _, _ in params
        if any(k in spec for k in ("enum", "pattern", "format", "minimum", "maximum", "minLength", "maxLength",
                                   "exclusiveMinimum", "exclusiveMaximum", "const", "minItems", "maxItems"))
    )
    with_default = sum(1 for _, spec, _, _ in params if "default" in spec)
    with_examples = sum(1 for _, spec, _, _ in params if "examples" in spec or "example" in spec)
    desc_lens = [len(str(spec.get("description", ""))) for _, spec, _, _ in params]
    return {
        "parameter_count": float(n_params),
        "nested_parameter_count": float(n_all - n_params),
        "total_parameter_count": float(n_all),
        "required_parameter_count": float(len(top_required)),
        "optional_parameter_count": float(n_params - len(top_required)),
        "required_parameter_ratio": len(top_required) / n_params if n_params else 0.0,
        "parameter_type_count": float(len([t for t in types if t != "untyped"])),
        "parameter_type_entropy": float(entropy),
        "untyped_parameter_count": float(types.get("untyped", 0)),
        "nested_schema_depth": float(max_depth(schema)),
        "object_parameter_count": float(types.get("object", 0)),
        "array_parameter_count": float(types.get("array", 0)),
        "enum_count": float(len(enum_params)),
        "enum_value_total": float(sum(len(s["enum"]) for s in enum_params)),
        "nullable_parameter_count": float(nullable),
        "constrained_parameter_count": float(constrained),
        "constrained_parameter_ratio": constrained / n_all if n_all else 0.0,
        "default_value_count": float(with_default),
        "schema_example_count": float(with_examples),
        "param_described_ratio": described / n_all if n_all else 1.0,
        "param_desc_mean_len": float(sum(desc_lens) / len(desc_lens)) if desc_lens else 0.0,
        "schema_issue_count": float(len(issues or [])),
        "schema_valid": float(not any(i in ("invalid_json", "schema_not_object", "properties_not_mapping") for i in (issues or []))),
        "type_alias_count": float(alias_count),
        "schema_char_len": float(len(json.dumps(schema))),
    }
