"""Interpretable structured features: lexical, schema, naming and risk scores.

Engineered risk scores (all bounded to [0, 1]; see docs/FEATURES.md):

    ambiguity_score            = clip(5 * vague / max(T,1) + 0.5 * [T < 6], 0, 1)
    missing_constraint_score   = 1 - (#params with enum/pattern/format/bounds/default
                                      or a format hint in their description) / #params
    parameter_complexity_score = 0.5 * log(1+P_all)/log(21) + 0.3 * min(D/4,1)
                                 + 0.2 * (#object + #array) / max(P_all,1)
    schema_complexity_score    = 0.4 * min(P_all/15,1) + 0.3 * min(D/4,1)
                                 + 0.15 * min(E/30,1) + 0.15 * min(H/2.5,1)
    description_quality_score  = mean(min(T/40,1), described_ratio, 1-ambiguity,
                                      min(examples,1), min(constraint_terms/3,1))
    instruction_clarity_score  = 0.5 * min((I + C)/6,1) + 0.25 * min(S/3,1)
                                 + 0.25 * (1 - ambiguity)

T = description tokens, S = sentences, I/C = instruction/constraint terms,
P_all = all (nested) parameters, D = schema depth, E = enum values,
H = parameter-type entropy.
"""
from __future__ import annotations

import math
import re

from src.preprocessing.schema import iter_params, normalize_types, parse_schema, schema_stats
from src.preprocessing.text import FORMAT_HINT_PATTERNS, lexical_stats, normalize_text, split_identifier

VERB_PREFIXES = ("get", "list", "create", "update", "delete", "search", "find", "read", "write", "add", "remove", "set",
                 "fetch", "send", "post", "calculate", "convert", "query", "run", "execute", "generate", "check", "book",
                 "retrieve", "compute", "analyze", "play", "move", "open", "lookup", "schedule")
GENERIC_NAMES = {"run", "do", "process", "handle", "execute", "action", "tool", "call", "invoke", "main", "helper", "util"}
CONSTRAINT_KEYS = ("enum", "pattern", "format", "minimum", "maximum", "minLength", "maxLength", "exclusiveMinimum",
                   "exclusiveMaximum", "const", "minItems", "maxItems", "default", "examples")


def _clip(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def structured_features(tool_name: str, description: str, input_schema) -> dict[str, float]:
    """All interpretable (non-vector) features for one tool."""
    desc = normalize_text(description)
    schema, issues = parse_schema(input_schema)
    schema, n_alias = normalize_types(schema)
    st = schema_stats(schema, issues, n_alias)
    lx = lexical_stats(desc)
    params = list(iter_params(schema))
    T, S = lx["desc_token_count"], lx["desc_sentence_count"]
    P_all = st["total_parameter_count"]
    D = st["nested_schema_depth"]

    hinted = 0
    for _, spec, _, _ in params:
        if any(k in spec for k in CONSTRAINT_KEYS) or FORMAT_HINT_PATTERNS.search(str(spec.get("description", ""))):
            hinted += 1
    props = schema.get("properties") or {}
    required = [r for r in (schema.get("required") or []) if r in props]
    req_doc = sum(1 for r in required if isinstance(props.get(r), dict) and str(props[r].get("description", "")).strip())
    low_desc = desc.lower()
    mentioned = sum(1 for p in props if p.lower() in low_desc or split_identifier(p) in low_desc)

    name_words = split_identifier(tool_name).split()
    ambiguity = _clip(5 * lx["vague_term_count"] / max(T, 1) + 0.5 * (T < 6))
    missing_constraint = _clip(1 - hinted / P_all) if P_all else 0.0
    param_complexity = _clip(0.5 * math.log1p(P_all) / math.log(21) + 0.3 * min(D / 4, 1)
                             + 0.2 * (st["object_parameter_count"] + st["array_parameter_count"]) / max(P_all, 1))
    schema_complexity = _clip(0.4 * min(P_all / 15, 1) + 0.3 * min(D / 4, 1) + 0.15 * min(st["enum_value_total"] / 30, 1)
                              + 0.15 * min(st["parameter_type_entropy"] / 2.5, 1))
    examples = lx["example_mentions"] + st["schema_example_count"]
    quality = (min(T / 40, 1) + st["param_described_ratio"] + (1 - ambiguity) + min(examples, 1)
               + min(lx["constraint_term_count"] / 3, 1)) / 5
    clarity = _clip(0.5 * min((lx["instruction_term_count"] + lx["constraint_term_count"]) / 6, 1)
                    + 0.25 * min(S / 3, 1) + 0.25 * (1 - ambiguity))

    feats = {**lx, **{k: v for k, v in st.items() if k not in ("type_alias_count",)}}
    feats.update({
        "instruction_density": lx["instruction_term_count"] / max(T, 1),
        "constraint_density": lx["constraint_term_count"] / max(T, 1),
        "example_presence": float(examples > 0),
        "required_documented_ratio": req_doc / len(required) if required else 1.0,
        "desc_mentions_param_ratio": mentioned / len(props) if props else 1.0,
        "hinted_parameter_ratio": hinted / P_all if P_all else 1.0,
        "name_token_count": float(len(name_words)),
        "name_char_len": float(len(tool_name or "")),
        "name_has_verb": float(bool(name_words) and name_words[0] in VERB_PREFIXES),
        "name_is_generic": float(bool(name_words) and (name_words[-1] in GENERIC_NAMES or len(name_words) == 1)),
        "name_has_namespace": float("." in (tool_name or "") or "_" in (tool_name or "") and len(name_words) > 3),
        "ambiguity_score": ambiguity,
        "missing_constraint_score": missing_constraint,
        "parameter_complexity_score": param_complexity,
        "schema_complexity_score": schema_complexity,
        "description_quality_score": float(quality),
        "instruction_clarity_score": clarity,
    })
    return {k: float(v) for k, v in feats.items()}


def tool_document(tool_name: str, description: str, input_schema) -> str:
    """Single text rendering of a tool (used by TF-IDF and the sentence encoder)."""
    schema, _ = parse_schema(input_schema)
    schema, _ = normalize_types(schema)
    parts = []
    for path, spec, _, req in iter_params(schema):
        t = spec.get("type", "untyped")
        d = normalize_text(spec.get("description", ""))
        enum = f" one of {spec['enum'][:6]}" if isinstance(spec.get("enum"), list) else ""
        parts.append(f"{split_identifier(path)} ({t}{', required' if req else ''}){': ' + d if d else ''}{enum}")
    params = "; ".join(parts) if parts else "none"
    return f"{split_identifier(tool_name)}: {normalize_text(description)} Parameters: {params}"


FAMILY_OF_PREFIX = re.compile(r"^(desc_|vague_|instruction_|constraint_|example_|format_hint)")
