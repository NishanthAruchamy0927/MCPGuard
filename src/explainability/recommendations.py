"""Rule-based recommendation layer on top of the ML prediction + SHAP drivers.

Each rule inspects the interpretable feature values of the submitted tool;
rules whose feature is also a top SHAP risk driver are promoted to HIGH priority
so the advice follows the model's explanation, not only static heuristics.
"""
from __future__ import annotations

RULES = [
    # (feature, predicate, priority, title, detail)
    ("desc_token_count", lambda v, f: v < 8, "HIGH", "Expand the description",
     "The description is very short. State what the tool does, when to use it, what it returns and any side effects."),
    ("ambiguity_score", lambda v, f: v >= 0.35, "HIGH", "Remove vague wording",
     "Replace vague terms ('handles', 'various', 'data', 'etc.') with explicit behaviour, inputs and constraints."),
    ("param_described_ratio", lambda v, f: f.get("total_parameter_count", 0) > 0 and v < 0.8, "HIGH", "Document every parameter",
     "Add a description to each parameter explaining its meaning, units and accepted values."),
    ("required_documented_ratio", lambda v, f: v < 1.0, "HIGH", "Document the required parameters",
     "Every required parameter should have a description — agents rely on it to construct a valid call."),
    ("hinted_parameter_ratio", lambda v, f: f.get("total_parameter_count", 0) > 0 and v < 0.5, "MEDIUM", "Specify formats and constraints",
     "Declare enum values, patterns, min/max bounds or an explicit format (e.g. 'YYYY-MM-DD', 'City, State') for free-text parameters."),
    ("example_presence", lambda v, f: v < 1 and f.get("total_parameter_count", 0) > 0, "MEDIUM", "Add valid input examples",
     "Include at least one concrete example value (e.g. \"e.g. 'America/New_York'\") — examples sharply reduce formatting errors."),
    ("required_parameter_ratio", lambda v, f: f.get("parameter_count", 0) >= 3 and v == 0, "MEDIUM", "Mark required parameters",
     "The schema has no 'required' list. Declare which parameters are mandatory so agents do not omit them."),
    ("untyped_parameter_count", lambda v, f: v > 0, "MEDIUM", "Declare parameter types",
     "Some parameters have no JSON-schema 'type'. Untyped parameters cause type errors in generated calls."),
    ("nested_schema_depth", lambda v, f: v >= 3, "MEDIUM", "Flatten nested inputs",
     "The input is deeply nested. Consider flattening objects or splitting the tool into smaller tools."),
    ("parameter_count", lambda v, f: v >= 8, "MEDIUM", "Reduce or group parameters",
     "Many parameters increase hallucinated or misplaced arguments. Remove rarely used ones or group related options."),
    ("schema_valid", lambda v, f: v < 1, "HIGH", "Fix the input schema",
     "The input schema is not valid JSON-schema. Agents/clients may fail to parse it before any call is made."),
    ("schema_issue_count", lambda v, f: v > 0 and f.get("schema_valid", 1) >= 1, "MEDIUM", "Resolve schema inconsistencies",
     "The schema has structural issues (e.g. 'required' references an undefined parameter or non-standard types)."),
    ("name_is_generic", lambda v, f: v >= 1, "LOW", "Use a descriptive verb_noun tool name",
     "Generic or single-word names make tool selection harder. Prefer names like 'search_customers' or 'create_invoice'."),
    ("desc_mentions_param_ratio", lambda v, f: f.get("parameter_count", 0) >= 2 and v < 0.3, "LOW", "Reference key parameters in the description",
     "Mention the most important parameters in the tool description so the agent links the request to the arguments."),
    ("constraint_term_count", lambda v, f: v == 0 and f.get("total_parameter_count", 0) > 0, "LOW", "State constraints explicitly",
     "No constraints are mentioned. Describe limits, valid ranges and error conditions."),
]
PRIORITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def recommend(features: dict[str, float], shap_top: list[dict] | None = None, max_items: int = 6) -> list[dict]:
    drivers = {c["feature"].split(":", 1)[-1] for c in (shap_top or []) if c.get("shap", 0) > 0}
    recs = []
    for feat, pred, prio, title, detail in RULES:
        if feat in features and pred(features[feat], features):
            p = "HIGH" if feat in drivers else prio
            recs.append({"title": title, "detail": detail, "priority": p, "feature": feat,
                         "model_driver": feat in drivers})
    recs.sort(key=lambda r: (PRIORITY_ORDER[r["priority"]], not r["model_driver"]))
    if not recs:
        recs.append({"title": "Documentation looks solid", "detail": "No rule-based issues found. Keep descriptions and schemas in sync as the tool evolves.",
                     "priority": "LOW", "feature": None, "model_driver": False})
    return recs[:max_items]


def risk_factors(features: dict[str, float], shap_top: list[dict]) -> list[str]:
    """Short human-readable risk factors from positive SHAP drivers."""
    out = []
    for c in shap_top:
        if c["shap"] <= 0:
            continue
        out.append(f"{c['display_name']} ({c['family_label'].lower()})")
    return out[:5]
