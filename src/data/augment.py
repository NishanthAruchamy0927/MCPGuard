"""Documented augmentation operators for MCP tool definitions.

Each operator rewrites only the *visible* tool (what an agent would be shown);
the backend contract (hidden schema) is never changed.  Records produced here are
labelled ``source = "augmented"`` / ``provenance = "SYNTHETIC_AUGMENTED"`` and keep
the ``lineage_id`` of the seed tool so they never cross the train/test split
boundary.  Labels for augmented records are **not** assigned here – they are
measured by running the controlled agent benchmark on the modified tool.
"""
from __future__ import annotations

import copy
import random
from typing import Callable

from src.data.intents import example_value
from src.preprocessing.text import split_identifier

VAGUE_TEMPLATES = [
    "Handles {n} operations.",
    "Tool for {n}.",
    "Performs the requested {n} action.",
    "Utility to manage {n} data as needed.",
    "Processes {n} information.",
    "Use this when needed.",
    "Does {n} stuff.",
]
NOISE_PARAMS = ["options", "mode", "flag", "config", "extra", "context", "settings", "meta", "opts", "params"]


def _walk_props(schema: dict, fn: Callable[[str, dict], None]) -> None:
    props = schema.get("properties")
    if isinstance(props, dict):
        for name, spec in props.items():
            if isinstance(spec, dict):
                fn(name, spec)
                _walk_props(spec, fn)
                if isinstance(spec.get("items"), dict):
                    _walk_props(spec["items"], fn)


def strip_param_descriptions(tool: dict, rng: random.Random) -> dict:
    _walk_props(tool["input_schema"], lambda n, s: s.pop("description", None))
    return tool


def drop_required_list(tool: dict, rng: random.Random) -> dict:
    tool["input_schema"].pop("required", None)
    return tool


def vague_description(tool: dict, rng: random.Random) -> dict:
    words = split_identifier(tool["tool_name"]).split()
    noun = " ".join(words[-2:]) if words else "item"
    tool["description"] = rng.choice(VAGUE_TEMPLATES).format(n=noun)
    return tool


def truncate_description(tool: dict, rng: random.Random) -> dict:
    words = tool["description"].split()
    keep = max(3, int(len(words) * rng.uniform(0.2, 0.4)))
    tool["description"] = " ".join(words[:keep])
    return tool


def remove_enums(tool: dict, rng: random.Random) -> dict:
    _walk_props(tool["input_schema"], lambda n, s: s.pop("enum", None))
    return tool


def remove_types(tool: dict, rng: random.Random) -> dict:
    def _f(n: str, s: dict) -> None:
        if s.get("type") not in ("object", "array") or rng.random() < 0.5:
            s.pop("type", None)
    _walk_props(tool["input_schema"], _f)
    return tool


def add_noise_params(tool: dict, rng: random.Random) -> dict:
    props = tool["input_schema"].setdefault("properties", {})
    for name in rng.sample(NOISE_PARAMS, rng.randint(3, 6)):
        if name not in props:
            props[name] = {"type": rng.choice(["string", "object", "string", "boolean"])}
    return tool


def malform_schema(tool: dict, rng: random.Random) -> dict:
    choice = rng.randint(0, 2)
    if choice == 0:  # stringified, truncated JSON
        import json
        txt = json.dumps(tool["input_schema"])
        tool["input_schema"] = txt[: max(10, int(len(txt) * 0.7))]
    elif choice == 1:
        tool["input_schema"]["properties"] = list((tool["input_schema"].get("properties") or {}).keys())
    else:
        tool["input_schema"]["required"] = (tool["input_schema"].get("required") or []) + ["undefined_param"]
        tool["input_schema"]["type"] = "dict"
    return tool


def enrich_documentation(tool: dict, rng: random.Random) -> dict:
    """Positive augmentation: add parameter descriptions, enum listings and examples."""
    schema = tool["input_schema"]
    req = set(schema.get("required") or [])
    notes = []

    def _f(name: str, s: dict) -> None:
        words = split_identifier(name)
        t = s.get("type", "string")
        if not str(s.get("description", "")).strip():
            s["description"] = f"The {words} ({t})."
        if isinstance(s.get("enum"), list) and s["enum"]:
            s["description"] += " One of: " + ", ".join(f"'{v}'" for v in s["enum"][:8]) + "."
        elif t in ("string", "integer", "number") and rng.random() < 0.8:
            ex = example_value(name, s, rng)
            s["description"] += f" e.g. '{ex}'."
    _walk_props(schema, _f)
    if req:
        notes.append("Required parameters: " + ", ".join(sorted(req)) + ".")
    tool["description"] = tool["description"].rstrip(". ") + ". " + " ".join(notes) if notes else tool["description"]
    return tool


OPERATORS: dict[str, Callable[[dict, random.Random], dict]] = {
    "strip_param_descriptions": strip_param_descriptions,
    "drop_required_list": drop_required_list,
    "vague_description": vague_description,
    "truncate_description": truncate_description,
    "remove_enums": remove_enums,
    "remove_types": remove_types,
    "add_noise_params": add_noise_params,
    "malform_schema": malform_schema,
    "enrich_documentation": enrich_documentation,
}


def augment(tool: dict, weights: dict[str, float], rng: random.Random) -> tuple[dict, list[str]]:
    """Apply 1–2 sampled operators to a deep copy of ``tool``."""
    names = list(weights)
    w = [weights[n] for n in names]
    k = 1 if rng.random() < 0.65 else 2
    ops: list[str] = []
    while len(ops) < k:
        op = rng.choices(names, weights=w, k=1)[0]
        if op not in ops:
            ops.append(op)
    if "enrich_documentation" in ops and len(ops) > 1:  # do not mix positive and negative edits
        ops = ["enrich_documentation"]
    if "malform_schema" in ops:  # malformation always applied last
        ops = [o for o in ops if o != "malform_schema"] + ["malform_schema"]
    new = copy.deepcopy(tool)
    for op in ops:
        new = OPERATORS[op](new, rng)
    return new, ops
