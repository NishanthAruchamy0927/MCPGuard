"""Controlled agent benchmark: simulated tool-calling agent + call validator.

Why a simulator?
----------------
Per-tool execution logs of production MCP agents are not publicly available.
Following the evaluation protocol of the Berkeley Function-Calling Leaderboard
(AST matching of a generated call against gold calls), every tool is exercised
in ``episodes_per_tool`` episodes by a *documentation-bounded* agent:

1. **Tool selection** – the agent sees the user request and a candidate set
   (sibling tools from the same server / BFCL entry, the most similar tools in
   the catalogue and random tools).  It ranks candidates by TF-IDF cosine
   similarity between the request and each tool's *visible* name+description+
   parameter names and samples from a temperature-scaled softmax.  Vague or
   generic descriptions therefore lead to genuine mis-routing.
2. **Argument construction** – for every gold argument the agent can copy a
   value that is stated literally in the request; otherwise its success
   probability depends on what the *visible* documentation exposes (enum,
   example/format hint, value mentioned, plain description, nothing).  Missing
   types, unmarked required parameters and nesting depth add failure modes.
3. **Validation** – the produced call is checked against the *hidden* backend
   contract (original schema + gold values): missing required arguments, type
   errors, enum/format violations, wrong values and hallucinated parameters on
   strict servers are reported.

All probabilities are explicit in ``configs/config.yaml -> harness`` so the
benchmark is fully reproducible and auditable.  The resulting labels are
*CONTROLLED_BENCHMARK* observations – not real production logs – and are
labelled as such everywhere in the project.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.preprocessing.text import FORMAT_HINT_PATTERNS, split_identifier

ERROR_TYPES = [
    "NONE", "WRONG_TOOL_SELECTION", "MISSING_REQUIRED_ARG", "INVALID_ARG_TYPE",
    "INVALID_ARG_FORMAT", "INVALID_ARG_VALUE", "HALLUCINATED_PARAMETER", "SCHEMA_PARSE_ERROR",
]
HALLUCINATED_NAMES = ["format", "limit", "verbose", "lang", "timeout", "options", "units", "filter"]
_MISSING = object()


def first_value(allowed):
    """Pick the canonical gold value from a BFCL allowed-values list."""
    if isinstance(allowed, list):
        for v in allowed:
            if v not in ("", None) and v != []:
                return realise(v)
        return _MISSING
    return realise(allowed)


def realise(v):
    if isinstance(v, dict):
        out = {}
        for k, allowed in v.items():
            fv = first_value(allowed) if isinstance(allowed, list) else allowed
            if fv is not _MISSING:
                out[k] = fv
        return out
    return v


def is_optional(allowed) -> bool:
    return isinstance(allowed, list) and any(v in ("", None) for v in allowed)


def _norm(v):
    if isinstance(v, str):
        return re.sub(r"\s+", " ", v.strip().lower())
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def value_matches(value, allowed) -> bool:
    """BFCL-style match of a produced value against the allowed list."""
    cands = allowed if isinstance(allowed, list) else [allowed]
    for a in cands:
        if isinstance(a, dict) and isinstance(value, dict):
            ok = True
            for k, sub in a.items():
                if k not in value:
                    if not is_optional(sub):
                        ok = False
                        break
                    continue
                if not value_matches(value[k], sub if isinstance(sub, list) else [sub]):
                    ok = False
                    break
            if ok:
                return True
        elif _norm(a) == _norm(value):
            return True
    return False


def _type_ok(value, t) -> bool:
    if t is None:
        return True
    if isinstance(t, list):
        return any(_type_ok(value, x) for x in t)
    return {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
        "null": value is None,
    }.get(t, True)


@dataclass
class Episode:
    query: str
    gold: dict
    siblings: list
    origin: str
    route_query: str = ""   # sub-task text used for routing (multi-call requests)

    def routing_text(self) -> str:
        return self.route_query or self.query


def subtask_text(query: str, gold: dict) -> str:
    """Decomposition proxy for multi-call requests: keep the clauses of the
    request that mention this call's literal gold values."""
    clauses = [c for c in re.split(r"(?<=[.!?;])\s+|,\s+(?:and\s+)?(?:then|also|after that|next|finally)\b|\b(?:then|after that|finally),?\s", query) if c and c.strip()]
    vals = []
    for allowed in gold.values():
        v = first_value(allowed)
        if isinstance(v, (str, int, float)) and not isinstance(v, bool) and str(v).strip():
            vals.append(str(v).strip().lower())
    hits = [c for c in clauses if any(v in c.lower() for v in vals)]
    return " ".join(hits) if hits else query


class CallValidator:
    """Validate a produced call against the hidden backend contract."""

    def check(self, args: dict, gold: dict, hidden_schema: dict, strict: bool) -> str:
        props = hidden_schema.get("properties") or {}
        required = set(hidden_schema.get("required") or [])
        for p, allowed in gold.items():
            needed = (p in required) or not is_optional(allowed)
            if needed and first_value(allowed) is not _MISSING and p not in args:
                return "MISSING_REQUIRED_ARG"
        for p in args:
            if p not in props and p not in gold:
                if strict:
                    return "HALLUCINATED_PARAMETER"
        for p, v in args.items():
            spec = props.get(p) if isinstance(props.get(p), dict) else {}
            if not _type_ok(v, spec.get("type")):
                return "INVALID_ARG_TYPE"
            if isinstance(spec.get("enum"), list) and v not in spec["enum"] and not isinstance(v, (dict, list)):
                return "INVALID_ARG_FORMAT"
            if p in gold and not value_matches(v, gold[p]):
                return "INVALID_ARG_FORMAT" if (spec.get("enum") or spec.get("pattern") or spec.get("format")) else "INVALID_ARG_VALUE"
        return "NONE"


class SimulatedAgent:
    """A documentation-bounded tool-calling agent (see module docstring)."""

    def __init__(self, probs: dict, temperature: float):
        self.p = probs
        self.temperature = temperature

    # ------------------------------------------------------------ selection
    def select(self, scores: np.ndarray, rng: random.Random) -> int:
        z = scores / self.temperature
        z = z - z.max()
        w = np.exp(z)
        w = w / w.sum()
        return int(rng.choices(range(len(w)), weights=w.tolist(), k=1)[0])

    # ------------------------------------------------------------ arguments
    @staticmethod
    def _literal(v, q: str) -> bool:
        if isinstance(v, bool) or v is None:
            return False
        if isinstance(v, (int, float)):
            s = str(int(v)) if float(v).is_integer() else str(v)
            return re.search(rf"(?<![\d.]){re.escape(s)}(?![\d])", q) is not None
        if isinstance(v, str):
            s = v.strip().lower()
            return bool(s) and s in q
        if isinstance(v, list):
            return bool(v) and all(SimulatedAgent._literal(x, q) for x in v)
        return False

    def value_prob(self, v, spec: dict, tool_desc: str, q: str, depth: int = 1) -> float:
        p = self.p
        spec = spec if isinstance(spec, dict) else {}
        decay = p["nested_depth_decay"] ** (depth - 1)
        if isinstance(v, dict):
            sub = spec.get("properties") if isinstance(spec.get("properties"), dict) else {}
            prob = 1.0
            for k, sv in v.items():
                prob *= self.value_prob(sv, sub.get(k, {}), tool_desc, q, depth + 1)
            return prob
        if isinstance(v, list) and v and not self._literal(v, q):
            item_spec = spec.get("items") if isinstance(spec.get("items"), dict) else {}
            probs = [self.value_prob(x, item_spec or {"description": spec.get("description", "")}, tool_desc, q, depth + 1) for x in v[:4]]
            return float(np.prod(probs) ** (1 / math.sqrt(len(probs))))
        desc = str(spec.get("description", ""))
        if isinstance(v, bool):
            return (p["boolean_described_ok"] if len(desc) >= 8 else p["boolean_undocumented_ok"]) * decay
        if self._literal(v, q):
            return p["literal_value_ok"] * decay
        enum = spec.get("enum")
        if isinstance(enum, list) and _norm(v) in [_norm(e) for e in enum]:
            return p["enum_visible_ok"] * decay
        sv = str(v).strip().lower()
        if len(sv) > 1 and (sv in desc.lower() or sv in tool_desc.lower()):
            return p["value_mentioned_ok"] * decay
        if FORMAT_HINT_PATTERNS.search(desc) or "examples" in spec or "default" in spec:
            return p["format_hint_ok"] * decay
        if len(desc) >= 8:
            return p["described_ok"] * decay
        return p["undocumented_ok"] * decay

    @staticmethod
    def corrupt(v, rng: random.Random):
        if isinstance(v, bool):
            return not v
        if isinstance(v, (int, float)):
            return v * rng.choice([10, 0.1, -1, 2]) + rng.choice([0, 1])
        if isinstance(v, str):
            return rng.choice([v.upper() + " ", f"approx. {v}", v.split(",")[0] + "_alt", (v[: max(1, len(v) // 2)] + "?")])
        if isinstance(v, list):
            return (v[:-1] + ["unknown"]) if v else ["unknown"]
        if isinstance(v, dict):
            d = dict(v)
            if d:
                d.pop(rng.choice(list(d)))
            d["unspecified"] = "?"
            return d
        return "?"

    @staticmethod
    def wrong_type(v):
        if isinstance(v, str):
            return {"value": v}
        if isinstance(v, (list, dict)):
            return str(v)
        return str(v)

    def build_call(self, visible: dict, visible_desc: str, gold: dict, query: str, rng: random.Random) -> dict:
        p = self.p
        q = query.lower()
        props = visible.get("properties") if isinstance(visible.get("properties"), dict) else {}
        vis_required = set(visible.get("required") or []) if isinstance(visible.get("required"), list) else set()
        desc_low = visible_desc.lower()
        args: dict = {}
        for name, allowed in gold.items():
            v = first_value(allowed)
            optional = is_optional(allowed)
            if v is _MISSING:
                continue
            spec = props.get(name) if isinstance(props.get(name), dict) else None
            if spec is None:  # parameter not documented in the visible schema
                named = split_identifier(name) in desc_low or name.lower() in desc_low
                if rng.random() >= (p["include_invisible_param_named"] if named else p["include_invisible_param_unnamed"]):
                    continue
                spec = {}
            elif name not in vis_required and not optional and rng.random() < p["omit_when_not_marked_required"] and not self._literal(v, q):
                continue
            if "type" not in spec and not isinstance(v, str):
                pt = p["type_error_untyped_container"] if isinstance(v, (list, dict)) else p["type_error_untyped"]
                if rng.random() < pt:
                    args[name] = self.wrong_type(v)
                    continue
            ok = rng.random() < self.value_prob(v, spec, visible_desc, q)
            args[name] = v if ok else self.corrupt(v, rng)
        # hallucinated / noise parameters
        n_opt = sum(1 for k in props if k not in vis_required)
        ph = p["hallucination_base"] + p["hallucination_many_optional"] * (n_opt >= 6) + p["hallucination_short_description"] * (len(visible_desc) < 40)
        if rng.random() < ph:
            extra = [k for k in props if k not in gold] or HALLUCINATED_NAMES
            args[rng.choice(extra)] = "auto"
        return args


def run_tool(visible_schema: dict, visible_desc: str, schema_parse_ok: bool, hidden_schema: dict,
             episodes: list[Episode], candidate_scores: list[np.ndarray], target_index: list[int],
             cfg: dict, rng: random.Random) -> dict:
    """Run ``episodes_per_tool`` episodes for one tool and aggregate the outcome."""
    probs = cfg["probabilities"]
    agent = SimulatedAgent(probs, cfg["selection_temperature"])
    validator = CallValidator()
    strict = rng.random() < probs["strict_server"]
    n = cfg["episodes_per_tool"]
    outcomes: list[str] = []
    for i in range(n):
        ei = i % len(episodes)
        ep = episodes[ei]
        if not schema_parse_ok and rng.random() < probs["malformed_schema_fail"]:
            outcomes.append("SCHEMA_PARSE_ERROR")
            continue
        chosen = agent.select(candidate_scores[ei], rng)
        if chosen != target_index[ei]:
            outcomes.append("WRONG_TOOL_SELECTION")
            continue
        args = agent.build_call(visible_schema if schema_parse_ok else {}, visible_desc, ep.gold, ep.query, rng)
        outcomes.append(validator.check(args, ep.gold, hidden_schema, strict))
    fails = [o for o in outcomes if o != "NONE"]
    counts = Counter(outcomes)
    failure_rate = len(fails) / n
    dominant = Counter(fails).most_common(1)[0][0] if fails else "NONE"
    severity = "NONE"
    for level, upper in cfg["severity_bins"].items():
        if failure_rate < upper:
            severity = level
            break
    return {
        "execution_count": n,
        "failure_count": len(fails),
        "failure_rate": failure_rate,
        "agent_success_rate": 1 - failure_rate,
        "error_occurred": int(failure_rate >= cfg["failure_rate_threshold"]),
        "error_type": dominant,
        "error_severity": severity,
        "error_type_counts": dict(counts),
        "strict_server": int(strict),
    }
