"""Task-intent generation for the controlled agent benchmark.

BFCL tools already come with real user questions and gold calls.  The curated
MCP reference tools do not, so for them we synthesise *intents*: a natural-
language request plus the gold argument values the agent must produce.  Values
are drawn from a name-aware generator; each value is either stated literally in
the request (the agent can copy it) or left implicit (the agent has to rely on
the tool's documentation).  Intent episodes are flagged ``origin="synthetic"``.
"""
from __future__ import annotations

import random
import re

from src.preprocessing.text import sentences, split_identifier

NAME_VALUES: list[tuple[str, list]] = [
    (r"repo_?path", ["/home/dev/projects/webapp", "/srv/git/api-server", "C:/code/analytics"]),
    (r"path|file|dir|destination|source", ["/home/dev/notes/todo.md", "/data/reports/q3_summary.csv", "src/main.py"]),
    (r"url|link|uri", ["https://example.com/docs/intro", "https://news.ycombinator.com", "https://arxiv.org/abs/1706.03762"]),
    (r"owner|org", ["octocat", "modelcontextprotocol", "openai"]),
    (r"repo", ["hello-world", "servers", "mcpguard"]),
    (r"branch", ["main", "feature/login-page", "release-2.1"]),
    (r"time", ["14:30", "09:15", "18:00"]),
    (r"timezone", ["America/New_York", "Europe/London", "Asia/Kolkata"]),
    (r"date|since", ["2025-03-14", "2024-12-01T00:00:00Z", "2025-07-04"]),
    (r"channel", ["C0123ABCDEF", "C9F8E7D6C5"]),
    (r"thread_ts|timestamp", ["1712345678.123456", "1699999999.000100"]),
    (r"user_?id|id$", ["U02ABCDEF", "48213", "place_ChIJN1t_tDeuEmsR"]),
    (r"email", ["ana@example.com", "dev.team@example.org"]),
    (r"selector", ["#submit-button", "input[name='email']", ".nav > a.login"]),
    (r"query|q$|pattern|search", ["vector database benchmarks", "TODO", "language:python stars:>100"]),
    (r"sql", ["SELECT name FROM users WHERE active = 1", "SELECT COUNT(*) FROM orders"]),
    (r"message|text|body|content|comment|insight|thought|prompt|script", ["Fix the null pointer in the parser", "Weekly status update attached", "console.log(document.title)"]),
    (r"title|name|entity|key", ["Quarterly Report", "deploy-bot", "customer:42"]),
    (r"address|origin|location", ["1600 Amphitheatre Parkway, Mountain View, CA", "221B Baker Street, London"]),
    (r"lat", [37.4221, 51.5237]),
    (r"lon|lng", [-122.0841, -0.1585]),
    (r"reaction|emoji", ["thumbsup", "rocket"]),
    (r"label", ["bug", "enhancement"]),
]
VERB_PREFIXES = ["Please {x}", "Can you {x}?", "I need you to {x}", "{x} for me", "Go ahead and {x}", "Help me {x}"]


def example_value(name: str, spec: dict, rng: random.Random):
    """A plausible value for parameter ``name`` given its JSON-schema ``spec``."""
    t = spec.get("type")
    if isinstance(spec.get("enum"), list) and spec["enum"]:
        return rng.choice(spec["enum"])
    if t == "boolean":
        return rng.choice([True, False])
    if t in ("integer", "number"):
        lo = spec.get("minimum", 1)
        hi = spec.get("maximum", 50)
        try:
            v = rng.randint(int(lo), max(int(lo), int(min(hi, 50))))
        except (TypeError, ValueError):
            v = rng.randint(1, 50)
        return v
    if t == "array":
        items = spec.get("items") if isinstance(spec.get("items"), dict) else {"type": "string"}
        return [example_value(name.rstrip("s"), items, rng) for _ in range(rng.randint(1, 2))]
    if t == "object":
        props = spec.get("properties") or {}
        req = spec.get("required") or list(props)[:2]
        return {k: example_value(k, props.get(k, {}), rng) for k in req if k in props}
    low = name.lower()
    for pat, values in NAME_VALUES:
        if re.search(pat, low):
            return rng.choice(values)
    return rng.choice(["alpha", "report-2025", "default", "sample value"])


def _mention(name: str, value) -> str:
    words = split_identifier(name)
    if isinstance(value, bool):
        return f"{'with' if value else 'without'} {words}"
    if isinstance(value, (list, dict)):
        return f"using {words} {value}"
    return f"{words} {value}" if len(str(value)) < 20 else f"'{value}'"


def synthesize_intents(tool: dict, n: int, rng: random.Random) -> list[dict]:
    """Generate ``n`` intents for a tool with an MCP-style ``input_schema``."""
    schema = tool["input_schema"] if isinstance(tool["input_schema"], dict) else {}
    props = schema.get("properties") or {}
    required = [r for r in (schema.get("required") or []) if r in props]
    optional = [p for p in props if p not in required]
    first = (sentences(tool["description"]) or [split_identifier(tool["tool_name"])])[0]
    words = [w for w in re.findall(r"[A-Za-z']+", first)]
    episodes = []
    for _ in range(n):
        kept = [w for w in words if rng.random() > 0.35] or words[:3]
        phrase = " ".join(kept[:14]).lower()
        gold: dict[str, list] = {}
        mentions = []
        chosen = required + [p for p in optional if rng.random() < 0.3]
        for p in chosen:
            v = example_value(p, props.get(p, {}) if isinstance(props.get(p), dict) else {}, rng)
            gold[p] = [v]
            if rng.random() < 0.72:
                mentions.append(_mention(p, v))
        for p in optional:
            if p not in gold:
                gold[p] = [""]  # omitting optional params is acceptable
        query = rng.choice(VERB_PREFIXES).format(x=phrase)
        if mentions:
            query += " — " + ", ".join(mentions)
        episodes.append({"query": query, "gold": gold, "origin": "synthetic"})
    return episodes
