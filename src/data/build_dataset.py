"""Assemble the MCPGuard research dataset.

Pipeline::

    BFCL functions + questions + gold calls ─┐
    MCP reference-server tools + intents  ───┼─> MCP tool records (seed)
                                             │      └─> documented augmentation
                                             └─> controlled agent benchmark ─> labels

Run: ``python -m src.data.build_dataset``
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.data import augment as aug
from src.data.harness import Episode, run_tool, subtask_text
from src.data.intents import synthesize_intents
from src.features.embeddings import EmbeddingEncoder
from src.preprocessing.schema import normalize_types, parse_schema, schema_stats
from src.preprocessing.text import lexical_stats, normalize_text, split_identifier
from src.utils.config import load_config, resolve
from src.utils.io import read_json, set_seed, write_json, write_jsonl
from src.utils.logging_utils import get_logger, log_event

log = get_logger("data.build")


def tool_key(name: str, desc: str, schema) -> str:
    blob = json.dumps([name, desc, schema], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def visible_text(name: str, desc: str, schema) -> str:
    """What the agent sees when routing: name, description and parameter names."""
    params = ""
    if isinstance(schema, dict) and isinstance(schema.get("properties"), dict):
        params = ", ".join(split_identifier(p) for p in schema["properties"])
    return f"{split_identifier(name)}: {normalize_text(desc)} Parameters: {params}"


def _read_lines(path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_bfcl(cfg) -> tuple[dict, dict]:
    """Return (catalog, episodes_by_key) for BFCL functions."""
    raw_dir = cfg.path("raw") / "bfcl"
    prefix = cfg["data"]["bfcl"]["version_prefix"]
    catalog: dict[str, dict] = {}
    episodes: dict[str, list[Episode]] = defaultdict(list)
    for cat in cfg["data"]["bfcl"]["categories"]:
        fpath, apath = raw_dir / f"{prefix}{cat}.json", raw_dir / f"pa_{prefix}{cat}.json"
        if not fpath.exists() or not apath.exists():
            log_event(log, "missing BFCL category, skipped", category=cat)
            continue
        answers = {r["id"]: r["ground_truth"] for r in _read_lines(apath)}
        for entry in _read_lines(fpath):
            funcs = entry.get("function") or []
            keys_by_name = {}
            for f in funcs:
                schema, n_alias = normalize_types(f.get("parameters") or {})
                if isinstance(schema, dict):
                    schema.setdefault("type", "object")
                    schema["type"] = "object"
                k = tool_key(f["name"], f.get("description", ""), schema)
                keys_by_name[f["name"]] = k
                if k not in catalog:
                    fs = entry["id"].rsplit("_", 1)[-1].split("-")
                    server = f["name"].split(".")[0] if "." in f["name"] else (f"bfcl_{cat}_fs{fs[1]}" if len(fs) == 3 else f"bfcl_{cat}")
                    catalog[k] = {
                        "key": k, "tool_name": f["name"], "description": normalize_text(f.get("description", "")),
                        "input_schema": schema, "server_name": server, "source": "bfcl_v3", "bfcl_category": cat,
                        "type_alias_count": n_alias, "provenance": "REAL_TOOL_DEFINITION",
                        "provenance_note": f"BFCL v3 ({cat}); function schema converted to MCP inputSchema.",
                    }
            turns = entry.get("question") or []
            first = turns[0] if turns else []
            query = " ".join(m.get("content", "") for m in first if m.get("role") == "user")
            calls = answers.get(entry["id"], [])
            # Multi-call requests are decomposed: other gold targets of the same
            # request are handled by their own sub-task, so they are not distractors.
            gold_keys = {keys_by_name.get(fn) for call in calls for fn in call}
            for call in calls:
                for fname, gold in call.items():
                    k = keys_by_name.get(fname)
                    if not k or not isinstance(gold, dict):
                        continue
                    siblings = [v for n, v in keys_by_name.items() if v != k and v not in gold_keys]
                    q = normalize_text(query)
                    route = subtask_text(q, gold) if len(gold_keys) > 1 else ""
                    episodes[k].append(Episode(query=q, gold=gold, siblings=siblings, origin="bfcl_real", route_query=route))
    return catalog, episodes


def load_mcp_reference(cfg, rng: random.Random) -> tuple[dict, dict]:
    path = resolve(cfg["data"]["mcp_reference_file"])
    if not path.exists():
        from src.data import mcp_reference
        write_json(mcp_reference.build(), path)
    tools = read_json(path)
    catalog, episodes = {}, {}
    by_server = defaultdict(list)
    for t in tools:
        k = tool_key(t["tool_name"], t["description"], t["input_schema"])
        catalog[k] = {"key": k, "tool_name": t["tool_name"], "description": normalize_text(t["description"]),
                      "input_schema": t["input_schema"], "server_name": t["server_name"], "source": "mcp_reference_servers",
                      "bfcl_category": None, "type_alias_count": 0, "provenance": t["provenance"],
                      "provenance_note": t["provenance_note"]}
        by_server[t["server_name"]].append(k)
    for k, t in catalog.items():
        sib = [s for s in by_server[t["server_name"]] if s != k]
        eps = synthesize_intents(t, 4, rng)
        episodes[k] = [Episode(query=e["query"], gold=e["gold"], siblings=sib, origin="synthetic_intent") for e in eps]
    return catalog, episodes


def build(cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    seed = cfg["seed"]
    set_seed(seed)
    rng = random.Random(seed)
    hcfg = cfg["harness"]

    bfcl_catalog, bfcl_eps = load_bfcl(cfg)
    mcp_catalog, mcp_eps = load_mcp_reference(cfg, rng)
    catalog = {**bfcl_catalog, **mcp_catalog}
    episodes = {**bfcl_eps, **mcp_eps}
    seeds = [k for k in catalog if episodes.get(k)]
    log_event(log, "seed tools", catalog=len(catalog), with_episodes=len(seeds),
              bfcl=sum(catalog[k]["source"] == "bfcl_v3" for k in seeds), mcp=sum(catalog[k]["source"] != "bfcl_v3" for k in seeds))

    # ------------------------------------------------------------ records
    # Near-duplicate definitions of the same function (same name + description,
    # slightly different schema) share one lineage so they never straddle splits.
    family = {k: hashlib.sha1((catalog[k]["tool_name"].lower() + "|" + catalog[k]["description"].lower()).encode()).hexdigest()[:12]
              for k in catalog}
    records: list[dict] = []
    for k in sorted(seeds):
        t = catalog[k]
        records.append({**t, "tool_id": k, "lineage_id": family[k], "parent_tool_id": None, "augmentation_ops": [], "seed_key": k,
                        "hidden_schema": t["input_schema"], "episode_origin": episodes[k][0].origin})
    acfg = cfg["augmentation"]
    if acfg["enabled"]:
        chosen = rng.sample(sorted(seeds), int(len(seeds) * acfg["fraction"]))
        for k in chosen:
            base = {"tool_name": catalog[k]["tool_name"], "description": catalog[k]["description"],
                    "input_schema": catalog[k]["input_schema"]}
            new, ops = aug.augment(base, acfg["operators"], rng)
            nk = tool_key(new["tool_name"], new["description"], new["input_schema"]) + "a"
            records.append({**catalog[k], "key": nk, "tool_id": nk, "lineage_id": family[k], "parent_tool_id": k, "seed_key": k,
                            "description": normalize_text(new["description"]), "input_schema": new["input_schema"],
                            "augmentation_ops": ops, "source": "augmented", "provenance": "SYNTHETIC_AUGMENTED",
                            "provenance_note": f"Variant of {k} produced by operators {ops}; label measured by the controlled benchmark.",
                            "hidden_schema": catalog[k]["input_schema"], "episode_origin": episodes[k][0].origin})
    log_event(log, "records assembled", total=len(records))

    # --------------------------------------------- selection text index
    all_texts = {k: visible_text(t["tool_name"], t["description"], t["input_schema"]) for k, t in catalog.items()}
    for r in records:
        all_texts[r["tool_id"]] = visible_text(r["tool_name"], r["description"],
                                               r["input_schema"] if isinstance(r["input_schema"], dict) else {})
    keys = list(all_texts)
    kidx = {k: i for i, k in enumerate(keys)}
    # The agent "reads" tools and requests through a sentence encoder (semantic routing).
    encoder = EmbeddingEncoder()
    M = encoder.encode([all_texts[k] for k in keys])
    queries = sorted({ep.routing_text() for r in records for ep in episodes[r["seed_key"]]})
    Q = dict(zip(queries, encoder.encode(queries)))
    encoder.save()
    log_event(log, "selection index built", tools=len(keys), queries=len(queries), backend=encoder.backend)
    lineage_of = {r["tool_id"]: r["lineage_id"] for r in records}
    lineage_of.update({k: family[k] for k in catalog if k not in lineage_of})
    name_of = {k: catalog[k]["tool_name"].lower() for k in catalog}
    name_of.update({r["tool_id"]: r["tool_name"].lower() for r in records})
    record_rows = np.array([kidx[r["tool_id"]] for r in records])
    sims = M[record_rows] @ M.T

    # ---------------------------------------------------------- benchmark
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for ri, r in enumerate(records):
        trng = random.Random(f"{seed}-{r['tool_id']}")
        eps = episodes[r["seed_key"]]
        target_name = r["tool_name"].lower()

        def _distinct(c: str) -> bool:  # same-named functions count as the target (BFCL matches by name)
            return lineage_of.get(c) != r["lineage_id"] and name_of.get(c) != target_name

        order = np.argsort(-sims[ri])
        # similar-but-distinct tools; near-functional duplicates (cos >= max_similarity) are not counted
        similar = [keys[j] for j in order if _distinct(keys[j]) and sims[ri, j] < hcfg["max_distractor_similarity"]][: hcfg["n_similar_distractors"]]
        cand_scores, target_idx = [], []
        for ep in eps:
            pool = [r["tool_id"]] + [s for s in ep.siblings if s in kidx and _distinct(s)] + similar
            others = [keys[j] for j in trng.sample(range(len(keys)), 8)]
            pool += [o for o in others if _distinct(o) and o not in pool][: hcfg["n_random_distractors"]]
            pool = list(dict.fromkeys(pool))
            s = M[[kidx[p] for p in pool]] @ Q[ep.routing_text()]
            trng.shuffle(idx := list(range(len(pool))))
            cand_scores.append(s[idx])
            target_idx.append(idx.index(0))
        vis_schema, issues = parse_schema(r["input_schema"])
        parse_ok = not any(i in ("invalid_json", "schema_not_object", "properties_not_mapping") for i in issues) \
            and not (isinstance(r["input_schema"], dict) and r["input_schema"].get("type") == "dict")
        res = run_tool(vis_schema, r["description"], parse_ok, r["hidden_schema"], eps, cand_scores, target_idx, hcfg, trng)
        stats = schema_stats(*parse_schema(r["input_schema"]))
        lex = lexical_stats(r["description"])
        rows.append({
            "tool_id": r["tool_id"], "tool_name": r["tool_name"], "server_name": r["server_name"],
            "tool_description": r["description"],
            "input_schema": r["input_schema"] if isinstance(r["input_schema"], str) else json.dumps(r["input_schema"]),
            **{k: stats[k] for k in ("parameter_count", "required_parameter_count", "optional_parameter_count", "parameter_type_count",
                                     "required_parameter_ratio", "nested_schema_depth", "enum_count", "array_parameter_count",
                                     "object_parameter_count", "nullable_parameter_count")},
            "description_length": lex["desc_char_len"],
            "example_presence": int(lex["example_mentions"] > 0 or stats["schema_example_count"] > 0),
            "example_count": int(lex["example_mentions"] + stats["schema_example_count"]),
            **res,
            "source": r["source"], "provenance": r["provenance"],
            "label_provenance": "CONTROLLED_BENCHMARK",
            "episode_origin": r["episode_origin"], "n_query_templates": len(eps),
            "bfcl_category": r["bfcl_category"], "lineage_id": r["lineage_id"], "parent_tool_id": r["parent_tool_id"],
            "augmentation_ops": "|".join(r["augmentation_ops"]), "type_alias_count": r["type_alias_count"],
            "provenance_note": r["provenance_note"], "timestamp": now,
        })
    df = pd.DataFrame(rows)
    df["error_type_counts"] = df["error_type_counts"].apply(json.dumps)
    write_jsonl(df.to_dict(orient="records"), resolve(cfg["data"]["dataset_file"]))
    df.to_csv(resolve(cfg["data"]["dataset_csv"]), index=False)

    summary = {
        "dataset_version": cfg["data"]["dataset_version"], "built_at": now, "records": len(df),
        "by_source": df["source"].value_counts().to_dict(), "by_provenance": df["provenance"].value_counts().to_dict(),
        "by_episode_origin": df["episode_origin"].value_counts().to_dict(),
        "positive_rate": float(df["error_occurred"].mean()), "error_type": df["error_type"].value_counts().to_dict(),
        "error_severity": df["error_severity"].value_counts().to_dict(), "servers": int(df["server_name"].nunique()),
        "lineages": int(df["lineage_id"].nunique()),
        "augmentation_ops": Counter(op for ops in df["augmentation_ops"] for op in ops.split("|") if op),
        "positive_rate_by_source": df.groupby("source")["error_occurred"].mean().round(4).to_dict(),
        "mean_failure_rate": float(df["failure_rate"].mean()),
        "episodes_per_tool": hcfg["episodes_per_tool"], "failure_rate_threshold": hcfg["failure_rate_threshold"],
    }
    write_json(summary, cfg.path("metadata") / "dataset_summary.json")
    log_event(log, "dataset built", records=len(df), positive_rate=round(summary["positive_rate"], 3))
    return df


if __name__ == "__main__":
    build()
