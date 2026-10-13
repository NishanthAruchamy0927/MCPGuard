# Dataset Card — MCPGuard MCP Tool Risk Dataset (v1.0)

> Exact counts are generated in `data/metadata/dataset_summary.json`, `data_quality_report.json` and `split_report.json`, and shown on the dashboard's **Dataset Explorer**.

## Sources and collection method
| Source | Provenance label | Collection |
|---|---|---|
| **BFCL v3**: `gorilla-llm/Berkeley-Function-Calling-Leaderboard` (Hugging Face, Apache-2.0), categories simple, multiple, parallel, parallel_multiple, live_simple, live_multiple, live_parallel, live_parallel_multiple | `REAL_TOOL_DEFINITION` | `python -m src.data.download` fetches function files **and** `possible_answer` gold calls, with SHA-256 checksums in `data/metadata/download_manifest.json`. Functions are converted to MCP `{name, description, inputSchema}`; type aliases (`dict`, `float`, `tuple`, …) are normalised |
| **MCP reference servers**: filesystem, git, github, fetch, memory, sqlite, time, brave-search, puppeteer, slack, google-maps, everything, sequential-thinking, … | `REAL_TOOL_DEFINITION` | Transcribed from public documentation (`src/data/mcp_reference.py`). These may differ in minor details from the latest upstream |
| **Augmentation** | `SYNTHETIC_AUGMENTED` | Nine documented operators (`src/data/augment.py`): strip parameter descriptions, drop `required`, vague description, truncate, remove enums, remove types, add noise parameters, malform schema, and *enrich documentation* (a positive edit). Variants keep their seed's `lineage_id` |

Only BFCL functions that are the gold target of at least one real question are included. Near-duplicate definitions (same name and description) share a lineage.

## Labels (`label_provenance = CONTROLLED_BENCHMARK`)
Real per-tool production failure logs are not public, so every label is **measured** by a controlled benchmark (`src/data/harness.py`):
- 10 episodes per tool. BFCL tools use their real user questions and gold calls; MCP tools use synthetic intents (`episode_origin = synthetic_intent`).
- Each episode: semantic routing among co-loaded tools → argument construction bounded by the visible documentation → validation against the hidden schema and gold values (BFCL AST-style).
- `failure_rate` = failed episodes / 10; `agent_success_rate` = 1 − failure_rate; `error_occurred` = failure_rate ≥ 0.30; `error_severity` ∈ {NONE, LOW, MEDIUM, HIGH, CRITICAL} by failure-rate bins; `error_type` = the dominant failure mode.
- All simulator probabilities are in `configs/config.yaml → harness`. The overall error level was calibrated once to BFCL-reported AST error ranges, never to model metrics.

## Fields
`tool_id, tool_name, server_name, tool_description, input_schema, parameter_count, required_parameter_count, optional_parameter_count, parameter_type_count, required_parameter_ratio, nested_schema_depth, enum_count, array_parameter_count, object_parameter_count, nullable_parameter_count, description_length, example_presence, example_count, execution_count, failure_count, failure_rate, agent_success_rate, error_occurred, error_type, error_severity, error_type_counts, strict_server, source, provenance, label_provenance, episode_origin, bfcl_category, lineage_id, parent_tool_id, augmentation_ops, timestamp`.

Engineered features (ambiguity, instruction density, constraint density, …) are computed by the feature pipeline (`data/processed/structured_features.csv`). Sentence embeddings are stored in `data/processed/embeddings.npy` and in the content-addressed cache.

## Data quality and missing values
- Exact duplicates are removed. Malformed schemas are **kept and flagged** (`schema_valid = 0`, `schema_issues`).
- Missing values appear only in optional metadata (`parent_tool_id` for seed tools, `bfcl_category` for MCP tools). There are no empty descriptions.
- Class imbalance is moderate (see `data_quality_report.json → imbalance_ratio`).

## Splits
Stratified grouped split by `lineage_id`: 70 / 15 / 15 with zero group overlap (`split_report.json`).

## Limitations
- The labels come from a simulated agent; validate against real LLM agents before drawing deployment conclusions.
- BFCL functions are research-benchmark APIs and not all are MCP servers. The schema contract is identical, but domain coverage differs.
- Augmented records are synthetic by construction.

## License and ethics
BFCL data: Apache-2.0. Please cite [20] in `docs/REFERENCES.md`. MCP reference tools: MIT-licensed upstream servers, transcribed for research. There is no personal data. Released for research and education.
