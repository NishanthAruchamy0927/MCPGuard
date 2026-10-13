# Dataset card

The full dataset card lives in [`docs/DATASET_CARD.md`](../docs/DATASET_CARD.md). Generated statistics are in `data/metadata/`.

| Folder | Contents |
|---|---|
| `raw/` | BFCL v3 function files and gold answers (re-downloadable: `python -m src.data.download`; git-ignored) |
| `external/` | `mcp_reference_tools.json`, the curated MCP reference-server tools |
| `processed/` | `mcp_tools_dataset.{jsonl,csv}` (labelled), `mcp_tools_clean.jsonl`, `splits.csv`, `structured_features.csv`, `embeddings.npy`, embedding cache |
| `metadata/` | `download_manifest.json` (URLs + SHA-256), `dataset_summary.json`, `data_quality_report.json`, `split_report.json` |
