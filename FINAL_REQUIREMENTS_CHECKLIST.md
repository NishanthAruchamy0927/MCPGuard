# Final Requirements Checklist — 19MAM56 CAT II

Status key: **COMPLETE** = verified by running it in this environment · **PARTIAL** = implemented, part needs user action · **PENDING USER ACTION** = needs your accounts or credentials.

| Requirement | Status | Evidence |
|---|---|---|
| Complex dataset | COMPLETE | 2,181 records (2,152 after cleaning): 1,579 real BFCL v3 functions with real queries and gold calls, 99 MCP reference-server tools, 503 documented augmentations; 670 servers; text + JSON schema + numeric + categorical; binary and 5-class targets. `data/metadata/dataset_summary.json`, `docs/DATASET_CARD.md` |
| Advanced preprocessing | COMPLETE | Unicode/whitespace normalisation, identifier splitting, JSON-schema parsing/repair, type-alias normalisation, malformed schemas kept and flagged, de-duplication, lineage grouping, median imputation. `src/preprocessing/`, `docs/PREPROCESSING.md`, `data/metadata/data_quality_report.json`; tests in `tests/test_preprocessing.py` |
| Feature engineering | COMPLETE | TF-IDF word/char, MiniLM sentence embeddings, 26 schema + 25 lexical features, 6 risk scores with formulas (`docs/FEATURES.md`), 169-d fused vector |
| Dimensionality reduction | COMPLETE | TruncatedSVD/LSA (64), PCA on embeddings (48), UMAP + 2-D PCA projections (`artifacts/figures/06_umap_projection.png`, `07_pca_projection.png`, interactive on the Explainability page) |
| 5+ visualizations | COMPLETE | 14 figures generated from real runs in `artifacts/figures/` (confusion matrix, ROC, PR, SHAP bar, SHAP beeswarm, UMAP, PCA, model comparison, error types, probability distribution, risk distribution, ablation, Optuna history, calibration) plus interactive Plotly charts in the dashboard |
| Advanced ML | COMPLETE | XGBoost, LightGBM, HistGB, MLP, RF and LR on fused features; experiment matrix E1–E6, benchmark B0–B7 (`artifacts/metrics/benchmark.json`) |
| Hyperparameter tuning | COMPLETE | Optuna TPE + median pruning, 40 XGBoost / 30 LightGBM trials, grouped 4-fold CV (`artifacts/experiments/optuna_*_trials.csv`, `models/best_params_*.json`) |
| Experiment tracking | COMPLETE | MLflow (SQLite `artifacts/mlflow.db`) + JSONL log (`artifacts/experiments/experiments.jsonl`, 16 runs), shown on the Experiments page |
| Benchmark comparison | COMPLETE | 14 models auto-tabulated; no-skill and documentation-quality heuristic baselines; paired bootstrap significance tests (`artifacts/reports/results_tables.md`, README Results). No published numeric benchmark exists for this exact task; this is documented in `docs/LITERATURE_REVIEW.md` |
| Error analysis | COMPLETE | TP/TN/FP/FN examples with SHAP and reasons, slice analysis, automated summary (`artifacts/metrics/error_analysis.json`, Error Analysis page) |
| FastAPI inference | COMPLETE | All endpoints exercised by `tests/test_api.py` (validation, 422s, batch CSV/JSON, reload). `/docs` and `/redoc` live; real predictions verified from the UI (`docs/screenshots/analyze.png`) |
| Docker | COMPLETE | `docker build -t mcpguard .` succeeded (image 2.98 GB, CPU-only torch, MiniLM baked in). Container verified: Docker health status `healthy`, runs as `appuser`, `GET /health` → `model_loaded: true`, `POST /predict` returned a CRITICAL prediction with SHAP factors, dashboard `GET /` → 200. Files: Dockerfile, docker-compose.yml, .dockerignore |
| Cloud deployment | PENDING USER ACTION | Ready as a free **Gradio-SDK** Hugging Face Space (Docker SDK is paid-only on Hugging Face): `space_app.py` serves the dashboard, REST API and a Gradio analyzer at `/gradio`; verified locally and by `tests/test_space_app.py`. Push the `hf-space` branch (steps in `docs/DEPLOYMENT.md`) and record the live URL here: `____` |
| GitHub repository | COMPLETE | Public: https://github.com/NishanthAruchamy0927/MCPGuard (code, configs, trained models and artefacts; large binaries via Git LFS) |
| Report | PARTIAL | IEEE report auto-filled from results: `docs/report/IEEE_REPORT.md` and `docs/report/ieee_report.tex` (IEEEtran). **PENDING USER ACTION:** add author names, compile to PDF (Overleaf) and check the 10-page limit |
| IEEE references | PARTIAL | 20 IEEE-style references in `docs/REFERENCES.md`. **PENDING USER ACTION:** verify each entry against the publisher before submission |
| Tests | COMPLETE | `pytest -q`: 53 passed (preprocessing, features, harness/validator, models, split leakage, prediction contract, SHAP additivity, API) |

## Remaining manual actions
1. Push the `hf-space` branch to a Gradio Space (see `docs/DEPLOYMENT.md`) and paste the public URL above.
2. ~~Push to a public GitHub repository~~ — done: https://github.com/NishanthAruchamy0927/MCPGuard
3. Fill in author names, compile `docs/report/ieee_report.tex` to PDF, and verify the references.
