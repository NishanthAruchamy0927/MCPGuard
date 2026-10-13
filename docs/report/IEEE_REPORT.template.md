<!-- TEMPLATE: {{TOKENS}} are filled by `python -m src.reporting.report` from the latest pipeline run.
     Edit this template, not IEEE_REPORT.md. A LaTeX (IEEEtran) version is generated alongside. -->

# Predicting Which MCP Tool Descriptions Cause Agent Errors: An Explainable Machine Learning Framework for Detecting High-Risk Model Context Protocol Tool Descriptions

*19MAM56 Advanced Machine Learning Lab — CAT II Project · Department of Computing (AI & ML), Coimbatore Institute of Technology*

---

## Abstract

AI agents act on external systems through Model Context Protocol (MCP) tools, which they understand only through a name, a natural-language description and a JSON input schema. Poorly written definitions cause wrong tool selection, missing arguments and invalid values. We frame tool-definition risk as a supervised learning problem. We assemble {{N_RAW}} MCP-format tool records ({{N_BFCL}} real Berkeley Function-Calling Leaderboard functions with real user queries and gold calls, {{N_MCP}} MCP reference-server tools and {{N_AUG}} documented augmentations) and label each tool with a controlled agent benchmark (10 simulated episodes, BFCL-style call validation). A fused representation of TF-IDF/LSA text features, all-MiniLM-L6-v2 sentence embeddings, JSON-schema statistics and engineered risk scores feeds 14 models; XGBoost and LightGBM are tuned with Optuna under grouped cross-validation. The selected {{MODEL}} reaches test PR-AUC {{PR_AUC}} (95% CI {{PR_CI}}), ROC-AUC {{ROC_AUC}} and F1 {{F1}}, versus {{DUMMY_PR_AUC}} for a no-skill baseline and {{RULE_PR_AUC}} for a documentation-quality heuristic. Ablation shows that feature fusion improves PR-AUC by {{ABL_GAIN}} over the best single family. SHAP explanations and a rule layer turn predictions into actionable fixes; the system is served through FastAPI and Docker.

**Index Terms** — Model Context Protocol, tool learning, LLM agents, function calling, explainable AI, SHAP, XGBoost, Optuna.

## I. Introduction

**Problem.** An MCP tool exposes `{name, description, inputSchema}` to an LLM agent. The agent must (i) pick the right tool among those loaded and (ii) construct a schema-valid call with correct values. Both steps depend on the quality of the definition.

**Motivation and impact.** Tool catalogues are written by many authors and rarely reviewed. A pre-deployment risk score with explanations lets maintainers fix documentation before agents fail in production, which reduces wasted tokens, retries and incorrect actions.

**Objectives.** (1) Build a reproducible, provenance-labelled dataset of MCP tool definitions with measured failure labels; (2) engineer text, semantic and structural features with dimensionality reduction; (3) compare baselines with advanced, Optuna-tuned models; (4) explain predictions with SHAP and generate recommendations; (5) deploy a validated FastAPI service and dashboard in Docker.

## II. Literature Review

Tool-augmented language models were popularised by Toolformer [6] and ReAct [7]. Gorilla [2] and the Berkeley Function-Calling Leaderboard (BFCL) evaluate *agents* on function-calling accuracy using AST matching against gold calls; ToolLLM [5] scales tool use to 16k real APIs. Closer to our question, Hsieh et al. [3] show that tool *documentation* alone enables zero-shot tool use, and EasyTool [4] shows that concise, structured tool instructions improve agent performance. Both studies change documentation to improve agents but do not *predict* which documents are risky. The MCP specification [1] standardises tool definitions, and recent surveys analyse the MCP ecosystem and its risks [11].

**Research gap.** In the literature surveyed (Section VIII), evaluation focuses on agent capability, and documentation is treated as an intervention rather than as the object of prediction. We found no public benchmark that predicts tool-level failure risk from the definition alone. We therefore compare against (a) standard text-classification baselines (TF-IDF + LR/RF), (b) a documentation-quality heuristic that operationalises the hypothesis of [3], [4], and (c) a no-skill baseline.

## III. Methodology

**A. Pipeline.** Data sources → MCP conversion → validation → controlled benchmark → preprocessing → feature engineering → TF-IDF/embeddings → LSA/PCA (UMAP for visualisation) → fusion → model training → Optuna → selection → SHAP → error analysis → FastAPI → dashboard → Docker (Fig. 1, `docs/figures/methodology.svg`).

**B. Controlled agent benchmark (labels).** For each tool, a documentation-bounded agent runs 10 episodes. Each episode has three steps:

1. **Routing:** the agent picks a tool from a temperature-scaled softmax over sentence-embedding cosine similarity between the request and each candidate tool. Candidates are co-loaded sibling tools plus random tools from other servers.
2. **Argument construction:** the agent builds each argument. Literal values are copied from the request with *p* = 0.97. Otherwise, success depends on what the visible documentation exposes: enum, value mentioned, format hint, plain description, or nothing (*p* = 0.95, 0.90, 0.86, 0.76, 0.55).
3. **Validation:** the call is checked against the hidden schema and gold values (missing/extra arguments, type, enum, value).

A tool is labelled `error_occurred = 1` if its failure rate is ≥ 0.30. The simulator's overall error level was calibrated once to the range of AST error rates reported by BFCL for strong models (mean episode failure {{MEAN_FAIL}}). It was never tuned on model metrics.

**C. Features.** Text: TF-IDF word 1–2-grams and character 3–5-grams, reduced with TruncatedSVD (LSA, *k* = 64). Semantic: all-MiniLM-L6-v2 [10] embeddings of name + description + parameter list, reduced with PCA (*k* = 48). Schema: 26 structural statistics (depth, required ratio, type entropy, enums, constraints…). Lexical/naming: 25 statistics. Six risk scores in [0, 1], e.g. ambiguity = clip(5·vague/T + 0.5·[T < 6]) and missing-constraint = 1 − hinted/P (full formulas in `docs/FEATURES.md`). Fused dimension: {{N_FEATURES}}.

**D. Models.** Gradient boosting minimises a regularised logistic loss L = Σ ℓ(yᵢ, ŷᵢ) + Σₖ (γT + ½λ‖w‖²) [8]. Class imbalance is handled with cost-sensitive weights (scale_pos_weight = n₋/n₊) and a decision threshold tuned for maximum F1 on validation; SMOTE is not used on embeddings because interpolated vectors correspond to no real tool description.

**E. Hyper-parameter search.** Optuna [9] TPE with median pruning maximises mean average precision over 4-fold stratified *grouped* CV inside the training split ({{OPT_XGB_TRIALS}} XGBoost and {{OPT_LGB_TRIALS}} LightGBM trials; best CV PR-AUC {{OPT_XGB_CV}}).

**F. Split and selection.** A stratified grouped split by tool lineage (70/15/15; {{N_TRAIN}}/{{N_VAL}}/{{N_TEST}}) keeps near-duplicates and augmented variants in one partition (zero group overlap). Model selection order: validation PR-AUC → F1 → ROC-AUC.

## IV. Implementation & Experiments

Python 3.14, scikit-learn 1.9, XGBoost 3.4, LightGBM 4.7, sentence-transformers 6.1 (CPU PyTorch), Optuna 5, SHAP 0.53, MLflow 3.17 (SQLite) plus a JSONL run log, FastAPI 0.143. Seed 42 throughout. Hardware: commodity CPU laptop. The full pipeline runs with `python -m src.pipeline.run_all`. Dataset: {{N_RECORDS}} clean records ({{N_DUP}} exact duplicates removed, {{N_MALFORMED}} malformed schemas kept and flagged), {{N_SERVERS}} servers, {{N_LINEAGES}} lineages, positive rate {{POS_RATE}}. Experiments: the matrix E1–E6 (TF-IDF+LR, TF-IDF+RF, Emb+RF, Emb+XGB, Fused+XGB, Optimised Fused+XGB), benchmark models B0–B7 and a six-way feature-family ablation. Global SHAP drivers: {{TOP_FEATURES}}. Importance by family: {{FAMILY_SHARES}}.

## V. Results & Comparative Analysis

**Table I — Test-split benchmark (\* = selected).**

{{TABLE_BENCHMARK_MD}}

The selected {{EXPERIMENT}} achieves accuracy {{ACC}}, precision {{PREC}}, recall {{REC}}, F1 {{F1}} (95% CI {{F1_CI}}), ROC-AUC {{ROC_AUC}} ({{ROC_CI}}), PR-AUC {{PR_AUC}} and MCC {{MCC}} at threshold {{THRESHOLD}}. Confusion matrix on test: TP {{TP}}, TN {{TN}}, FP {{FP}}, FN {{FN}}.

**Table II — Paired bootstrap, selected minus baseline (test PR-AUC).**

{{TABLE_PAIRED_MD}}

**Table III — Ablation (XGBoost, 5-fold grouped CV).**

{{TABLE_ABLATION_MD}}

All fused features reach CV PR-AUC {{ABL_ALL}}, against {{ABL_BEST_SINGLE}} for the best single family ({{ABL_BEST_SINGLE_NAME}}).

**Error analysis.** {{EA_HEADLINE}} Many missed failures depend on information outside a single tool definition: how users phrase values, sibling-tool routing and server strictness. This is a structural limit on any description-only predictor. False positives concentrate on tools with weak documentation signals whose benchmark requests supplied literal values. Figures: confusion matrix, ROC/PR curves, SHAP importance and beeswarm, UMAP/PCA projections, calibration and ablation (`artifacts/figures/`).

## VI. Conclusion & Future Scope

Tool-definition risk is learnable. Fused text, semantic and schema features clearly outperform both a no-skill baseline and a documentation-quality heuristic, and SHAP makes the drivers actionable. Several fused models are statistically tied, so the gain comes mainly from the representation rather than the choice of learner. **Deployment:** a single non-root container serves the validated API and dashboard; descriptions are stored only as hashes. **Limitations:** labels come from a calibrated *simulated* agent. **Future work:** relabel with real LLM agents (pass@k across models), add LLM-based description rewriting with a closed-loop risk check, and train on production telemetry.

## References

See `docs/REFERENCES.md` (IEEE style; verify every entry against the publisher before submission).
