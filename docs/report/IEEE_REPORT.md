<!-- TEMPLATE: {{TOKENS}} are filled by `python -m src.reporting.report` from the latest pipeline run.
     Edit this template, not IEEE_REPORT.md. A LaTeX (IEEEtran) version is generated alongside. -->

# Predicting Which MCP Tool Descriptions Cause Agent Errors: An Explainable Machine Learning Framework for Detecting High-Risk Model Context Protocol Tool Descriptions

*19MAM56 Advanced Machine Learning Lab — CAT II Project · Department of Computing (AI & ML), Coimbatore Institute of Technology*

---

## Abstract

AI agents act on external systems through Model Context Protocol (MCP) tools, which they understand only through a name, a natural-language description and a JSON input schema. Poorly written definitions cause wrong tool selection, missing arguments and invalid values. We frame tool-definition risk as a supervised learning problem. We assemble 2181 MCP-format tool records (1579 real Berkeley Function-Calling Leaderboard functions with real user queries and gold calls, 99 MCP reference-server tools and 503 documented augmentations) and label each tool with a controlled agent benchmark (10 simulated episodes, BFCL-style call validation). A fused representation of TF-IDF/LSA text features, all-MiniLM-L6-v2 sentence embeddings, JSON-schema statistics and engineered risk scores feeds 14 models; XGBoost and LightGBM are tuned with Optuna under grouped cross-validation. The selected XGBoost (Optuna-tuned) reaches test PR-AUC 0.709 (95% CI [0.624, 0.793]), ROC-AUC 0.779 and F1 0.684, versus 0.406 for a no-skill baseline and 0.416 for a documentation-quality heuristic. Ablation shows that feature fusion improves PR-AUC by +0.033 over the best single family. SHAP explanations and a rule layer turn predictions into actionable fixes; the system is served through FastAPI and Docker.

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

A tool is labelled `error_occurred = 1` if its failure rate is ≥ 0.30. The simulator's overall error level was calibrated once to the range of AST error rates reported by BFCL for strong models (mean episode failure 27.4%). It was never tuned on model metrics.

**C. Features.** Text: TF-IDF word 1–2-grams and character 3–5-grams, reduced with TruncatedSVD (LSA, *k* = 64). Semantic: all-MiniLM-L6-v2 [10] embeddings of name + description + parameter list, reduced with PCA (*k* = 48). Schema: 26 structural statistics (depth, required ratio, type entropy, enums, constraints…). Lexical/naming: 25 statistics. Six risk scores in [0, 1], e.g. ambiguity = clip(5·vague/T + 0.5·[T < 6]) and missing-constraint = 1 − hinted/P (full formulas in `docs/FEATURES.md`). Fused dimension: 169.

**D. Models.** Gradient boosting minimises a regularised logistic loss L = Σ ℓ(yᵢ, ŷᵢ) + Σₖ (γT + ½λ‖w‖²) [8]. Class imbalance is handled with cost-sensitive weights (scale_pos_weight = n₋/n₊) and a decision threshold tuned for maximum F1 on validation; SMOTE is not used on embeddings because interpolated vectors correspond to no real tool description.

**E. Hyper-parameter search.** Optuna [9] TPE with median pruning maximises mean average precision over 4-fold stratified *grouped* CV inside the training split (40 XGBoost and 30 LightGBM trials; best CV PR-AUC 0.690).

**F. Split and selection.** A stratified grouped split by tool lineage (70/15/15; 1537/307/308) keeps near-duplicates and augmented variants in one partition (zero group overlap). Model selection order: validation PR-AUC → F1 → ROC-AUC.

## IV. Implementation & Experiments

Python 3.14, scikit-learn 1.9, XGBoost 3.4, LightGBM 4.7, sentence-transformers 6.1 (CPU PyTorch), Optuna 5, SHAP 0.53, MLflow 3.17 (SQLite) plus a JSONL run log, FastAPI 0.143. Seed 42 throughout. Hardware: commodity CPU laptop. The full pipeline runs with `python -m src.pipeline.run_all`. Dataset: 2152 clean records (29 exact duplicates removed, 13 malformed schemas kept and flagged), 670 servers, 1403 lineages, positive rate 40.7%. Experiments: the matrix E1–E6 (TF-IDF+LR, TF-IDF+RF, Emb+RF, Emb+XGB, Fused+XGB, Optimised Fused+XGB), benchmark models B0–B7 and a six-way feature-family ablation. Global SHAP drivers: Schema size, Text theme #23 (distance, untyped, search), Schema complexity score, Text theme #45 (function, coordinates, longitude), Average word length. Importance by family: Text (LSA) 44%, Semantic embedding 29%, Schema structure 14%, Lexical / naming 8%, Engineered risk scores 4%.

## V. Results & Comparative Analysis

**Table I — Test-split benchmark (\* = selected).**

| ID | Model / features | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| B6 | Combined Features + MLP Neural Network | 0.604 | 0.507 | 0.928 | 0.655 | 0.800 | 0.745 |
| B3 | Combined Features + Random Forest | 0.744 | 0.672 | 0.720 | 0.695 | 0.783 | 0.711 |
| E6 * | Optimized Combined Features + XGBoost | 0.724 | 0.639 | 0.736 | 0.684 | 0.779 | 0.709 |
| B2 | Combined Features + Logistic Regression | 0.718 | 0.617 | 0.800 | 0.697 | 0.803 | 0.706 |
| E2 | TF-IDF + Random Forest | 0.692 | 0.584 | 0.832 | 0.686 | 0.779 | 0.698 |
| E5 | Combined Features + XGBoost | 0.643 | 0.538 | 0.856 | 0.660 | 0.762 | 0.693 |
| B4 | Combined Features + LightGBM | 0.536 | 0.464 | 0.928 | 0.619 | 0.750 | 0.692 |
| B7 | Optimized Combined Features + LightGBM | 0.594 | 0.500 | 0.880 | 0.638 | 0.763 | 0.690 |
| B5 | Combined Features + Hist. Gradient Boosting | 0.506 | 0.447 | 0.912 | 0.600 | 0.735 | 0.676 |
| E1 | TF-IDF + Logistic Regression | 0.601 | 0.505 | 0.824 | 0.626 | 0.735 | 0.661 |
| E3 | Semantic Embedding + Random Forest | 0.640 | 0.544 | 0.696 | 0.611 | 0.695 | 0.616 |
| E4 | Semantic Embedding + XGBoost | 0.617 | 0.520 | 0.728 | 0.607 | 0.673 | 0.599 |
| B1 | Doc-quality heuristic (rule) | 0.406 | 0.405 | 0.984 | 0.573 | 0.492 | 0.416 |
| B0 | Majority / prior baseline | 0.406 | 0.406 | 1.000 | 0.577 | 0.500 | 0.406 |

The selected Optimized Combined Features + XGBoost achieves accuracy 0.724, precision 0.639, recall 0.736, F1 0.684 (95% CI [0.615, 0.743]), ROC-AUC 0.779 ([0.726, 0.828]), PR-AUC 0.709 and MCC 0.445 at threshold 0.389. Confusion matrix on test: TP 92, TN 131, FP 52, FN 33.

**Table II — Paired bootstrap, selected minus baseline (test PR-AUC).**

| vs | Baseline | ΔPR-AUC | 95% CI | p |
|---|---|---|---|---|
| E1 | TF-IDF + Logistic Regression | +0.048 | [-0.014, +0.111] | 0.074 |
| E2 | TF-IDF + Random Forest | +0.011 | [-0.035, +0.053] | 0.315 |
| E3 | Semantic Embedding + Random Forest | +0.092 | [+0.045, +0.142] | 0.000 |
| E4 | Semantic Embedding + XGBoost | +0.110 | [+0.053, +0.166] | 0.000 |
| E5 | Combined Features + XGBoost | +0.017 | [-0.010, +0.045] | 0.103 |
| B1 | Doc-quality heuristic (rule) | +0.289 | [+0.188, +0.377] | 0.000 |
| B2 | Combined Features + Logistic Regression | +0.004 | [-0.058, +0.063] | 0.428 |
| B3 | Combined Features + Random Forest | -0.002 | [-0.027, +0.024] | 0.563 |
| B4 | Combined Features + LightGBM | +0.018 | [-0.008, +0.047] | 0.086 |
| B5 | Combined Features + Hist. Gradient Boosting | +0.033 | [+0.007, +0.061] | 0.003 |
| B6 | Combined Features + MLP Neural Network | -0.034 | [-0.084, +0.019] | 0.913 |
| B7 | Optimized Combined Features + LightGBM | +0.021 | [-0.002, +0.045] | 0.044 |

**Table III — Ablation (XGBoost, 5-fold grouped CV).**

| Set | Families | #feat | CV PR-AUC | CV ROC-AUC | Test PR-AUC |
|---|---|---|---|---|---|
| A | Text (LSA) + Lexical / naming | 89 | 0.656 ± 0.051 | 0.729 | 0.656 |
| B | Schema structure | 26 | 0.643 ± 0.034 | 0.697 | 0.640 |
| C | Semantic embedding | 48 | 0.619 ± 0.056 | 0.710 | 0.599 |
| D | Text (LSA) + Lexical / naming + Schema structure | 115 | 0.684 ± 0.041 | 0.752 | 0.694 |
| E | Semantic embedding + Schema structure | 74 | 0.664 ± 0.033 | 0.747 | 0.671 |
| F | Text (LSA) + Lexical / naming + Schema structure + Engineered risk scores + Semantic embedding | 169 | 0.690 ± 0.042 | 0.756 | 0.693 |

All fused features reach CV PR-AUC 0.690, against 0.656 for the best single family (Text (LSA) + Lexical / naming).

**Error analysis.** Most common source of model mistakes: missed failures whose dominant benchmark failure mode is invalid arg value (52% of false negatives). Many missed failures depend on information outside a single tool definition: how users phrase values, sibling-tool routing and server strictness. This is a structural limit on any description-only predictor. False positives concentrate on tools with weak documentation signals whose benchmark requests supplied literal values. Figures: confusion matrix, ROC/PR curves, SHAP importance and beeswarm, UMAP/PCA projections, calibration and ablation (`artifacts/figures/`).

## VI. Conclusion & Future Scope

Tool-definition risk is learnable. Fused text, semantic and schema features clearly outperform both a no-skill baseline and a documentation-quality heuristic, and SHAP makes the drivers actionable. Several fused models are statistically tied, so the gain comes mainly from the representation rather than the choice of learner. **Deployment:** a single non-root container serves the validated API and dashboard; descriptions are stored only as hashes. **Limitations:** labels come from a calibrated *simulated* agent. **Future work:** relabel with real LLM agents (pass@k across models), add LLM-based description rewriting with a closed-loop risk check, and train on production telemetry.

## References

See `docs/REFERENCES.md` (IEEE style; verify every entry against the publisher before submission).
