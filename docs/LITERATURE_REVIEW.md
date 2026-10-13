# Literature review (working notes)

Add new references to `docs/REFERENCES.md` and summarise them here. Numbers refer to that file.

| # | Work | What it studies | Relevance to MCPGuard | Gap w.r.t. our question |
|---|---|---|---|---|
| [6] | Toolformer | LMs learn when and how to call APIs (self-supervised) | Establishes tool use as a core LM capability | Studies the model, not the tool definitions |
| [7] | ReAct | Interleaving reasoning and tool actions | Agent loop that consumes tool descriptions | No analysis of description quality |
| [2], [20] | Gorilla / BFCL | Function-calling accuracy, AST matching against gold calls | **Our data source and evaluation protocol** (functions, real queries, gold calls; AST-style validation in the harness) | Ranks *models*; does not predict which *functions* cause failures |
| [5] | ToolLLM | Scaling tool use to 16k real APIs (ToolBench) | Real-world API documentation is noisy | Improves the agent, not the documentation |
| [3] | Hsieh et al. | Documentation alone enables zero-shot tool use | Motivates documentation quality as a causal factor | Intervention study; no predictive model for risky docs |
| [4] | EasyTool | Concise, structured tool instructions improve agents | Motivates our recommendation rules and the doc-quality baseline (B1) | Rewrites docs; no per-tool risk estimate |
| [1], [11] | MCP spec, MCP landscape | Standardised tool definitions; ecosystem and security analysis | Defines `{name, description, inputSchema}` | No reliability prediction from definitions |
| [12] | SHAP | Additive feature attributions | Global and local explanations | — |
| [8], [13], [9] | XGBoost, LightGBM, Optuna | Gradient boosting, Bayesian HPO | Core models and tuning | — |
| [10], [15], [14] | Sentence-BERT, MiniLM, UMAP | Sentence embeddings, projection | Semantic features and visualisation | — |
| [17], [18], [19] | PR vs ROC, bootstrap | Evaluation under imbalance, CIs | Metric choice (PR-AUC primary), bootstrap tests | — |

## Synthesised research gap

In the works above, *agents* are evaluated or improved, and tool documentation is treated as an *intervention* ([3], [4]). Within this survey we did not find work that treats the tool definition itself as the unit of prediction: estimating, before deployment, whether a given MCP tool description will cause agent errors, and explaining why. MCPGuard addresses that question. The claim is scoped to the literature reviewed here; extend this table before making stronger statements.

## Baseline methodology used for comparison

No published numeric benchmark exists for this exact task, so the comparison uses:

1. **No-skill baseline (B0):** PR-AUC equals the positive rate.
2. **Documentation-quality heuristic (B1):** risk = 1 − description_quality_score. This operationalises the hypothesis of [3], [4] that better-documented tools fail less.
3. **Standard text-classification baselines (E1, E2):** TF-IDF with Logistic Regression and Random Forest.

## To read / add

- Newer MCP-specific agent benchmarks (2025) evaluating agents on real MCP servers. Add them with verified bibliographic data.
- Work on API documentation smells in software engineering.
