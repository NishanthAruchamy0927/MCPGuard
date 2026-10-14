# MCPGuard — results tables (auto-generated)

Selected production model: **Optimized Combined Features + XGBoost** (XGBoost (Optuna-tuned)), threshold 0.389.

## Benchmark (held-out test split)
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

## Experiment matrix
| ID | Experiment | Val PR-AUC | Test F1 | Test ROC-AUC | Test PR-AUC | Train s |
|---|---|---|---|---|---|---|
| E1 | TF-IDF + Logistic Regression | 0.616 | 0.626 | 0.735 | 0.661 | 0.09 |
| E2 | TF-IDF + Random Forest | 0.651 | 0.686 | 0.779 | 0.698 | 0.81 |
| E3 | Semantic Embedding + Random Forest | 0.637 | 0.611 | 0.695 | 0.616 | 0.60 |
| E4 | Semantic Embedding + XGBoost | 0.608 | 0.607 | 0.673 | 0.599 | 0.33 |
| E5 | Combined Features + XGBoost | 0.669 | 0.660 | 0.762 | 0.693 | 1.04 |
| E6 | Optimized Combined Features + XGBoost | 0.706 | 0.684 | 0.779 | 0.709 | 2.63 |

## Ablation (XGBoost, 5-fold grouped CV on train+val)
| Set | Families | #feat | CV PR-AUC | CV ROC-AUC | Test PR-AUC |
|---|---|---|---|---|---|
| A | Text (LSA) + Lexical / naming | 89 | 0.656 ± 0.051 | 0.729 | 0.656 |
| B | Schema structure | 26 | 0.643 ± 0.034 | 0.697 | 0.640 |
| C | Semantic embedding | 48 | 0.619 ± 0.056 | 0.710 | 0.599 |
| D | Text (LSA) + Lexical / naming + Schema structure | 115 | 0.684 ± 0.041 | 0.752 | 0.694 |
| E | Semantic embedding + Schema structure | 74 | 0.664 ± 0.033 | 0.747 | 0.671 |
| F | Text (LSA) + Lexical / naming + Schema structure + Engineered risk scores + Semantic embedding | 169 | 0.690 ± 0.042 | 0.756 | 0.693 |

## Paired bootstrap: selected model minus baseline (test PR-AUC)
| vs | Baseline | ΔPR-AUC | 95% CI | p (one-sided) |
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

## Error analysis
Most common source of model mistakes: missed failures whose dominant benchmark failure mode is invalid arg value (52% of false negatives).

- Test outcomes — TP 92, TN 131, FP 52, FN 33.
- Hardest failure-mode slice: MISSING_REQUIRED_ARG (error rate 35%, n=26).
- 79% of false negatives are failure modes that depend on context outside the tool definition (argument values in the user request, sibling-tool routing, server strictness).
- False positives vs. true negatives: mean description length 14.75 vs 12.794 tokens; mean parameter count 3.481 vs 2.534.