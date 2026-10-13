"""SHAP explanations for the fused MCPGuard pipeline.

Tree models use ``shap.TreeExplainer`` (exact, log-odds space); other models fall
back to a permutation explainer over a background sample.  Feature names are
humanised; LSA text components are labelled with their top TF-IDF terms so a
"text" contribution is still interpretable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build import FAMILY_LABELS

HUMAN_NAMES = {
    "desc_char_len": "Description length (chars)", "desc_token_count": "Description length (tokens)",
    "desc_sentence_count": "Sentence count", "desc_avg_token_len": "Average word length", "desc_vocab_size": "Vocabulary size",
    "desc_lexical_diversity": "Lexical diversity", "desc_stopword_ratio": "Stop-word ratio", "desc_punct_density": "Punctuation density",
    "desc_special_char_density": "Special-character density", "desc_uppercase_ratio": "Upper-case ratio", "desc_digit_count": "Digits in description",
    "vague_term_count": "Vague terms", "instruction_term_count": "Instruction terms", "constraint_term_count": "Constraint terms",
    "example_mentions": "Examples in description", "format_hint_mentions": "Format hints in description",
    "instruction_density": "Instruction density", "constraint_density": "Constraint density", "example_presence": "Examples present",
    "desc_mentions_param_ratio": "Parameters mentioned in description", "name_token_count": "Tool-name word count",
    "name_char_len": "Tool-name length", "name_has_verb": "Tool name starts with a verb", "name_is_generic": "Generic tool name",
    "name_has_namespace": "Namespaced tool name",
    "parameter_count": "Parameter count", "nested_parameter_count": "Nested parameters", "total_parameter_count": "Total parameters (incl. nested)",
    "required_parameter_count": "Required parameters", "optional_parameter_count": "Optional parameters",
    "required_parameter_ratio": "Required-parameter ratio", "parameter_type_count": "Distinct parameter types",
    "parameter_type_entropy": "Parameter-type entropy", "untyped_parameter_count": "Untyped parameters", "nested_schema_depth": "Schema nesting depth",
    "object_parameter_count": "Object parameters", "array_parameter_count": "Array parameters", "enum_count": "Enum parameters",
    "enum_value_total": "Enum values", "nullable_parameter_count": "Nullable parameters", "constrained_parameter_count": "Constrained parameters",
    "constrained_parameter_ratio": "Constrained-parameter ratio", "default_value_count": "Parameters with defaults",
    "schema_example_count": "Schema examples", "param_described_ratio": "Documented-parameter ratio",
    "param_desc_mean_len": "Parameter description length", "schema_issue_count": "Schema issues", "schema_valid": "Schema is valid JSON",
    "schema_char_len": "Schema size", "required_documented_ratio": "Documented required parameters",
    "hinted_parameter_ratio": "Parameters with format hints/constraints",
    "ambiguity_score": "Ambiguity score", "missing_constraint_score": "Missing-constraint score",
    "parameter_complexity_score": "Parameter complexity score", "schema_complexity_score": "Schema complexity score",
    "description_quality_score": "Description quality score", "instruction_clarity_score": "Instruction clarity score",
}


def lsa_topics(builder, top_n: int = 4) -> dict[int, list[str]]:
    """Top word n-grams for each LSA component (word part of the TF-IDF union)."""
    if getattr(builder, "svd_", None) is None:
        return {}
    names = builder.tfidf_.get_feature_names_out()
    word_mask = np.array([n.startswith("word__") for n in names])
    comps = builder.svd_.components_
    out = {}
    for i, comp in enumerate(comps):
        c = np.where(word_mask, np.abs(comp), 0)
        idx = np.argsort(-c)[:top_n]
        out[i] = [names[j].split("__", 1)[1] for j in idx]
    return out


def humanize(feature: str, topics: dict | None = None) -> str:
    fam, _, raw = feature.partition(":")
    if fam == "text" and raw.startswith("lsa_"):
        k = int(raw.split("_")[1])
        terms = (topics or {}).get(k)
        return f"Text theme #{k}" + (f" ({', '.join(terms[:3])})" if terms else "")
    if fam == "emb":
        return f"Semantic dimension #{int(raw.split('_')[1])}"
    if fam == "tfidf":
        return f"Term '{raw.split('__', 1)[-1]}'"
    return HUMAN_NAMES.get(raw, raw.replace("_", " ").capitalize())


class PipelineExplainer:
    def __init__(self, pipeline, background_df: pd.DataFrame | None = None, background_size: int = 200, seed: int = 42):
        import shap
        self.pipeline = pipeline
        self.builder = pipeline.named_steps["features"]
        self.model = pipeline.named_steps["model"]
        self.names = list(self.builder.feature_names_)
        self.families = list(self.builder.feature_families_)
        self.topics = lsa_topics(self.builder)
        self.display = [humanize(n, self.topics) for n in self.names]
        kind = type(self.model).__name__
        if kind in ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier", "HistGradientBoostingClassifier"):
            self.kind = "tree"
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.kind = "permutation"
            if background_df is None:
                raise ValueError("background data required for non-tree models")
            bg = self.builder.transform(background_df.sample(min(background_size, len(background_df)), random_state=seed))
            self.explainer = shap.Explainer(lambda X: self.model.predict_proba(X)[:, 1], bg, algorithm="permutation")

    def shap_values(self, X_features: np.ndarray) -> tuple[np.ndarray, float]:
        if self.kind == "tree":
            sv = self.explainer.shap_values(X_features)
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            base = self.explainer.expected_value
            base = float(np.ravel(base)[-1])
            return sv, base
        exp = self.explainer(X_features)
        return np.asarray(exp.values), float(np.ravel(exp.base_values)[0])

    def explain_frame(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
        X = self.builder.transform(df)
        sv, base = self.shap_values(X)
        return X, sv, base

    def top_contributions(self, x_row: np.ndarray, sv_row: np.ndarray, k: int = 8) -> list[dict]:
        order = np.argsort(-np.abs(sv_row))[:k]
        return [{"feature": self.names[i], "display_name": self.display[i], "family": self.families[i],
                 "family_label": FAMILY_LABELS[self.families[i]], "value": float(x_row[i]), "shap": float(sv_row[i]),
                 "direction": "increases risk" if sv_row[i] > 0 else "decreases risk"} for i in order]

    def family_contributions(self, sv_row: np.ndarray) -> dict[str, float]:
        out: dict[str, float] = {}
        for fam, v in zip(self.families, sv_row):
            out[fam] = out.get(fam, 0.0) + float(v)
        return out
