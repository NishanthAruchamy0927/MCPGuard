"""Multimodal feature fusion (scikit-learn compatible).

Families
--------
``tfidf``    sparse TF-IDF word 1–2-grams + char 3–5-grams           (baselines)
``text``     LSA: TF-IDF -> TruncatedSVD(k)                          (fusion)
``lexical``  interpretable description / naming statistics
``schema``   structural JSON-schema statistics
``risk``     engineered risk scores (src.features.risk)
``emb``      sentence embedding (all-MiniLM-L6-v2) -> PCA(k)

``python -m src.features.build`` materialises the full feature matrix for the
whole dataset (for EDA, PCA/UMAP and the dataset explorer).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.features.embeddings import EmbeddingEncoder
from src.features.risk import structured_features, tool_document
from src.utils.config import load_config

RISK_FEATURES = ["ambiguity_score", "missing_constraint_score", "parameter_complexity_score", "schema_complexity_score",
                 "description_quality_score", "instruction_clarity_score"]
SCHEMA_FEATURES = [
    "parameter_count", "nested_parameter_count", "total_parameter_count", "required_parameter_count", "optional_parameter_count",
    "required_parameter_ratio", "parameter_type_count", "parameter_type_entropy", "untyped_parameter_count", "nested_schema_depth",
    "object_parameter_count", "array_parameter_count", "enum_count", "enum_value_total", "nullable_parameter_count",
    "constrained_parameter_count", "constrained_parameter_ratio", "default_value_count", "schema_example_count",
    "param_described_ratio", "param_desc_mean_len", "schema_issue_count", "schema_valid", "schema_char_len",
    "required_documented_ratio", "hinted_parameter_ratio",
]
LEXICAL_FEATURES = [
    "desc_char_len", "desc_token_count", "desc_sentence_count", "desc_avg_token_len", "desc_vocab_size", "desc_lexical_diversity",
    "desc_stopword_ratio", "desc_punct_density", "desc_special_char_density", "desc_uppercase_ratio", "desc_digit_count",
    "vague_term_count", "instruction_term_count", "constraint_term_count", "example_mentions", "format_hint_mentions",
    "instruction_density", "constraint_density", "example_presence", "desc_mentions_param_ratio",
    "name_token_count", "name_char_len", "name_has_verb", "name_is_generic", "name_has_namespace",
]
FAMILIES = ("tfidf", "text", "lexical", "schema", "risk", "emb")
ABLATION_SETS = {
    "A_text_only": ("text", "lexical"),
    "B_schema_only": ("schema",),
    "C_embeddings_only": ("emb",),
    "D_text_schema": ("text", "lexical", "schema"),
    "E_embeddings_schema": ("emb", "schema"),
    "F_all_features": ("text", "lexical", "schema", "risk", "emb"),
}
FAMILY_LABELS = {"tfidf": "TF-IDF n-grams", "text": "Text (LSA)", "lexical": "Lexical / naming", "schema": "Schema structure",
                 "risk": "Engineered risk scores", "emb": "Semantic embedding"}


def to_frame(tools) -> pd.DataFrame:
    """Accept a DataFrame or list of dicts with tool_name / tool_description / input_schema."""
    df = tools if isinstance(tools, pd.DataFrame) else pd.DataFrame(list(tools))
    if "tool_description" not in df.columns and "description" in df.columns:
        df = df.rename(columns={"description": "tool_description"})
    for col in ("tool_name", "tool_description", "input_schema"):
        if col not in df.columns:
            df[col] = ""
    return df


def structured_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = [structured_features(n, d, s) for n, d, s in zip(df["tool_name"], df["tool_description"], df["input_schema"])]
    return pd.DataFrame(rows, index=df.index)


def documents(df: pd.DataFrame) -> list[str]:
    return [tool_document(n, d, s) for n, d, s in zip(df["tool_name"], df["tool_description"], df["input_schema"])]


class MCPFeatureBuilder(BaseEstimator, TransformerMixin):
    """Fit on training tools only; transform any tools (DataFrame / list of dicts)."""

    def __init__(self, families: tuple = ("text", "lexical", "schema", "risk", "emb"), text_svd_components: int | None = None,
                 embedding_pca_components: int | None = None, random_state: int = 42):
        self.families = families
        self.text_svd_components = text_svd_components
        self.embedding_pca_components = embedding_pca_components
        self.random_state = random_state

    # ----------------------------------------------------------- internals
    def _tfidf(self) -> FeatureUnion:
        f = load_config()["features"]
        w, c = f["tfidf_word"], f["tfidf_char"]
        return FeatureUnion([
            ("word", TfidfVectorizer(max_features=w["max_features"], ngram_range=tuple(w["ngram_range"]), min_df=w["min_df"],
                                     sublinear_tf=w["sublinear_tf"], stop_words="english")),
            ("char", TfidfVectorizer(analyzer="char_wb", max_features=c["max_features"], ngram_range=tuple(c["ngram_range"]),
                                     min_df=c["min_df"], sublinear_tf=c["sublinear_tf"])),
        ])

    def _encoder(self) -> EmbeddingEncoder:
        if getattr(self, "_enc", None) is None:
            self._enc = EmbeddingEncoder()
        return self._enc

    def __getstate__(self):  # never pickle the transformer network
        state = self.__dict__.copy()
        state["_enc"] = None
        return state

    def _needs(self, *fams) -> bool:
        return any(f in self.families for f in fams)

    # ---------------------------------------------------------------- fit
    def fit(self, X, y=None):
        cfg = load_config()["features"]
        df = to_frame(X)
        docs = documents(df)
        self.feature_names_: list[str] = []
        self.feature_families_: list[str] = []
        if self._needs("tfidf", "text"):
            self.tfidf_ = self._tfidf().fit(docs)
        if "tfidf" in self.families:
            names = [f"tfidf:{n}" for n in self.tfidf_.get_feature_names_out()]
            self.feature_names_ += names
            self.feature_families_ += ["tfidf"] * len(names)
        if "text" in self.families:
            k = self.text_svd_components or cfg["text_svd_components"]
            tf = self.tfidf_.transform(docs)
            self.svd_ = TruncatedSVD(n_components=min(k, tf.shape[1] - 1), random_state=self.random_state).fit(tf)
            self.feature_names_ += [f"text:lsa_{i:02d}" for i in range(self.svd_.n_components)]
            self.feature_families_ += ["text"] * self.svd_.n_components
        if self._needs("lexical", "schema", "risk"):
            st = structured_frame(df)
            self.struct_medians_ = st.median(numeric_only=True)
            for fam, cols in (("lexical", LEXICAL_FEATURES), ("schema", SCHEMA_FEATURES), ("risk", RISK_FEATURES)):
                if fam in self.families:
                    self.feature_names_ += [f"{fam}:{c}" for c in cols]
                    self.feature_families_ += [fam] * len(cols)
        if "emb" in self.families:
            k = self.embedding_pca_components or cfg["embedding_pca_components"]
            E = self._encoder().encode(docs)
            self._encoder().save()
            self.embedding_backend_ = self._encoder().backend
            self.pca_ = PCA(n_components=min(k, E.shape[1], len(E) - 1), random_state=self.random_state).fit(E)
            self.feature_names_ += [f"emb:pc_{i:02d}" for i in range(self.pca_.n_components_)]
            self.feature_families_ += ["emb"] * self.pca_.n_components_
        return self

    # ---------------------------------------------------------- transform
    def transform(self, X):
        df = to_frame(X)
        docs = documents(df)
        blocks = []
        is_sparse = "tfidf" in self.families
        tf = self.tfidf_.transform(docs) if self._needs("tfidf", "text") else None
        if "tfidf" in self.families:
            blocks.append(tf)
        if "text" in self.families:
            blocks.append(self.svd_.transform(tf))
        if self._needs("lexical", "schema", "risk"):
            st = structured_frame(df).fillna(self.struct_medians_)
            for fam, cols in (("lexical", LEXICAL_FEATURES), ("schema", SCHEMA_FEATURES), ("risk", RISK_FEATURES)):
                if fam in self.families:
                    blocks.append(st[cols].to_numpy(dtype=np.float64))
        if "emb" in self.families:
            E = self._encoder().encode(docs)
            blocks.append(self.pca_.transform(E))
        if is_sparse:
            return sparse.hstack([b if sparse.issparse(b) else sparse.csr_matrix(b) for b in blocks]).tocsr()
        return np.hstack(blocks).astype(np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)

    def raw_embeddings(self, X) -> np.ndarray:
        return self._encoder().encode(documents(to_frame(X)))


def main() -> None:
    """Materialise full-dataset features for analysis (not used for training)."""
    from src.utils.config import resolve
    from src.utils.io import read_jsonl
    cfg = load_config()
    df = pd.DataFrame(read_jsonl(resolve(cfg["data"]["clean_file"])))
    st = structured_frame(df)
    out = pd.concat([df[["tool_id", "tool_name", "server_name", "source", "error_occurred", "error_type", "error_severity",
                         "failure_rate", "lineage_id"]], st], axis=1)
    out.to_csv(cfg.path("processed") / "structured_features.csv", index=False)
    enc = EmbeddingEncoder()
    E = enc.encode(documents(df))
    enc.save()
    np.save(cfg.path("processed") / "embeddings.npy", E)
    print(f"structured features: {st.shape}, embeddings: {E.shape} ({enc.backend})")


if __name__ == "__main__":
    main()
