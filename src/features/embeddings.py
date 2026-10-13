"""Sentence-embedding encoder with a content-addressed on-disk cache.

Backend ``sentence-transformers`` (default: all-MiniLM-L6-v2, 384-d) is used
when available; otherwise the encoder transparently falls back to an LSA
(TF-IDF + TruncatedSVD) embedding and records that in the model metadata.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from src.utils.config import load_config, resolve
from src.utils.logging_utils import get_logger, log_event

log = get_logger("features.embeddings")


def _h(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbeddingEncoder:
    _model_cache: dict = {}

    def __init__(self, model_name: str | None = None, backend: str | None = None, batch_size: int | None = None,
                 cache_file: str | None = None):
        cfg = load_config()["features"]["embedding"]
        self.model_name = model_name or cfg["model_name"]
        self.backend = backend or cfg["backend"]
        self.batch_size = batch_size or cfg["batch_size"]
        self.cache_path: Path | None = resolve(cache_file or cfg["cache_file"]) if (cache_file or cfg.get("cache_file")) else None
        self._cache: dict[str, np.ndarray] = {}
        self._dirty = False
        self._lsa = None
        if self.cache_path and self.cache_path.exists():
            data = np.load(self.cache_path, allow_pickle=False)
            self._cache = dict(zip(data["keys"].tolist(), data["vecs"]))

    # ---------------------------------------------------------------- model
    def _model(self):
        if self.backend != "sentence-transformers":
            return None
        if self.model_name not in self._model_cache:
            try:
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                from sentence_transformers import SentenceTransformer
                self._model_cache[self.model_name] = SentenceTransformer(self.model_name, device="cpu")
                log_event(log, "embedding model loaded", model=self.model_name)
            except Exception as exc:  # pragma: no cover - environment dependent
                log_event(log, "sentence-transformers unavailable, falling back to LSA", error=str(exc)[:200])
                self.backend = "lsa"
                return None
        return self._model_cache[self.model_name]

    @property
    def dim(self) -> int:
        m = self._model()
        return int(m.get_sentence_embedding_dimension()) if m is not None else 384

    # --------------------------------------------------------------- encode
    def encode(self, texts: list[str]) -> np.ndarray:
        texts = [t or "" for t in texts]
        model = self._model()
        if model is None:
            return self._encode_lsa(texts)
        keys = [_h(f"{self.model_name}|{t}") for t in texts]
        missing = sorted({(k, t) for k, t in zip(keys, texts) if k not in self._cache})
        if missing:
            vecs = model.encode([t for _, t in missing], batch_size=self.batch_size, normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True)
            for (k, _), v in zip(missing, vecs):
                self._cache[k] = v.astype(np.float32)
            self._dirty = True
        return np.vstack([self._cache[k] for k in keys]).astype(np.float32)

    def _encode_lsa(self, texts: list[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer
        if self._lsa is None:
            self._lsa = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1),
                                      TruncatedSVD(n_components=min(128, max(2, len(texts) - 1)), random_state=42),
                                      Normalizer())
            self._lsa.fit(texts)
        return self._lsa.transform(texts).astype(np.float32)

    def save(self) -> None:
        if self.cache_path and self._dirty and self._cache:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            keys = np.array(list(self._cache.keys()))
            vecs = np.vstack(list(self._cache.values()))
            np.savez_compressed(self.cache_path, keys=keys, vecs=vecs)
            self._dirty = False
