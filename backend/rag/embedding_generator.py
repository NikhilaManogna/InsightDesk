from __future__ import annotations

from hashlib import sha256
from time import perf_counter

import numpy as np

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Uses sentence-transformers when present, with a deterministic fallback."""

    def __init__(self, model_name: str, dimensions: int = 384) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self._model = self._load_model(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        started = perf_counter()
        if self._model is not None:
            vectors = self._model.encode(texts, normalize_embeddings=True)
            result = np.asarray(vectors, dtype="float32")
        else:
            result = np.asarray([self._fallback_vector(text) for text in texts], dtype="float32")
        logger.info(
            "embeddings_generated count=%s model=%s latency_ms=%s",
            len(texts),
            self.model_name if self._model else "hash-fallback",
            int((perf_counter() - started) * 1000),
        )
        return result

    @staticmethod
    def _load_model(model_name: str):
        try:
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(model_name)
        except Exception:
            logger.warning("embedding_model_unavailable using_hash_fallback=true model=%s", model_name)
            return None

    def _fallback_vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype="float32")
        for token in text.lower().replace("_", " ").split():
            digest = sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector
