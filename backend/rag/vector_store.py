from __future__ import annotations

import numpy as np

from backend.rag.rag_models import RetrievalResult, SchemaDocument


class VectorStore:
    def __init__(self) -> None:
        self.documents: list[SchemaDocument] = []
        self.vectors: np.ndarray | None = None
        self._index = None

    def build(self, documents: list[SchemaDocument], vectors: np.ndarray) -> None:
        self.documents = documents
        self.vectors = vectors.astype("float32")
        try:
            import faiss

            self._index = faiss.IndexFlatIP(self.vectors.shape[1])
            self._index.add(self.vectors)
        except Exception:
            self._index = None

    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievalResult]:
        if not self.documents or self.vectors is None:
            return []
        query = query_vector.astype("float32").reshape(1, -1)
        if self._index is not None:
            scores, indexes = self._index.search(query, min(top_k, len(self.documents)))
            return [
                RetrievalResult(self.documents[int(index)], float(score))
                for score, index in zip(scores[0], indexes[0])
                if index >= 0
            ]

        scores = self.vectors @ query[0]
        indexes = np.argsort(scores)[::-1][:top_k]
        return [RetrievalResult(self.documents[int(index)], float(scores[index])) for index in indexes]
