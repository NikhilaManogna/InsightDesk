from __future__ import annotations

from time import perf_counter

from backend.db.schema_metadata import SchemaMetadata
from backend.rag.embedding_generator import EmbeddingGenerator
from backend.rag.retrieval_ranker import RetrievalRanker
from backend.rag.schema_embedder import SchemaEmbedder
from backend.rag.vector_store import VectorStore
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class SchemaRetriever:
    def __init__(
        self,
        model_name: str,
        max_tables: int,
        aliases: dict[str, str] | None = None,
    ) -> None:
        self.embedder = SchemaEmbedder()
        self.embedding_generator = EmbeddingGenerator(model_name)
        self.vector_store = VectorStore()
        self.ranker = RetrievalRanker()
        self.max_tables = max_tables
        self.aliases = aliases or {}

    def retrieve(self, question: str, metadata: SchemaMetadata) -> list[str]:
        started = perf_counter()
        documents = self.embedder.documents(metadata, self.aliases)
        if not documents:
            return []
        vectors = self.embedding_generator.embed([doc.text for doc in documents])
        self.vector_store.build(documents, vectors)
        query = self._expand_business_terms(question)
        query_vector = self.embedding_generator.embed([query])[0]
        results = self.vector_store.search(query_vector, top_k=max(self.max_tables * 2, 4))
        tables = self.ranker.table_names(results, self.max_tables)
        logger.info(
            "schema_retrieved tables=%s latency_ms=%s",
            ",".join(tables),
            int((perf_counter() - started) * 1000),
        )
        return tables

    def _expand_business_terms(self, question: str) -> str:
        expanded = question
        lowered = question.lower()
        for alias, target in self.aliases.items():
            if alias.lower() in lowered:
                expanded += f" {target}"
        return expanded
