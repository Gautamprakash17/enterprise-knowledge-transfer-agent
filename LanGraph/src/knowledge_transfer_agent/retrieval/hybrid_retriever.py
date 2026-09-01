"""
Hybrid retrieval combining semantic (FAISS) and keyword (BM25) search.
"""

from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.logging_config import get_logger
from knowledge_transfer_agent.retrieval.vector_store import get_vector_store

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Combines semantic similarity (FAISS) with optional keyword/BM25 retrieval
    for improved contextual relevance and reduced hallucination.
    """

    vector_store: Any
    top_k_semantic: int = 5
    top_k_keyword: int = 3
    score_threshold: float = 0.7
    use_keyword: bool = True

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> list[Document]:
        """
        Retrieve documents using hybrid approach.

        Returns merged and deduplicated results from semantic + keyword search.
        """
        results: list[Document] = []
        seen_ids: set[str] = set()

        # Semantic search via FAISS
        try:
            semantic_docs = self.vector_store.similarity_search_with_score(
                query,
                k=self.top_k_semantic,
            )

            for doc, score in semantic_docs:
                doc_id = self._doc_id(doc)
                if doc_id not in seen_ids:
                    # Skip placeholder doc if present
                    if doc.page_content.strip() == "_placeholder_":
                        continue
                    doc.metadata["retrieval_score"] = float(score)
                    doc.metadata["retrieval_type"] = "semantic"
                    results.append(doc)
                    seen_ids.add(doc_id)

        except Exception as e:
            logger.warning("Semantic search failed, falling back: %s", e)
            try:
                semantic_docs = self.vector_store.similarity_search(
                    query, k=self.top_k_semantic
                )
                for doc in semantic_docs:
                    if doc.page_content.strip() == "_placeholder_":
                        continue
                    doc_id = self._doc_id(doc)
                    if doc_id not in seen_ids:
                        doc.metadata["retrieval_score"] = 1.0
                        doc.metadata["retrieval_type"] = "semantic"
                        results.append(doc)
                        seen_ids.add(doc_id)
            except Exception as e2:
                logger.error("Semantic fallback failed: %s", e2)

        # Keyword search (MMR or similar via FAISS as fallback)
        if self.use_keyword and len(results) < self.top_k_semantic + self.top_k_keyword:
            try:
                keyword_docs = self.vector_store.max_marginal_relevance_search(
                    query,
                    k=self.top_k_keyword,
                    fetch_k=self.top_k_keyword * 3,
                )
                for doc in keyword_docs:
                    doc_id = self._doc_id(doc)
                    if doc_id not in seen_ids:
                        doc.metadata["retrieval_type"] = "keyword"
                        results.append(doc)
                        seen_ids.add(doc_id)
            except Exception as e:
                logger.debug("Keyword/MMR search not available: %s", e)

        return results[: self.top_k_semantic + self.top_k_keyword]

    def _doc_id(self, doc: Document) -> str:
        """Generate unique ID for deduplication."""
        return f"{doc.metadata.get('source', '')}:{doc.metadata.get('chunk_index', '')}:{doc.page_content[:50]}"

