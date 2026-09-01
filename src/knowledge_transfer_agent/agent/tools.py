"""
Agent tools for retrieval. Context formatting moved to context_formatter.
"""

from typing import Any

from langchain_core.tools import tool

from knowledge_transfer_agent.retrieval.reranker import maybe_rerank
from knowledge_transfer_agent.retrieval.retriever_factory import get_hybrid_retriever


def create_retrieval_tool() -> Any:
    """Create the retrieval tool (for external use, e.g. as LangChain tool)."""
    retriever = get_hybrid_retriever()

    @tool
    def search_knowledge_base(query: str) -> list[dict[str, Any]]:
        """Search the internal knowledge base for documentation."""
        docs = retriever.invoke(query)
        docs = maybe_rerank(query, docs)
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "source_type": doc.metadata.get("source_type", "generic"),
                "doc_id": doc.metadata.get("doc_id", ""),
            }
            for doc in docs
        ]

    return search_knowledge_base
