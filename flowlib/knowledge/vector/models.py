"""Models for vector storage flow."""

from typing import Any

from flowlib.core.models import StrictBaseModel
from flowlib.knowledge.models import (
    DocumentContent,
    VectorEmbedding,
    VectorSearchResult,
    VectorStoreInput,
    VectorStoreOutput,
)


class SearchQuery(StrictBaseModel):
    """Query for vector search."""

    query_text: str
    collection_name: str
    top_k: int = 10
    vector_provider_config: str | None = "default-vector-db"
    embedding_provider_config: str | None = "default-embedding"
    vector_dimensions: int | None = None
    filter_metadata: dict[str, Any] | None = None


class SearchResult(StrictBaseModel):
    """Search result from vector store."""

    document_id: str
    chunk_index: int
    score: float
    text: str
    metadata: dict[str, Any]
    rank: int


__all__ = [
    "VectorStoreInput",
    "VectorStoreOutput",
    "VectorEmbedding",
    "VectorSearchResult",
    "DocumentContent",
    "SearchQuery",
    "SearchResult",
]
