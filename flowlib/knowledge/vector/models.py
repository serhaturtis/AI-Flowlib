"""Models for vector storage flow."""

from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import Field
from flowlib.core.models import StrictBaseModel

from flowlib.knowledge.models.models import (
    VectorStoreInput,
    VectorStoreOutput,
    VectorEmbedding,
    VectorSearchResult,
    DocumentContent
)


class VectorDocument(StrictBaseModel):
    """Document indexed in vector store."""
    document_id: str
    chunk_count: int
    status: str
    indexed_at: datetime


class SearchQuery(StrictBaseModel):
    """Query for vector search."""
    query_text: str
    collection_name: str
    top_k: int = 10
    vector_provider_name: Optional[str] = None
    embedding_provider_name: Optional[str] = None
    vector_dimensions: Optional[int] = None
    filter_metadata: Optional[Dict[str, Any]] = None


class SearchResult(StrictBaseModel):
    """Search result from vector store."""
    document_id: str
    chunk_index: int
    score: float
    text: str
    metadata: Dict[str, Any]
    rank: int


__all__ = [
    "VectorStoreInput",
    "VectorStoreOutput", 
    "VectorEmbedding",
    "VectorSearchResult",
    "DocumentContent",
    "VectorDocument",
    "SearchQuery",
    "SearchResult"
]