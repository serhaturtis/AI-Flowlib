"""Vector database provider package.

This package contains providers for vector databases, offering a common
interface for working with different vector database systems.
"""

from .base import (
    SimilaritySearchResult,
    VectorDBProvider,
    VectorDBProviderSettings,
    VectorMetadata,
)
from .chroma.provider import ChromaDBProvider, ChromaDBProviderSettings
from .pinecone.provider import PineconeProvider, PineconeProviderSettings
from .qdrant.provider import QdrantProvider, QdrantProviderSettings

__all__ = [
    "VectorDBProvider",
    "VectorDBProviderSettings",
    "VectorMetadata",
    "SimilaritySearchResult",
    "ChromaDBProvider",
    "ChromaDBProviderSettings",
    "PineconeProvider",
    "PineconeProviderSettings",
    "QdrantProvider",
    "QdrantProviderSettings",
]

# Makes vector a package
