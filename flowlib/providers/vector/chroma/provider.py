"""ChromaDB vector database provider implementation.

This module provides a concrete implementation of the VectorDBProvider
for ChromaDB, an open-source embedding database.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Collection, Generic, TypeVar, Union
import uuid
import asyncio
from pydantic import Field

from flowlib.core.errors.errors import ProviderError, ErrorContext
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.vector.base import VectorDBProvider, SimilaritySearchResult
from flowlib.providers.core.base import ProviderSettings
from flowlib.providers.core.decorators import provider
# Removed ProviderType import - using config-driven provider access
from ..base import Provider # Import Provider base class
from flowlib.providers.embedding.base import EmbeddingProvider
# from ..registry import provider_registry

logger = logging.getLogger(__name__)


# Lazy import chromadb
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.api import Collection
    # Removed unused EmbeddingFunction, Embeddings types
except ImportError:
    logger.warning("ChromaDB package not found. Install with 'pip install chromadb'")
    chromadb = None
    # EmbeddingFunction = None # Removed

# Removed FlowlibChromaEmbeddingFunction wrapper class

class ChromaDBProviderSettings(ProviderSettings):
    """Settings for ChromaDB provider - direct inheritance, only ChromaDB-specific fields.
    
    ChromaDB can operate in multiple modes:
    1. Persistent (file-based): uses persist_directory
    2. HTTP client: uses http_host/http_port  
    3. In-memory: no persistence settings
    
    This follows Interface Segregation - only fields ChromaDB actually needs.
    """
    
    # ChromaDB-specific connection options
    persist_directory: Optional[str] = Field(
        default=None,
        description="Directory to persist ChromaDB data (e.g., './chroma_data', '/data/vectors')"
    )
    client_type: str = Field(
        default="persistent", 
        description="Client type: 'persistent' (local) or 'http' (server mode)"
    )
    
    # HTTP client settings (only used if client_type='http')
    http_host: Optional[str] = Field(default="localhost", description="HTTP client host")
    http_port: Optional[int] = Field(default=8000, description="HTTP client port")
    http_headers: Optional[Dict[str, str]] = Field(default=None, description="HTTP headers")
    
    # Aliases for compatibility
    host: Optional[str] = Field(default="localhost", description="Alias for http_host")
    port: Optional[int] = Field(default=8000, description="Alias for http_port")
    
    # ChromaDB-specific vector settings
    collection_name: str = Field(default="default", description="Collection name")
    index_name: str = Field(default="default", description="Index name (alias for collection_name)")
    vector_dimension: int = Field(default=384, description="Vector dimension size")
    distance_function: str = Field(
        default="cosine", 
        description="Distance function: 'cosine', 'l2', or 'ip'"
    )
    
    # ChromaDB-specific configuration
    anonymized_telemetry: bool = Field(default=True, description="Enable anonymized telemetry")

@provider(provider_type="vector_db", name="chroma", settings_class=ChromaDBProviderSettings)
class ChromaDBProvider(VectorDBProvider):
    """Provider for ChromaDB, an open-source embedding database.
    
    Manages vector storage and retrieval using ChromaDB.
    """
    
    def __init__(self, name: str, provider_type: str, settings: Optional[ChromaDBProviderSettings] = None, **kwargs: Any):
        """Initialize ChromaDB provider.
        
        Args:
            name: Unique provider name
            provider_type: The type of the provider (e.g., 'vector_db')
            settings: Optional provider settings
            **kwargs: Additional keyword arguments for the base VectorDBProvider
        """
        super().__init__(name=name, provider_type=provider_type, settings=settings, **kwargs)
        # self.provider_type = provider_type # Removed, base class handles this
        
        # Check settings type, allowing for mocking in tests but enforcing real type checking
        if self.settings is not None:
            try:
                # Try the isinstance check - this should work in normal cases
                if not isinstance(self.settings, ChromaDBProviderSettings):
                    # Check if it's a mock (allow mocks during testing)
                    if not ('Mock' in str(type(self.settings)) or hasattr(self.settings, '_mock_name')):
                        raise TypeError(f"settings must be a ChromaDBProviderSettings instance, got {type(self.settings)}")
            except TypeError as e:
                # If isinstance itself fails (ChromaDBProviderSettings is mocked), check if settings is a mock
                if not ('Mock' in str(type(self.settings)) or hasattr(self.settings, '_mock_name')):
                    # Re-raise the original TypeError if it's not a mock
                    raise e

        self._client = None
        self._collections = {}
        # Removed _embedding_provider and _embedding_function attributes
        
    async def initialize(self):
        """Initialize the ChromaDB client and default collection."""
        if self._initialized:
            return
            
        try:
            # Check if ChromaDB is installed
            if chromadb is None:
                raise ProviderError(
                    message="ChromaDB package not installed. Install with 'pip install chromadb'",
                    context=ErrorContext.create(
                        flow_name="chroma_provider",
                        error_type="ImportError",
                        error_location="initialize",
                        component=self.name,
                        operation="package_check"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="vector",
                        operation="package_check",
                        retry_count=0
                    )
                )
                
            # Removed embedding provider retrieval logic

            # Create client based on settings
            if self.settings.client_type == "persistent":
                # Ensure persistence directory exists
                persist_dir = self.settings.persist_directory or "./chroma_data"
                os.makedirs(persist_dir, exist_ok=True)
                
                self._client = chromadb.PersistentClient(
                    path=persist_dir,
                    settings=ChromaSettings(
                        anonymized_telemetry=self.settings.anonymized_telemetry
                    )
                )
            elif self.settings.client_type == "http":
                self._client = chromadb.HttpClient(
                    host=self.settings.http_host,
                    port=self.settings.http_port,
                    headers=self.settings.http_headers
                )
            else:
                # In-memory client as fallback
                self._client = chromadb.Client(
                    settings=ChromaSettings(
                        anonymized_telemetry=self.settings.anonymized_telemetry
                    )
                )
                
            # Create or get default collection (without embedding function)
            await self._get_or_create_collection(self.settings.index_name)
            
            self._initialized = True
            logger.debug(f"{self.name} provider initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.name} provider: {str(e)}")
            raise ProviderError(
                message=f"Failed to initialize ChromaDB provider: {str(e)}",
                context=ErrorContext.create(
                    flow_name="chroma_provider",
                    error_type="InitializationError",
                    error_location="initialize",
                    component=self.name,
                    operation="client_initialization"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="vector",
                    operation="client_initialization",
                    retry_count=0
                ),
                cause=e
            )
            
    async def shutdown(self):
        """Close ChromaDB client and release resources."""
        if not self._initialized:
            return
            
        try:
            # ChromaDB client doesn't have a close method, so we just 
            # nullify our reference
            self._client = None
            self._collections = {}
            self._initialized = False
            # Removed embedding provider cleanup
            
            logger.debug(f"{self.name} provider shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during {self.name} provider shutdown: {str(e)}")
            raise ProviderError(
                message=f"Failed to shut down ChromaDB provider: {str(e)}",
                context=ErrorContext.create(
                    flow_name="chroma_provider",
                    error_type="ShutdownError",
                    error_location="shutdown",
                    component=self.name,
                    operation="client_shutdown"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="vector",
                    operation="client_shutdown",
                    retry_count=0
                ),
                cause=e
            )
        
    async def _initialize(self) -> None:
        pass
    
    def _sanitize_collection_name(self, name: str) -> str:
        """Sanitize collection name to meet ChromaDB requirements.
        
        ChromaDB collection names must:
        1. Contain 3-63 characters
        2. Start and end with alphanumeric character
        3. Only contain alphanumeric characters, underscores or hyphens
        4. Not contain two consecutive periods
        5. Not be a valid IPv4 address
        """
        # Replace invalid characters with underscores
        sanitized = ""
        for char in name:
            if char.isalnum() or char in ['_', '-']:
                sanitized += char
            else:
                sanitized += '_'
        
        # Ensure it starts and ends with alphanumeric
        if sanitized and not sanitized[0].isalnum():
            sanitized = 'c' + sanitized
        if sanitized and not sanitized[-1].isalnum():
            sanitized = sanitized + 'c'
            
        # Ensure minimum length
        if len(sanitized) < 3:
            sanitized = sanitized + 'col'
            
        # Ensure maximum length
        if len(sanitized) > 63:
            sanitized = sanitized[:63]
            
        return sanitized
            
    async def _get_or_create_collection(
        self, 
        index_name: str
        # Removed embedding_function parameter
    ) -> Collection:
        """Get or create a ChromaDB collection.
        
        Args:
            index_name: Collection name
            
        Returns:
            ChromaDB collection
            
        Raises:
            ProviderError: If collection creation fails
        """
        # Use the provided index_name directly (it's already the desired collection name)
        collection_name = self._sanitize_collection_name(index_name)
        
        try:
            # Check if we already have this collection cached
            if collection_name in self._collections:
                return self._collections[collection_name]
            
            if not self._client:
                raise ProviderError(
                    message="Provider not initialized",
                    context=ErrorContext.create(
                        flow_name="chroma_provider",
                        error_type="InitializationError",
                        error_location="_get_or_create_collection",
                        component=self.name,
                        operation="collection_access"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="vector",
                        operation="collection_access",
                        retry_count=0
                    )
                )
                
            # Get or create collection using ChromaDB's built-in method
            # Note: Simplified without metadata to avoid version compatibility issues
            collection = self._client.get_or_create_collection(name=collection_name)
            logger.debug(f"Got or created collection: {collection_name}")
                
            # Cache and return the collection
            self._collections[collection_name] = collection
            return collection
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to get or create collection {collection_name}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="chroma_provider",
                    error_type="CollectionError",
                    error_location="_get_or_create_collection",
                    component=self.name,
                    operation="collection_creation"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="vector",
                    operation="collection_creation",
                    retry_count=0
                ),
                cause=e
            )
            
    async def create_index(self, index_name: str, vector_dimension: int, metric: str = "cosine", **kwargs) -> bool:
        """Create a new vector index/collection.
        
        Args:
            index_name: Index name
            vector_dimension: Vector dimension
            metric: Distance metric (cosine, euclidean, dot)
            **kwargs: Additional provider-specific arguments
            
        Returns:
            True if index was created successfully
            
        Raises:
            ProviderError: If index creation fails
        """
        try:
            # Get or create collection (no embedding function needed)
            await self._get_or_create_collection(index_name)
            return True
            
        except Exception as e:
            # Wrap and re-raise all errors with consistent message
            raise ProviderError(
                message=f"Failed to create index {index_name}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="chroma_provider",
                    error_type="IndexError",
                    error_location="create_index",
                    component=self.name,
                    operation="index_creation"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="vector",
                    operation="index_creation",
                    retry_count=0
                ),
                cause=e
            )
            
    async def delete_index(self, index_name: str) -> bool:
        """Delete a vector index/collection.
        
        Args:
            index_name: Index name
            
        Returns:
            True if index was deleted successfully
            
        Raises:
            ProviderError: If index deletion fails
        """
        try:
            # Use the provided index_name directly (it's already the desired collection name)
            collection_name = self._sanitize_collection_name(index_name)
            
            # Delete collection
            if not self._client:
                raise ProviderError(
                    message="Provider not initialized",
                    context=ErrorContext.create(
                        flow_name="chroma_provider",
                        error_type="InitializationError",
                        error_location="delete_index",
                        component=self.name,
                        operation="index_deletion"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="vector",
                        operation="index_deletion",
                        retry_count=0
                    )
                )
                
            try:
                self._client.delete_collection(name=collection_name)
                logger.debug(f"Deleted collection: {collection_name}")
                
                # Remove from cache
                if collection_name in self._collections:
                    del self._collections[collection_name]
                    
                return True
                
            except ValueError:
                # Collection doesn't exist, treat as success
                logger.debug(f"Collection {collection_name} doesn't exist, nothing to delete")
                return True
                
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to delete index {index_name}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="chroma_provider",
                    error_type="IndexError",
                    error_location="delete_index",
                    component=self.name,
                    operation="index_deletion"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="vector",
                    operation="index_deletion",
                    retry_count=0
                ),
                cause=e
            )
            
    async def index_exists(self, index_name: str) -> bool:
        """Check if index exists.
        
        Args:
            index_name: Index name
            
        Returns:
            True if index exists, False otherwise
            
        Raises:
            ProviderError: If check fails
        """
        try:
            if not self._client:
                raise ProviderError(
                    message="Provider not initialized",
                    context=ErrorContext.create(
                        flow_name="chroma_provider",
                        error_type="InitializationError",
                        error_location="index_exists",
                        component=self.name,
                        operation="index_check"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="vector",
                        operation="index_check",
                        retry_count=0
                    )
                )
                
            # Use the provided index_name directly (it's already the desired collection name)
            collection_name = self._sanitize_collection_name(index_name)
            
            # List collections and check if our collection exists
            collections = self._client.list_collections()
            return collection_name in [col.name for col in collections]
                
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to check if index {index_name} exists: {str(e)}",
                context=ErrorContext.create(
                    flow_name="chroma_provider",
                    error_type="IndexError",
                    error_location="index_exists",
                    component=self.name,
                    operation="index_check"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="vector",
                    operation="index_check",
                    retry_count=0
                ),
                cause=e
            )

    async def insert_vectors(self, index_name: str, vectors: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> bool:
        """Insert multiple vectors with metadata.
        
        Args:
            index_name: Index name
            vectors: List of vector data
            metadata: Optional list of vector metadata
            ids: Optional list of vector IDs (generated if None)
            
        Returns:
            True if insertion was successful
            
        Raises:
            ProviderError: If insertion fails
        """
        try:
            # Get collection
            collection = await self._get_or_create_collection(index_name)
            
            # Validate input lengths
            vector_count = len(vectors)
            if metadata and len(metadata) != vector_count:
                raise ValueError(f"Number of vectors ({vector_count}) must match number of metadata items ({len(metadata)})")
                
            # Generate IDs if not provided
            if ids is None:
                ids = [uuid.uuid4().hex for _ in range(vector_count)]
            elif len(ids) != vector_count:
                raise ValueError(f"Number of vectors ({vector_count}) must match number of ids ({len(ids)})")
                
            # Clean metadata (ChromaDB doesn't support nested objects)
            cleaned_metadatas = []
            for i in range(vector_count):
                if metadata and i < len(metadata):
                    cleaned_metadatas.append(self._clean_metadata(metadata[i]))
                else:
                    cleaned_metadatas.append({})
            
            # Add vectors to collection
            collection.add(
                embeddings=vectors,
                metadatas=cleaned_metadatas,
                ids=ids
            )
            
            logger.debug(f"Inserted {vector_count} vectors into {index_name}")
            return True
            
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to insert vectors: {str(e)}",
                context=ErrorContext.create(
                    flow_name="chroma_provider",
                    error_type="InsertError",
                    error_location="insert_vectors",
                    component=self.name,
                    operation="vector_insertion"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="vector",
                    operation="vector_insertion",
                    retry_count=0
                ),
                cause=e
            )
            
    async def search_vectors(self, index_name: str, query_vector: List[float], top_k: int = 10, filter_conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            index_name: Index name
            query_vector: Query vector data
            top_k: Number of results to return
            filter_conditions: Optional metadata filter
            
        Returns:
            List of search results with id, score, and metadata
            
        Raises:
            ProviderError: If search fails
        """
        try:
            # Get collection
            collection = await self._get_or_create_collection(index_name)
            
            # Perform similarity search
            query_result = collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=filter_conditions
            )
            
            # Parse results
            results = []
            if query_result and "ids" in query_result and query_result["ids"] and query_result["ids"][0]:
                ids = query_result["ids"][0]
                if "distances" not in query_result:
                    raise ValueError("Query result missing required 'distances' field")
                distances = query_result["distances"][0] if query_result["distances"] else []
                if "metadatas" not in query_result:
                    raise ValueError("Query result missing required 'metadatas' field")
                metadatas = query_result["metadatas"][0] if query_result["metadatas"] else []

                for i in range(len(ids)):
                    # Convert distance to score (1 - distance for cosine)
                    distance = distances[i] if i < len(distances) else 0.0
                    score = 1.0 - distance if distance <= 1.0 else distance
                    
                    result = {
                        "id": ids[i],
                        "score": score,
                        "metadata": metadatas[i] if i < len(metadatas) else {}
                    }
                    results.append(result)
                    
            return results
            
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to search vectors: {str(e)}",
                context=ErrorContext.create(
                    flow_name="chroma_provider",
                    error_type="SearchError",
                    error_location="search_vectors",
                    component=self.name,
                    operation="vector_search"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="vector",
                    operation="vector_search",
                    retry_count=0
                ),
                cause=e
            )
            
    async def delete_vectors(self, index_name: str, ids: List[str]) -> bool:
        """Delete vectors by IDs.
        
        Args:
            index_name: Index name
            ids: List of vector IDs to delete
            
        Returns:
            True if deletion was successful
            
        Raises:
            ProviderError: If deletion fails
        """
        try:
            # Get collection
            collection = await self._get_or_create_collection(index_name)
            
            # Delete vectors by ID
            collection.delete(ids=ids)
            logger.debug(f"Deleted {len(ids)} vectors from {index_name}")
            
            return True
            
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to delete vectors: {str(e)}",
                context=ErrorContext.create(
                    flow_name="chroma_provider",
                    error_type="DeleteError",
                    error_location="delete_vectors",
                    component=self.name,
                    operation="vector_deletion"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="vector",
                    operation="vector_deletion",
                    retry_count=0
                ),
                cause=e
            )
            
    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get index statistics.
        
        Args:
            index_name: Index name
            
        Returns:
            Dictionary with index statistics
            
        Raises:
            ProviderError: If getting stats fails
        """
        try:
            # Get collection
            collection = await self._get_or_create_collection(index_name)
            
            # Get collection count
            total_vectors = collection.count()
            
            # Try to get dimension from actual data if possible
            dimension = self.settings.vector_dimension
            try:
                peek_result = collection.peek(1)
                if peek_result and "embeddings" in peek_result and peek_result["embeddings"]:
                    # The embeddings are nested: [[[embedding_vector]]]
                    if peek_result["embeddings"][0]:
                        dimension = len(peek_result["embeddings"][0][0])
            except Exception:
                pass  # Use default dimension if peek fails
            
            return {
                "total_vectors": total_vectors,
                "dimension": dimension,
                "distance_function": self.settings.distance_function
            }
            
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to get index stats: {str(e)}",
                context=ErrorContext.create(
                    flow_name="chroma_provider",
                    error_type="StatsError",
                    error_location="get_index_stats",
                    component=self.name,
                    operation="stats_retrieval"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="vector",
                    operation="stats_retrieval",
                    retry_count=0
                ),
                cause=e
            )
            
    async def insert(self, vector: List[float], metadata: Dict[str, Any], id: Optional[str] = None,
                   index_name: Optional[str] = None) -> str:
        """Insert a vector with metadata.
        
        Args:
            vector: Vector data (embedding)
            metadata: Vector metadata
            id: Optional vector ID (generated if None)
            index_name: Index name (default from settings if None)
            
        Returns:
            Vector ID
            
        Raises:
            ProviderError: If insertion fails
        """
        try:
            # Use settings as defaults
            index_name = index_name or self.settings.index_name
            
            # Generate ID if not provided
            id = id or uuid.uuid4().hex
            
            # Get collection (no embedding function needed)
            collection = await self._get_or_create_collection(index_name)
            
            # Clean metadata (ChromaDB doesn't support nested objects)
            cleaned_metadata = self._clean_metadata(metadata)
            
            # Add vector to collection
            collection.add(
                embeddings=[vector],
                metadatas=[cleaned_metadata],
                ids=[id]
            )
            
            logger.debug(f"Inserted vector with ID {id} into {index_name}")
            return id
            
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to insert vector: {str(e)}",
                context=ErrorContext.create(
                    flow_name="chroma_provider",
                    error_type="InsertError",
                    error_location="insert",
                    component=self.name,
                    operation="single_vector_insertion"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="vector",
                    operation="single_vector_insertion",
                    retry_count=0
                ),
                cause=e
            )
            
    async def insert_batch(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]],
                         ids: Optional[List[str]] = None, index_name: Optional[str] = None) -> List[str]:
        """Insert multiple vectors with metadata.
        
        Args:
            vectors: List of vector data
            metadatas: List of vector metadata
            ids: Optional list of vector IDs (generated if None)
            index_name: Index name (default from settings if None)
            
        Returns:
            List of vector IDs
            
        Raises:
            ProviderError: If batch insertion fails
        """
        try:
            # Use settings as defaults
            index_name = index_name or self.settings.index_name
            
            # Get collection (no embedding function needed)
            collection = await self._get_or_create_collection(index_name)
            
            # Validate input lengths
            vector_count = len(vectors)
            if len(metadatas) != vector_count:
                raise ValueError(f"Number of vectors ({vector_count}) must match number of metadatas ({len(metadatas)})")
                
            # Generate IDs if not provided
            if ids is None:
                ids = [uuid.uuid4().hex for _ in range(vector_count)]
            elif len(ids) != vector_count:
                raise ValueError(f"Number of vectors ({vector_count}) must match number of ids ({len(ids)})")
                
            # Clean metadata (ChromaDB doesn't support nested objects)
            cleaned_metadatas = [self._clean_metadata(metadata) for metadata in metadatas]
            
            # Add vectors to collection
            collection.add(
                embeddings=vectors,
                metadatas=cleaned_metadatas,
                ids=ids
            )
            
            logger.debug(f"Inserted {vector_count} vectors into {index_name}")
            return ids
            
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to insert vectors in batch: {str(e)}",
                context=ErrorContext.create(
                    flow_name="chroma_provider",
                    error_type="BatchInsertError",
                    error_location="insert_batch",
                    component=self.name,
                    operation="batch_vector_insertion"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="vector",
                    operation="batch_vector_insertion",
                    retry_count=0
                ),
                cause=e
            )
            
    async def get(self, id: str, include_vector: bool = False,
                index_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a vector by ID.
        
        Args:
            id: Vector ID
            include_vector: Whether to include vector data
            index_name: Index name (default from settings if None)
            
        Returns:
            Vector metadata and optionally vector data, or None if not found
            
        Raises:
            ProviderError: If retrieval fails
        """
        try:
            # Use settings as defaults
            index_name = index_name or self.settings.index_name
            
            # Get collection (no embedding function needed)
            collection = await self._get_or_create_collection(index_name)
            
            # Get vector by ID
            result = collection.get(
                ids=[id],
                include_embeddings=include_vector
            )
            
            # Check if any results were returned
            if not result["ids"]:
                return None
                
            # Construct response
            response = {
                "id": result["ids"][0],
                "metadata": result["metadatas"][0]
            }
            
            # Add vector if requested
            if include_vector and "embeddings" in result:
                response["vector"] = result["embeddings"][0]
                
            return response
            
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to get vector {id}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="chroma_provider",
                    error_type="RetrievalError",
                    error_location="get",
                    component=self.name,
                    operation="vector_retrieval"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="vector",
                    operation="vector_retrieval",
                    retry_count=0
                ),
                cause=e
            )
            
    async def delete(self, id: str, index_name: Optional[str] = None) -> bool:
        """Delete a vector by ID.
        
        Args:
            id: Vector ID
            index_name: Index name (default from settings if None)
            
        Returns:
            True if vector was deleted successfully
            
        Raises:
            ProviderError: If deletion fails
        """
        try:
            # Use settings as defaults
            index_name = index_name or self.settings.index_name
            
            # Get collection (no embedding function needed)
            collection = await self._get_or_create_collection(index_name)
            
            # Delete vector by ID
            collection.delete(ids=[id])
            logger.debug(f"Deleted vector with ID {id} from {index_name}")
            
            return True
            
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to delete vector {id}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="chroma_provider",
                    error_type="DeleteError",
                    error_location="delete",
                    component=self.name,
                    operation="single_vector_deletion"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="vector",
                    operation="single_vector_deletion",
                    retry_count=0
                ),
                cause=e
            )
            
    async def search(self, query_vector: List[float], top_k: int = 10, filter: Optional[Dict[str, Any]] = None,
                   include_vectors: bool = False, index_name: Optional[str] = None) -> List[SimilaritySearchResult]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector data
            top_k: Number of results to return
            filter: Optional metadata filter
            include_vectors: Whether to include vector data in results
            index_name: Index name (default from settings if None)
            
        Returns:
            List of similarity search results
            
        Raises:
            ProviderError: If search fails
        """
        try:
            # Use settings as defaults
            index_name = index_name or self.settings.index_name
            
            # Get collection (no embedding function needed)
            collection = await self._get_or_create_collection(index_name)
            
            # Perform similarity search
            query_result = collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=filter,
                include=['metadatas', 'documents', 'distances'] + (['embeddings'] if include_vectors else [])
            )
            
            # Parse results
            results = []
            if query_result and "ids" in query_result and query_result["ids"] and query_result["ids"][0]:
                ids = query_result["ids"][0]
                if "distances" not in query_result:
                    raise ValueError("Query result missing required 'distances' field")
                distances = query_result["distances"][0] if query_result["distances"] else []
                if "metadatas" not in query_result:
                    raise ValueError("Query result missing required 'metadatas' field")
                metadatas = query_result["metadatas"][0] if query_result["metadatas"] else []
                if "embeddings" not in query_result:
                    embeddings = []
                else:
                    embeddings = query_result["embeddings"][0] if query_result["embeddings"] else []
                if "documents" not in query_result:
                    documents = []
                else:
                    documents = query_result["documents"][0] if query_result["documents"] else []

                for i in range(len(ids)):
                    result = SimilaritySearchResult(
                        id=ids[i],
                        score=distances[i] if i < len(distances) else 0.0,
                        metadata=metadatas[i] if i < len(metadatas) else {},
                        text=documents[i] if i < len(documents) else None
                    )
                    
                    # Add vector if requested and available
                    if include_vectors and embeddings and i < len(embeddings):
                        result.vector = embeddings[i]
                        
                    results.append(result)
                    
            return results
            
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to search vectors: {str(e)}",
                context=ErrorContext.create(
                    flow_name="chroma_provider",
                    error_type="SearchError",
                    error_location="search",
                    component=self.name,
                    operation="similarity_search"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="vector",
                    operation="similarity_search",
                    retry_count=0
                ),
                cause=e
            )
            
    async def search_by_id(self, id: str, top_k: int = 10, filter: Optional[Dict[str, Any]] = None,
                         include_vectors: bool = False, index_name: Optional[str] = None) -> List[SimilaritySearchResult]:
        """Search for similar vectors using an existing vector ID.
        
        Args:
            id: Vector ID to use as query
            top_k: Number of results to return
            filter: Optional metadata filter
            include_vectors: Whether to include vector data in results
            index_name: Index name (default from settings if None)
            
        Returns:
            List of similarity search results
            
        Raises:
            ProviderError: If search fails
        """
        try:
            # Use settings as defaults
            index_name = index_name or self.settings.index_name
            
            # Get the vector by ID first
            vector_data = await self.get(id, include_vector=True, index_name=index_name)
            
            if not vector_data or "vector" not in vector_data:
                raise ProviderError(
                    message=f"Vector with ID {id} not found",
                    context=ErrorContext.create(
                        flow_name="chroma_provider",
                        error_type="NotFoundError",
                        error_location="search_by_id",
                        component=self.name,
                        operation="vector_lookup"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="vector",
                        operation="vector_lookup",
                        retry_count=0
                    )
                )
                
            # Now search using the vector
            return await self.search(
                query_vector=vector_data["vector"],
                top_k=top_k,
                filter=filter,
                include_vectors=include_vectors,
                index_name=index_name
            )
            
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to search by ID {id}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="chroma_provider",
                    error_type="SearchError",
                    error_location="search_by_id",
                    component=self.name,
                    operation="search_by_vector_id"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="vector",
                    operation="search_by_vector_id",
                    retry_count=0
                ),
                cause=e
            )
            
    async def count(self, filter: Optional[Dict[str, Any]] = None, index_name: Optional[str] = None) -> int:
        """Count vectors in the index.
        
        Args:
            filter: Optional metadata filter
            index_name: Index name (default from settings if None)
            
        Returns:
            Vector count
            
        Raises:
            ProviderError: If count fails
        """
        try:
            # Use settings as defaults
            index_name = index_name or self.settings.index_name
            
            # Get collection (no embedding function needed)
            collection = await self._get_or_create_collection(index_name)
            
            # Count with filter if provided
            if filter:
                result = collection.get(where=filter)
                return len(result["ids"]) if "ids" in result else 0
            else:
                # Get collection info for total count
                return collection.count()
                
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to count vectors: {str(e)}",
                context=ErrorContext.create(
                    flow_name="chroma_provider",
                    error_type="CountError",
                    error_location="count",
                    component=self.name,
                    operation="vector_count"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="vector",
                    operation="vector_count",
                    retry_count=0
                ),
                cause=e
            )
            
    async def check_connection(self) -> bool:
        """Check if vector database connection is active.
        
        Returns:
            True if connection is active, False otherwise
        """
        if not self._initialized or not self._client:
            return False
            
        try:
            # Try to access the client's heartbeat method
            self._client.heartbeat()
            return True
        except Exception as e:
            logger.error(f"ChromaDB connection check failed: {str(e)}")
            raise ProviderError(
                message=f"ChromaDB connection check failed: {str(e)}",
                context=ErrorContext.create(
                    flow_name="chroma_provider",
                    error_type="ConnectionError",
                    error_location="check_connection",
                    component=self.name,
                    operation="health_check"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="vector",
                    operation="health_check",
                    retry_count=0
                ),
                cause=e
            )
            
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata for ChromaDB compatibility.
        
        ChromaDB only supports simple metadata types (str, int, float, bool).
        This method flattens and converts complex types.
        
        Args:
            metadata: Original metadata
            
        Returns:
            Cleaned metadata
        """
        cleaned = {}
        
        for key, value in metadata.items():
            # Skip None values
            if value is None:
                continue
                
            # Handle lists and dicts by converting to strings
            if isinstance(value, (list, dict)):
                import json
                try:
                    cleaned[key] = json.dumps(value)
                except TypeError:
                    cleaned[key] = str(value)
            # Handle basic types
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            # Convert other types to strings
            else:
                cleaned[key] = str(value)
                
        return cleaned
        
    def _map_distance_function(self, metric: str) -> str:
        """Map metric name to ChromaDB space.
        
        Args:
            metric: Distance metric name
            
        Returns:
            ChromaDB space name
        """
        mapping = {
            "cosine": "cosine",
            "l2": "l2", 
            "ip": "ip",
            "euclidean": "l2"  # Map euclidean to l2
        }
        if metric not in mapping:
            raise ValueError(f"Unsupported distance metric: {metric}. Supported metrics: {list(mapping.keys())}")
        return mapping[metric] 