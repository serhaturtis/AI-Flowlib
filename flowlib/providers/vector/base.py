"""Vector database provider base class and related functionality.

This module provides the base class for implementing vector database providers
that share common functionality for storing, retrieving, and searching
vector embeddings with metadata.
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
from pydantic import BaseModel, Field, ConfigDict

from flowlib.core.errors.errors import ProviderError, ErrorContext
from flowlib.providers.core.base import Provider, ProviderSettings
from flowlib.core.errors.models import ProviderErrorContext

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class VectorDBProviderSettings(ProviderSettings):
    """Base settings for vector database providers.
    
    Attributes:
        host: Vector database host address
        port: Vector database port
        api_key: API key for cloud vector databases
        username: Authentication username (if required)
        password: Authentication password (if required)
        index_name: Default vector index/collection name
        vector_dimension: Dimension of vector embeddings
        metric: Distance metric for similarity search
        batch_size: Batch size for bulk operations
    """
    
    # Connection settings
    host: Optional[str] = Field(default=None, description="Vector database server host")
    port: Optional[int] = Field(default=None, description="Vector database server port")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    username: Optional[str] = Field(default=None, description="Username for authentication")
    password: Optional[str] = Field(default=None, description="Password for authentication")
    
    # Vector settings
    index_name: str = Field(default="default", description="Default vector index/collection name")
    vector_dimension: int = Field(default=1536, description="Vector dimension (1536 for OpenAI embeddings)")
    metric: str = Field(default="cosine", description="Distance metric: cosine, euclidean, or dot")
    
    # Performance settings
    batch_size: int = Field(default=100, description="Batch size for bulk vector operations")
    timeout: float = Field(default=30.0, description="Operation timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")


class VectorMetadata(BaseModel):
    """Metadata for vector entries.
    
    Attributes:
        id: Unique identifier for the vector
        text: Original text that was embedded (if applicable)
        metadata: Custom metadata for the vector
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    id: str
    text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[int] = None
    updated_at: Optional[int] = None


class VectorSearchResult(BaseModel):
    """Result from a vector similarity search.
    
    Attributes:
        id: Vector ID
        score: Similarity score
        metadata: Vector metadata
        vector: Vector data (if requested)
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    id: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    text: Optional[str] = None
    vector: Optional[List[float]] = None


class VectorIndexStats(BaseModel):
    """Statistics for a vector index.
    
    Attributes:
        name: Index name
        total_vectors: Total number of vectors
        vector_dimension: Dimension of vectors
        index_size_bytes: Size of index in bytes
        last_updated: Last update timestamp
        metric: Distance metric used
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    name: str
    total_vectors: int = 0
    vector_dimension: int
    index_size_bytes: Optional[int] = None
    last_updated: Optional[str] = None
    metric: str = "cosine"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorInsertResult(BaseModel):
    """Result from vector insertion operation.
    
    Attributes:
        success: Whether insertion was successful
        inserted_ids: List of inserted vector IDs
        failed_ids: List of IDs that failed to insert
        error_details: Details about any errors
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    success: bool
    inserted_ids: List[str] = Field(default_factory=list)
    failed_ids: List[str] = Field(default_factory=list)
    error_details: Dict[str, Any] = Field(default_factory=dict)


class VectorDeleteResult(BaseModel):
    """Result from vector deletion operation.
    
    Attributes:
        success: Whether deletion was successful
        deleted_count: Number of vectors deleted
        not_found_ids: List of IDs that were not found
        error_details: Details about any errors
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    success: bool
    deleted_count: int = 0
    not_found_ids: List[str] = Field(default_factory=list)
    error_details: Dict[str, Any] = Field(default_factory=dict)


class VectorBatchSearchResult(BaseModel):
    """Result from batch vector search operation.
    
    Attributes:
        success: Whether search was successful
        results: List of search results for each query
        query_count: Number of queries processed
        total_results: Total number of results across all queries
        error_details: Details about any errors
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    success: bool
    results: List[List[VectorSearchResult]] = Field(default_factory=list)
    query_count: int = 0
    total_results: int = 0
    error_details: Dict[str, Any] = Field(default_factory=dict)


# Maintain backward compatibility
SimilaritySearchResult = VectorSearchResult


class VectorDBProvider(Provider):
    """Base class for vector database providers.
    
    This class provides:
    1. Vector storage and retrieval
    2. Similarity search
    3. Metadata storage and filtering
    4. Type-safe operations with Pydantic models
    """
    
    def __init__(self, name: str, provider_type: str, settings: Optional[VectorDBProviderSettings] = None, **kwargs: Any):
        """Initialize VectorDB provider.
        
        Args:
            name: Unique provider name
            provider_type: The type of the provider (e.g., 'vector_db')
            settings: Optional provider settings
            **kwargs: Additional keyword arguments for the base Provider
        """
        super().__init__(name=name, provider_type=provider_type, settings=settings, **kwargs)
        self._initialized = False
        self._client = None
        self._settings = settings or VectorDBProviderSettings()
        
    @property
    def initialized(self) -> bool:
        """Return whether provider has been initialized."""
        return self._initialized
        
    async def initialize(self):
        """Initialize the vector database connection.
        
        This method should be implemented by subclasses to establish
        connections to the vector database.
        """
        self._initialized = True
        
    async def shutdown(self):
        """Close all connections and release resources.
        
        This method should be implemented by subclasses to properly
        close connections and clean up resources.
        """
        self._initialized = False
        self._client = None
        
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
        raise NotImplementedError("Subclasses must implement create_index()")
        
    async def delete_index(self, index_name: str) -> bool:
        """Delete a vector index/collection.
        
        Args:
            index_name: Index name
            
        Returns:
            True if index was deleted successfully
            
        Raises:
            ProviderError: If index deletion fails
        """
        raise NotImplementedError("Subclasses must implement delete_index()")
        
    async def index_exists(self, index_name: str) -> bool:
        """Check if index exists.
        
        Args:
            index_name: Index name
            
        Returns:
            True if index exists, False otherwise
            
        Raises:
            ProviderError: If check fails
        """
        raise NotImplementedError("Subclasses must implement index_exists()")
        
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
        raise NotImplementedError("Subclasses must implement insert_vectors()")
        
    async def search_vectors(self, index_name: str, query_vector: List[float], top_k: int = 10, filter_conditions: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
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
        raise NotImplementedError("Subclasses must implement search_vectors()")
        
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
        raise NotImplementedError("Subclasses must implement delete_vectors()")
        
    async def get_index_stats(self, index_name: str) -> VectorIndexStats:
        """Get index statistics.
        
        Args:
            index_name: Index name
            
        Returns:
            Index statistics
            
        Raises:
            ProviderError: If getting stats fails
        """
        raise NotImplementedError("Subclasses must implement get_index_stats()")
        
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
        raise NotImplementedError("Subclasses must implement insert()")
        
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
        raise NotImplementedError("Subclasses must implement insert_batch()")
        
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
        raise NotImplementedError("Subclasses must implement get()")
        
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
        raise NotImplementedError("Subclasses must implement delete()")
        
    async def search(self, query_vector: List[float], top_k: int = 10, filter: Optional[Dict[str, Any]] = None,
                   include_vectors: bool = False, index_name: Optional[str] = None) -> List[VectorSearchResult]:
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
        raise NotImplementedError("Subclasses must implement search()")
        
    async def search_by_id(self, id: str, top_k: int = 10, filter: Optional[Dict[str, Any]] = None,
                         include_vectors: bool = False, index_name: Optional[str] = None) -> List[VectorSearchResult]:
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
        raise NotImplementedError("Subclasses must implement search_by_id()")
        
    async def search_structured(self, query_vector: List[float], output_type: Type[T], top_k: int = 10,
                              filter: Optional[Dict[str, Any]] = None, index_name: Optional[str] = None) -> List[T]:
        """Search for similar vectors and parse results into structured types.
        
        Args:
            query_vector: Query vector data
            output_type: Pydantic model for parsing results
            top_k: Number of results to return
            filter: Optional metadata filter
            index_name: Index name (default from settings if None)
            
        Returns:
            List of parsed model instances
            
        Raises:
            ProviderError: If search or parsing fails
        """
        try:
            # Perform the search
            results = await self.search(
                query_vector=query_vector,
                top_k=top_k,
                filter=filter,
                include_vectors=False,
                index_name=index_name
            )
            
            # Parse results into the output type
            parsed_results = []
            for result in results:
                # Combine metadata with score
                data = result.metadata.copy()
                data["score"] = result.score
                data["id"] = result.id
                
                # Parse into output type
                parsed_results.append(output_type.parse_obj(data))
                
            return parsed_results
            
        except Exception as e:
            # Wrap and re-raise errors with context
            raise ProviderError(
                message=f"Failed to perform structured vector search: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    output_type=output_type.__name__,
                    top_k=top_k,
                    filter=filter
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
        raise NotImplementedError("Subclasses must implement count()")
        
    async def get_by_filter(self, filter: Dict[str, Any], top_k: int = 10, 
                           include_vectors: bool = False, index_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get vectors by metadata filter without vector similarity search.
        
        This method is for metadata-only queries where you want to find vectors
        based on their metadata properties, not vector similarity.
        
        Args:
            filter: Metadata filter conditions
            top_k: Maximum number of results to return
            include_vectors: Whether to include vector data in results
            index_name: Index name (default from settings if None)
            
        Returns:
            List of results with id, metadata, and optionally vector data
            
        Raises:
            ProviderError: If query fails
        """
        raise NotImplementedError("Subclasses must implement get_by_filter()")

    async def check_connection(self) -> bool:
        """Check if vector database connection is active.
        
        Returns:
            True if connection is active, False otherwise
        """
        raise NotImplementedError("Subclasses must implement check_connection()") 