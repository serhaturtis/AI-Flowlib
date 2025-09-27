"""
Modern Vector Memory Component.

Provides semantic memory capabilities using vector embeddings and the
modernized agent framework patterns with config-driven providers.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

from pydantic import Field
from flowlib.core.models import StrictBaseModel
from flowlib.providers.vector.base import VectorDBProvider
from flowlib.providers.embedding.base import EmbeddingProvider

from ...core.errors import MemoryError
from .models import (
    MemoryStoreRequest,
    MemoryRetrieveRequest,
    MemorySearchRequest
)
# Import MemoryItem and MemorySearchResult from the proper location
from .models import MemoryItem, MemorySearchResult, MemorySearchMetadata
from flowlib.providers.core.registry import provider_registry

logger = logging.getLogger(__name__)


class VectorMemoryConfig(StrictBaseModel):
    """Configuration for vector memory."""
    
    vector_provider_config: str = Field(
        default="default-vector-db",
        description="Provider config name for vector database"
    )
    embedding_provider_config: str = Field(
        default="default-embedding",
        description="Provider config name for embedding generation"
    )
    collection_name: str = Field(
        default="agent_memory",
        description="Vector database collection name"
    )
    embedding_dimensions: Optional[int] = Field(
        default=None,
        description="Embedding dimensions (auto-detected if None)"
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for search results"
    )
    max_results: int = Field(
        default=50,
        ge=1,
        description="Maximum results to return from vector search"
    )
    distance_metric: str = Field(
        default="cosine",
        description="Vector similarity metric (cosine, euclidean, dot_product)"
    )


class VectorMemory:
    """Modern vector memory implementation with semantic search capabilities."""
    
    def __init__(self, config: Optional[VectorMemoryConfig] = None):
        """Initialize vector memory with config-driven providers."""
        self._config = config or VectorMemoryConfig()
        
        # Provider instances (resolved during initialization)
        self._vector_provider: Optional[VectorDBProvider] = None
        self._embedding_provider: Optional[EmbeddingProvider] = None
        
        # Contexts tracking
        self._contexts: set[str] = set()
        self._initialized = False
        
        logger.info(f"Initialized VectorMemory with config: {self._config}")
    
    @property
    def initialized(self) -> bool:
        """Check if memory is initialized."""
        return self._initialized
    
    async def initialize(self) -> None:
        """Initialize vector memory and providers."""
        if self._initialized:
            return
            
        logger.info("Initializing VectorMemory...")
        
        try:
            # Get providers using config-driven approach
            vector_provider = await provider_registry.get_by_config(
                self._config.vector_provider_config
            )
            if not vector_provider:
                raise MemoryError(
                    f"Vector provider not found: {self._config.vector_provider_config}",
                    operation="initialize",
                    provider_config=self._config.vector_provider_config
                )
            self._vector_provider = cast(VectorDBProvider, vector_provider)

            embedding_provider = await provider_registry.get_by_config(
                self._config.embedding_provider_config
            )
            if not embedding_provider:
                raise MemoryError(
                    f"Embedding provider not found: {self._config.embedding_provider_config}",
                    operation="initialize",
                    provider_config=self._config.embedding_provider_config
                )
            self._embedding_provider = cast(EmbeddingProvider, embedding_provider)
            
            # Ensure collection exists
            await self._ensure_collection()
            
            self._initialized = True
            logger.info("VectorMemory initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize VectorMemory: {e}")
            raise MemoryError(
                f"Vector memory initialization failed: {str(e)}",
                operation="initialize",
                cause=e
            ) from e
    
    async def shutdown(self) -> None:
        """Shutdown vector memory."""
        if not self._initialized:
            return
            
        logger.info("Shutting down VectorMemory...")
        
        # Providers are managed by the registry, no explicit cleanup needed
        self._vector_provider = None
        self._embedding_provider = None
        self._contexts.clear()
        
        self._initialized = False
        logger.info("VectorMemory shutdown completed")
    
    async def _ensure_collection(self) -> None:
        """Ensure the vector collection exists using standard interface methods."""
        try:
            # Use the standard index_exists method from base interface
            if self._vector_provider is None:
                raise RuntimeError("Vector provider not initialized")
            collection_exists = await self._vector_provider.index_exists(self._config.collection_name)
            
            if not collection_exists:
                # Use the standard create_index method from base interface
                # Use a default dimension if not specified (1536 is OpenAI's default)
                dimension = self._config.embedding_dimensions if self._config.embedding_dimensions is not None else 1536
                created = await self._vector_provider.create_index(
                    index_name=self._config.collection_name,
                    vector_dimension=dimension,
                    metric=self._config.distance_metric
                )
                if created:
                    logger.info(f"Created vector index: {self._config.collection_name}")
                else:
                    logger.warning(f"Failed to create vector index: {self._config.collection_name}")
            else:
                logger.debug(f"Vector index exists: {self._config.collection_name}")
                
        except AttributeError as e:
            # Provider doesn't have these methods - continue without them
            logger.debug(f"Provider doesn't support index management: {e}")
        except Exception as e:
            logger.warning(f"Could not ensure index exists: {e}")
            # Continue anyway - index might be created on first insert
    
    async def create_context(self, context_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a memory context."""
        if not self._initialized:
            raise MemoryError("VectorMemory not initialized")
            
        logger.debug(f"Creating context: {context_name}")
        
        # Vector memory doesn't need explicit context creation
        # Contexts are managed through metadata in vector records
        self._contexts.add(context_name)
        
        logger.debug(f"Context '{context_name}' registered")
        return context_name
    
    async def store(self, request: MemoryStoreRequest) -> str:
        """Store a memory item with vector embedding."""
        if not self._initialized:
            raise MemoryError("VectorMemory not initialized")
            
        try:
            # Generate embedding for the content
            content = self._extract_content(request.value)
            embedding = await self._get_embedding(content)
            
            # Create document ID
            doc_id = f"{request.context}_{request.key}"
            
            # Prepare metadata
            metadata = {
                "context": request.context,
                "key": request.key,
                "content": content,
                "item_type": type(request.value).__name__
            }
            
            # Add custom metadata from request
            if request.metadata:
                metadata.update(request.metadata)
            
            # Store in vector database using standard interface
            if self._vector_provider is None:
                raise RuntimeError("Vector provider not initialized")
            await self._vector_provider.insert_batch(
                vectors=[embedding],
                metadatas=[metadata],
                ids=[doc_id],
                index_name=self._config.collection_name
            )
            
            logger.debug(f"Stored vector memory item '{request.key}' in context '{request.context}'")
            return request.key
            
        except Exception as e:
            raise MemoryError(
                f"Failed to store vector memory item '{request.key}': {str(e)}",
                operation="store",
                context=request.context,
                key=request.key,
                cause=e
            ) from e
    
    async def retrieve(self, request: MemoryRetrieveRequest) -> Optional[MemoryItem]:
        """Retrieve a memory item by exact key match."""
        if not self._initialized:
            raise MemoryError("VectorMemory not initialized")
            
        try:
            # Query by metadata filter for exact match using new standard interface
            if self._vector_provider is None:
                raise RuntimeError("Vector provider not initialized")
            results = await self._vector_provider.get_by_filter(
                filter={
                    "context": request.context,
                    "key": request.key
                },
                top_k=1,
                index_name=self._config.collection_name
            )
            
            if results and len(results) > 0:
                # Reconstruct memory item from stored data - new format is list of dicts
                result_item = results[0]
                doc = result_item.get('document', '')
                metadata = result_item.get('metadata', {})
                
                # Create basic memory item - only pass user metadata to MemoryItem
                # System metadata stays in the vector db, user metadata goes to MemoryItem
                user_metadata = {}
                for key_meta, value_meta in metadata.items():
                    if key_meta not in ('context', 'key', 'content', 'item_type'):
                        user_metadata[key_meta] = value_meta
                
                item = MemoryItem(
                    key=request.key,
                    value=doc,
                    context=request.context if request.context is not None else "default",
                    updated_at=datetime.now()
                )
                
                logger.debug(f"Retrieved vector memory item '{request.key}' from context '{request.context}'")
                return item
            else:
                logger.debug(f"Vector memory item '{request.key}' not found in context '{request.context}'")
                return None
                
        except Exception as e:
            raise MemoryError(f"Vector retrieval failed: {e}", operation="retrieve", key=request.key, cause=e)
    
    async def search(self, request: MemorySearchRequest) -> List[MemorySearchResult]:
        """Search memory items using semantic similarity."""
        if not self._initialized:
            raise MemoryError("VectorMemory not initialized")
            
        try:
            # Generate query embedding
            query_embedding = await self._get_embedding(request.query)
            
            # Prepare context filter
            where_filter = {}
            if request.context:
                where_filter["context"] = request.context
            
            # Perform vector search using standard interface
            if self._vector_provider is None:
                raise RuntimeError("Vector provider not initialized")
            results = await self._vector_provider.search(
                query_vector=query_embedding,
                top_k=min(request.limit or self._config.max_results, self._config.max_results),
                filter=where_filter if where_filter else None,
                index_name=self._config.collection_name
            )
            
            search_results: List[MemorySearchResult] = []
            
            # Results are now List[SimilaritySearchResult] from standard interface
            for result in results:
                # Extract data from SimilaritySearchResult object
                doc = result.text or ""
                metadata = result.metadata or {}
                similarity = 1.0 - result.score  # Convert score back to similarity
                
                # Filter by similarity threshold
                if similarity >= self._config.similarity_threshold:
                    # Reconstruct memory item
                    # Fail fast - no fallbacks allowed
                    if 'key' not in metadata:
                        raise ValueError("Memory item metadata must contain 'key' field")
                    if 'context' not in metadata:
                        raise ValueError("Memory item metadata must contain 'context' field")
                        
                    item = MemoryItem(
                        key=metadata['key'],
                        value=doc,
                        context=metadata['context'],
                        updated_at=datetime.now()
                    )
                    
                    search_results.append(MemorySearchResult(
                        item=item,
                        score=similarity,
                        metadata=MemorySearchMetadata(
                            search_query=request.query,
                            search_type="semantic",
                            search_time_ms=0.0,
                            total_results=len(results),
                            result_rank=len(search_results) + 1
                        )
                    ))
            
            logger.debug(f"Found {len(search_results)} vector memory results for query '{request.query}'")
            return search_results
            
        except Exception as e:
            raise MemoryError(f"Vector search failed: {e}", operation="search", cause=e)
    
    async def retrieve_relevant(
        self, 
        query: str, 
        context: Optional[str] = None, 
        limit: int = 10
    ) -> List[str]:
        """Retrieve relevant memories based on semantic similarity."""
        if not self._initialized:
            raise MemoryError("VectorMemory not initialized")
            
        search_request = MemorySearchRequest(
            query=query,
            context=context,
            limit=limit,
            threshold=None,
            sort_by=None,
            search_type="hybrid",
            metadata_filter=None
        )
        
        search_results = await self.search(search_request)
        
        # Convert to string list format
        return [
            f"{result.item.key}: {result.item.value}"
            for result in search_results
        ]
    
    async def wipe_context(self, context: str) -> None:
        """Remove all items from a specific context."""
        if not self._initialized:
            raise MemoryError("VectorMemory not initialized")
            
        try:
            # Get all documents in this context using standard interface
            if self._vector_provider is None:
                raise RuntimeError("Vector provider not initialized")
            results = await self._vector_provider.get_by_filter(
                filter={"context": context},
                top_k=10000,  # Large number to get all
                index_name=self._config.collection_name
            )
            
            if results and len(results) > 0:
                ids_to_delete = [result['id'] for result in results]
                
                if ids_to_delete:
                    await self._vector_provider.delete_vectors(
                        index_name=self._config.collection_name,
                        ids=ids_to_delete
                    )
                    
                    logger.info(f"Wiped vector memory context '{context}', removed {len(ids_to_delete)} items")
                else:
                    logger.debug(f"Vector memory context '{context}' was empty")
            
            self._contexts.discard(context)
            
        except Exception as e:
            raise MemoryError(
                f"Failed to wipe vector memory context '{context}': {str(e)}",
                operation="wipe",
                context=context,
                cause=e
            ) from e
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using the embedding provider."""
        try:
            # Generate embedding
            if self._embedding_provider is None:
                raise RuntimeError("Embedding provider not initialized")
            embeddings = await self._embedding_provider.embed([text])

            # Type validation following flowlib's no-fallbacks principle
            if not embeddings or len(embeddings) == 0:
                raise MemoryError(
                    "Embedding provider returned empty result",
                    operation="embed",
                    text_length=len(text)
                )

            if not isinstance(embeddings[0], list) or not all(isinstance(x, (int, float)) for x in embeddings[0]):
                raise MemoryError(
                    f"Embedding provider returned invalid type: expected list[float], got {type(embeddings[0])}",
                    operation="embed",
                    text_length=len(text)
                )

            # Explicitly cast to ensure type correctness
            embedding: List[float] = [float(x) for x in embeddings[0]]
            return embedding
            
        except Exception as e:
            raise MemoryError(
                f"Failed to generate embedding: {str(e)}",
                operation="embed",
                text_length=len(text),
                cause=e
            ) from e
    
    def _extract_content(self, item: MemoryItem) -> str:
        """Extract searchable content from a memory item."""
        if hasattr(item, 'content'):
            return str(item.content)
        elif hasattr(item, 'text'):
            return str(item.text)
        elif hasattr(item, 'message'):
            return str(item.message)
        else:
            return str(item)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector memory statistics."""
        return {
            "initialized": self._initialized,
            "context_count": len(self._contexts),
            "contexts": list(self._contexts),
            "config": self._config.model_dump(),
            "providers": {
                "vector": self._config.vector_provider_config,
                "embedding": self._config.embedding_provider_config
            }
        }