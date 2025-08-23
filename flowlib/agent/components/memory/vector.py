"""
Modern Vector Memory Component.

Provides semantic memory capabilities using vector embeddings and the
modernized agent framework patterns with config-driven providers.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import Field
from flowlib.core.models import StrictBaseModel

from ...core.errors import MemoryError, ErrorContext
from .interfaces import MemoryInterface
from .models import (
    MemoryStoreRequest,
    MemoryRetrieveRequest,
    MemorySearchRequest,
    MemoryContext
)
# Import MemoryItem and MemorySearchResult from the same location to ensure compatibility
from ...models.memory import MemoryItem, MemorySearchResult
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


class VectorMemory(MemoryInterface):
    """Modern vector memory implementation with semantic search capabilities."""
    
    def __init__(self, config: Optional[VectorMemoryConfig] = None):
        """Initialize vector memory with config-driven providers."""
        self._config = config or VectorMemoryConfig()
        
        # Provider instances (resolved during initialization)
        self._vector_provider = None
        self._embedding_provider = None
        
        # Contexts tracking
        self._contexts = set()
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
            self._vector_provider = await provider_registry.get_by_config(
                self._config.vector_provider_config
            )
            if not self._vector_provider:
                raise MemoryError(
                    f"Vector provider not found: {self._config.vector_provider_config}",
                    operation="initialize",
                    provider_config=self._config.vector_provider_config
                )
            
            self._embedding_provider = await provider_registry.get_by_config(
                self._config.embedding_provider_config
            )
            if not self._embedding_provider:
                raise MemoryError(
                    f"Embedding provider not found: {self._config.embedding_provider_config}",
                    operation="initialize",
                    provider_config=self._config.embedding_provider_config
                )
            
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
        """Ensure the vector collection exists."""
        try:
            # Check if collection exists
            collections = await self._vector_provider.list_collections()
            if self._config.collection_name not in collections:
                # Create collection
                await self._vector_provider.create_collection(
                    name=self._config.collection_name,
                    dimension=self._config.embedding_dimensions
                )
                logger.info(f"Created vector collection: {self._config.collection_name}")
            else:
                logger.debug(f"Vector collection exists: {self._config.collection_name}")
                
        except Exception as e:
            logger.warning(f"Could not ensure collection exists: {e}")
            # Continue anyway - collection might be created on first insert
    
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
            
            # Store in vector database
            await self._vector_provider.add_documents(
                collection_name=self._config.collection_name,
                documents=[content],
                embeddings=[embedding],
                ids=[doc_id],
                metadatas=[metadata]
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
            # Query by metadata filter for exact match
            results = await self._vector_provider.query(
                collection_name=self._config.collection_name,
                query_embeddings=None,
                n_results=1,
                where={
                    "context": request.context,
                    "key": request.key
                }
            )
            
            if results and 'documents' in results and results['documents'] and len(results['documents']) > 0:
                # Reconstruct memory item from stored data
                doc = results['documents'][0][0] if isinstance(results['documents'][0], list) else results['documents'][0]
                metadatas = results['metadatas'] if 'metadatas' in results else []
                metadata = metadatas[0][0] if metadatas and isinstance(metadatas[0], list) else (metadatas[0] if metadatas else {})
                
                # Create basic memory item
                item = MemoryItem(
                    key=request.key,
                    value=doc,
                    context=request.context,
                    metadata=metadata
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
            
            # Perform vector search
            results = await self._vector_provider.query(
                collection_name=self._config.collection_name,
                query_embeddings=[query_embedding],
                n_results=min(request.limit or self._config.max_results, self._config.max_results),
                where=where_filter if where_filter else None
            )
            
            search_results = []
            
            if results and 'documents' in results and results['documents']:
                documents = results['documents'][0] if results['documents'] else []
                metadatas = results['metadatas'][0] if 'metadatas' in results and results['metadatas'] else []
                distances = results['distances'][0] if 'distances' in results and results['distances'] else []
                
                for i, doc in enumerate(documents):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    distance = distances[i] if i < len(distances) else 1.0
                    
                    # Convert distance to similarity score (0-1 scale)
                    similarity = 1.0 - distance
                    
                    # Filter by similarity threshold
                    if similarity >= self._config.similarity_threshold:
                        # Reconstruct memory item
                        item = MemoryItem(
                            key=metadata['key'] if 'key' in metadata else 'unknown',
                            value=doc,
                            context=metadata['context'] if 'context' in metadata else (request.context or 'default'),
                            metadata=metadata
                        )
                        
                        search_results.append(MemorySearchResult(
                            item=item,
                            score=similarity,
                            metadata={
                                "search_type": "semantic",
                                "distance": distance,
                                "similarity": similarity
                            }
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
            limit=limit
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
            # Get all documents in this context
            results = await self._vector_provider.query(
                collection_name=self._config.collection_name,
                query_embeddings=None,
                n_results=10000,  # Large number to get all
                where={"context": context}
            )
            
            if results and 'ids' in results and results['ids']:
                ids_to_delete = results['ids'][0] if results['ids'] else []
                
                if ids_to_delete:
                    await self._vector_provider.delete(
                        collection_name=self._config.collection_name,
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
            embeddings = await self._embedding_provider.embed_texts([text])
            
            if not embeddings or len(embeddings) == 0:
                raise MemoryError(
                    "Embedding provider returned empty result",
                    operation="embed",
                    text_length=len(text)
                )
            
            return embeddings[0]
            
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