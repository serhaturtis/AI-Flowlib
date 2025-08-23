"""
Agent Memory System.

Orchestrates interactions between different specialized memory components
(vector, knowledge graph, working memory) using modernized agent framework patterns.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional

from pydantic import Field
from flowlib.core.models import StrictBaseModel

from ...core.errors import MemoryError, ErrorContext
from .interfaces import MemoryInterface
from ...core.base import AgentComponent
from .models import (
    MemoryItem,
    MemoryStoreRequest,
    MemoryRetrieveRequest,
    MemorySearchRequest,
    MemorySearchResult,
    MemoryContext
)

# Import modernized memory types
from .vector import VectorMemory, VectorMemoryConfig
from .knowledge import KnowledgeMemory, KnowledgeMemoryConfig
from .working import WorkingMemory, WorkingMemoryConfig

# Import provider/registry for config-driven access
from flowlib.providers.core.registry import provider_registry

logger = logging.getLogger(__name__)


class AgentMemoryConfig(StrictBaseModel):
    """Configuration for agent memory system."""
    
    working_memory: WorkingMemoryConfig = Field(
        default_factory=WorkingMemoryConfig,
        description="Configuration for working memory component"
    )
    vector_memory: VectorMemoryConfig = Field(
        default_factory=VectorMemoryConfig,
        description="Configuration for vector memory component"
    )
    knowledge_memory: KnowledgeMemoryConfig = Field(
        default_factory=KnowledgeMemoryConfig,
        description="Configuration for knowledge memory component"
    )
    fusion_llm_config: str = Field(
        default="default-llm",
        description="Provider config name for LLM fusion"
    )
    store_execution_history: bool = Field(
        default=True,
        description="Whether to store execution history in memory"
    )


class AgentMemory(AgentComponent, MemoryInterface):
    """Agent memory system orchestrating specialized memory components."""

    def __init__(
        self,
        config: Optional[AgentMemoryConfig] = None,
        activity_stream=None,
        knowledge_plugins=None
    ):
        """Initialize agent memory with config-driven components."""
        super().__init__("agent_memory")
        self._config = config or AgentMemoryConfig()
        
        # Memory component instances
        self._vector_memory = None
        self._knowledge_memory = None
        self._working_memory = None
        self._fusion_llm = None
        
        # Optional components
        self._activity_stream = activity_stream
        self._knowledge_plugins = knowledge_plugins
        
        # State tracking
        self._contexts = set()
        self._initialized = False
        
        logger.info(f"Initialized AgentMemory with config: {self._config}")

    @property
    def initialized(self) -> bool:
        """Check if memory is initialized."""
        return self._initialized
    
    async def initialize(self) -> None:
        """Initialize agent memory and all components."""
        if self._initialized:
            return
            
        logger.info("Initializing AgentMemory...")
        
        try:
            # Create memory component instances
            self._working_memory = WorkingMemory(self._config.working_memory)
            self._vector_memory = VectorMemory(self._config.vector_memory)
            self._knowledge_memory = KnowledgeMemory(self._config.knowledge_memory)
            
            # Initialize all memory components
            await asyncio.gather(
                self._working_memory.initialize(),
                self._vector_memory.initialize(),
                self._knowledge_memory.initialize()
            )
            
            # Initialize LLM provider for fusion
            self._fusion_llm = await provider_registry.get_by_config(
                self._config.fusion_llm_config
            )
            if not self._fusion_llm:
                raise MemoryError(
                    f"Fusion LLM provider not found: {self._config.fusion_llm_config}"
                )
            
            self._initialized = True
            logger.info("AgentMemory initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AgentMemory: {e}")
            raise MemoryError(
                f"Comprehensive memory initialization failed: {str(e)}",
                cause=e
            ) from e

    async def shutdown(self) -> None:
        """Shutdown comprehensive memory and all components."""
        if not self._initialized:
            return
            
        logger.info("Shutting down AgentMemory...")
        
        # Shutdown all memory components
        shutdown_tasks = []
        if self._working_memory:
            shutdown_tasks.append(self._working_memory.shutdown())
        if self._vector_memory:
            shutdown_tasks.append(self._vector_memory.shutdown())
        if self._knowledge_memory:
            shutdown_tasks.append(self._knowledge_memory.shutdown())
            
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        # Clean up references
        self._fusion_llm = None
        self._working_memory = None
        self._vector_memory = None
        self._knowledge_memory = None
        self._contexts.clear()
        
        self._initialized = False
        logger.info("AgentMemory shutdown completed")

    async def create_context(self, context_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a memory context across all components."""
        if not self._initialized:
            raise MemoryError("AgentMemory not initialized")
            
        logger.debug(f"Creating context: {context_name}")
        
        # Create context in all memory components
        tasks = [
            self._working_memory.create_context(context_name, metadata),
            self._vector_memory.create_context(context_name, metadata),
            self._knowledge_memory.create_context(context_name, metadata)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for errors
        errors = [res for res in results if isinstance(res, Exception)]
        if errors:
            logger.error(f"Errors creating context '{context_name}': {errors}")
            raise MemoryError(f"Failed to create context in all components: {errors[0]}") from errors[0]
        
        self._contexts.add(context_name)
        logger.debug(f"Context '{context_name}' created in all memory components")
        return context_name
    
    async def store(self, request: MemoryStoreRequest) -> str:
        """Store data across all appropriate memory components."""
        if not self._initialized:
            raise MemoryError("AgentMemory not initialized")
            
        try:
            # Determine storage strategy based on content type
            from ....providers.graph.models import Entity
            is_entity = isinstance(request.value, Entity) or (
                hasattr(request.value, 'model_dump') and 
                isinstance(request.value.model_dump(), dict) and 
                'type' in request.value.model_dump()
            )

            storage_tasks = []

            # 1. Store in Working Memory (for quick access with TTL)
            # Skip if execution history is disabled and this is execution-related content
            should_store_in_working = True
            if not self._config.store_execution_history:
                # Skip working memory for execution-related contexts
                execution_contexts = {"agent_execution", "execution", "execution_step"}
                if (request.context in execution_contexts or 
                    (request.metadata and request.metadata.get("type") == "execution_step")):
                    should_store_in_working = False
                    logger.debug(f"Skipping working memory for '{request.key}' (execution history disabled)")
            
            if should_store_in_working:
                logger.debug(f"Storing '{request.key}' in working memory")
                storage_tasks.append(self._working_memory.store(request))
            
            # 2. Store entities in Knowledge Base
            if is_entity:
                logger.debug(f"Storing '{request.key}' as entity in knowledge memory")
                storage_tasks.append(self._knowledge_memory.store(request))

            # 3. Store in Vector Memory for semantic search
            logger.debug(f"Storing '{request.key}' in vector memory for semantic search")
            storage_tasks.append(self._vector_memory.store(request))
                
            # Execute storage operations concurrently
            results = await asyncio.gather(*storage_tasks, return_exceptions=True)
            
            # Check for errors and successes
            errors = [res for res in results if isinstance(res, Exception)]
            successes = [res for res in results if not isinstance(res, Exception)]
            
            if errors:
                logger.warning(f"Partial storage failure for key '{request.key}': {errors}")
                
            # Fail only if ALL components failed
            if errors and len(successes) == 0:
                logger.error(f"Complete storage failure for key '{request.key}': {errors}")
                raise MemoryError(
                    f"Failed to store item in all memories: {errors[0]}"
                ) from errors[0]
            
            # Stream storage activity
            if self._activity_stream:
                self._activity_stream.memory_store(
                    request.key,
                    request.value,
                    context=request.context,
                    importance=request.importance
                )
            
            logger.debug(f"Successfully stored '{request.key}' across all memory components")
            return request.key
            
        except Exception as e:
            logger.error(f"Failed to store item '{request.key}': {e}")
            raise MemoryError(
                f"Comprehensive memory storage failed: {str(e)}",
                cause=e
            ) from e

    async def retrieve(self, request: MemoryRetrieveRequest) -> Optional[MemoryItem]:
        """Retrieve by key, checking Working Memory first, then Knowledge Memory."""
        if not self._initialized:
            raise MemoryError("AgentMemory not initialized")
            
        # 1. Check Working Memory first (fastest access)
        logger.debug(f"Attempting to retrieve key '{request.key}' from working memory")
        try:
            item = await self._working_memory.retrieve(request)
            if item is not None: 
                logger.debug(f"Retrieved key '{request.key}' from working memory")
                return item
        except Exception as e:
            logger.warning(f"Error retrieving from working memory (key: {request.key}): {e}")
            # Continue to knowledge memory

        # 2. Check Knowledge Memory (for entities and structured data)
        logger.debug(f"Attempting to retrieve key '{request.key}' from knowledge memory")
        try:
            item = await self._knowledge_memory.retrieve(request)
            if item is not None:
                logger.debug(f"Retrieved key '{request.key}' from knowledge memory")
                return item
        except Exception as e:
            logger.warning(f"Error retrieving from knowledge memory: {e}")
            # Continue to vector memory
            
        # 3. Check Vector Memory (last resort for key-based retrieval)
        logger.debug(f"Attempting to retrieve key '{request.key}' from vector memory")
        try:
            item = await self._vector_memory.retrieve(request)
            if item is not None:
                logger.debug(f"Retrieved key '{request.key}' from vector memory")
                return item
        except Exception as e:
            logger.warning(f"Error retrieving from vector memory: {e}")
        
        logger.debug(f"Key '{request.key}' not found in any memory component")
        return None

    async def search(self, request: MemorySearchRequest) -> List[MemorySearchResult]:
        """Perform fused search across all memory components."""
        if not self._initialized:
            raise MemoryError("AgentMemory not initialized")

        if not request.query.strip():
            logger.warning("Empty query provided")
            return []

        logger.debug(f"Performing fused search for query: '{request.query}'")
        
        try:
            # Execute searches across all memory components
            search_tasks = [
                self._working_memory.search(request),
                self._vector_memory.search(request),
                self._knowledge_memory.search(request)
            ]
            
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine results from all memory components
            combined_results = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Search failed in memory component {i}: {result}")
                    continue
                    
                if isinstance(result, list):
                    combined_results.extend(result)
            
            # Sort by score if available
            combined_results.sort(
                key=lambda x: x.score if hasattr(x, 'score') else 0.0, 
                reverse=True
            )
            
            # Apply limit
            if request.limit:
                combined_results = combined_results[:request.limit]
            
            # Stream search activity
            if self._activity_stream:
                self._activity_stream.memory_retrieval(
                    request.query,
                    context=request.context,
                    search_type="fused",
                    results=[r.item.value[:100] for r in combined_results[:3]]
                )
            
            logger.debug(f"Found {len(combined_results)} total results across all memory components")
            return combined_results
            
        except Exception as e:
            if self._activity_stream:
                self._activity_stream.error(f"Memory search failed: {str(e)}")
            raise MemoryError(f"Comprehensive memory search failed: {e}", operation="search", cause=e)

    async def retrieve_relevant(
        self, 
        query: str, 
        context: Optional[str] = None, 
        limit: int = 10
    ) -> List[str]:
        """Retrieve relevant memories using fused search across all components."""
        if not self._initialized:
            raise MemoryError("AgentMemory not initialized")
            
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
        """Wipe a specific context across all memory components."""
        if not self._initialized:
            raise MemoryError("AgentMemory not initialized")
            
        logger.warning(f"Wiping memory context '{context}' across all components...")
        
        wipe_tasks = [
            self._working_memory.wipe_context(context),
            self._vector_memory.wipe_context(context),
            self._knowledge_memory.wipe_context(context)
        ]
        
        results = await asyncio.gather(*wipe_tasks, return_exceptions=True)
        errors = [res for res in results if isinstance(res, Exception)]
        
        if errors:
            for i, err in enumerate(errors):
                logger.error(f"Error wiping context '{context}' in component {i}: {err}")
            raise MemoryError(
                f"Failed to wipe context '{context}' in one or more components: {errors[0]}"
            ) from errors[0]
            
        self._contexts.discard(context)
        logger.info(f"Context '{context}' wiped from all memory components")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {
            "initialized": self._initialized,
            "context_count": len(self._contexts),
            "contexts": list(self._contexts),
            "config": self._config.model_dump()
        }
        
        # Add component stats if initialized
        if self._initialized:
            if self._working_memory:
                stats["working_memory"] = self._working_memory.get_stats()
            if self._vector_memory:
                stats["vector_memory"] = self._vector_memory.get_stats()
            if self._knowledge_memory:
                stats["knowledge_memory"] = self._knowledge_memory.get_stats()
                
        return stats