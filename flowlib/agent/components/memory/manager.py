"""
Agent memory management component.

This module handles agent memory operations and coordination
that were previously in BaseAgent.
"""

import logging
from typing import Any, cast

from flowlib.agent.core.activity_stream import ActivityStream
from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.errors import NotInitializedError
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.knowledge.plugin_manager import KnowledgePluginManager

from .component import AgentMemoryConfig, MemoryComponent
from .knowledge import KnowledgeMemoryConfig
from .models import MemoryRetrieveRequest, MemorySearchRequest, MemoryStoreRequest
from .vector import VectorMemoryConfig
from .working import WorkingMemoryConfig

logger = logging.getLogger(__name__)


class AgentMemoryManager(AgentComponent):
    """Handles agent memory operations.

    This component is responsible for:
    - Memory system initialization and coordination
    - Memory storage, retrieval, and search operations
    - Memory statistics and management
    """

    def __init__(self, name: str = "memory_manager"):
        """Initialize the memory manager.

        Args:
            name: Component name
        """
        super().__init__(name)
        self._memory: MemoryComponent | None = None

    async def _initialize_impl(self) -> None:
        """Initialize the memory manager."""
        logger.info("Memory manager initialized")

    async def _shutdown_impl(self) -> None:
        """Shutdown the memory manager."""
        if self._memory and self._memory.initialized:
            await self._memory.shutdown()
        logger.info("Memory manager shutdown")

    async def setup_memory(self, memory_config: AgentMemoryConfig) -> MemoryComponent:
        """Setup memory system from configuration.

        Args:
            memory_config: Memory configuration

        Returns:
            Initialized MemoryComponent instance
        """
        if self._memory:
            return self._memory

        mem_config = memory_config

        # Create providers using config-driven approach
        # Fail fast - all providers are required
        embedding_provider = await provider_registry.get_by_config("default-embedding")
        if not embedding_provider:
            raise MemoryError("Embedding provider 'default-embedding' is required but not found")

        vector_provider = await provider_registry.get_by_config("default-vector-db")
        if not vector_provider:
            raise MemoryError("Vector provider 'default-vector-db' is required but not found")

        graph_provider = await provider_registry.get_by_config("default-graph-db")
        if not graph_provider:
            raise MemoryError("Graph provider 'default-graph-db' is required but not found")

        # Create comprehensive AgentMemoryConfig
        working_mem_config = WorkingMemoryConfig(
            default_ttl_seconds=mem_config.working_memory.default_ttl_seconds
        )

        # All providers are required - no conditional creation
        vector_mem_config = VectorMemoryConfig(
            vector_provider_config="default-vector-db",
            embedding_provider_config="default-embedding",
        )
        knowledge_mem_config = KnowledgeMemoryConfig(graph_provider_config="default-graph-db")

        agent_memory_config = AgentMemoryConfig(
            working_memory=working_mem_config,
            vector_memory=vector_mem_config,
            knowledge_memory=knowledge_mem_config,
            fusion_llm_config=mem_config.fusion_llm_config,
        )

        # Instantiate comprehensive memory
        self._memory = MemoryComponent(
            config=agent_memory_config,
            activity_stream=cast(ActivityStream | None, self.get_component("activity_stream")),
            knowledge_plugins=cast(
                KnowledgePluginManager | None, self.get_component("knowledge_plugins")
            ),
        )
        # AgentMemory no longer uses parent relationships

        # Initialize memory
        await self._memory.initialize()

        # Create required memory contexts
        try:
            await self._memory.create_context(
                context_name="agent_execution",
                metadata={"builtin": True, "description": "Context for agent execution cycles"},
            )
            logger.debug("Created agent_execution memory context")
        except Exception as e:
            logger.debug(f"Could not create agent_execution context: {e}")

        return self._memory

    async def store_memory(self, key: str, value: Any, **kwargs: Any) -> None:
        """Store a value in memory.

        Args:
            key: Memory key
            value: Value to store
            **kwargs: Additional arguments for memory storage

        Raises:
            NotInitializedError: If memory is not initialized
        """
        if not self._memory or not self._memory.initialized:
            raise NotInitializedError(component_name=self._name, operation="store_memory")

        # Extract valid MemoryStoreRequest parameters from kwargs
        valid_params = {}
        for k, v in kwargs.items():
            if k in ["context", "ttl", "metadata", "importance"]:
                valid_params[k] = v

        request = MemoryStoreRequest(key=key, value=value, **valid_params)

        await self._memory.store_with_model(request)

    async def retrieve_memory(self, key: str, **kwargs: Any) -> Any:
        """Retrieve a value from memory.

        Args:
            key: Memory key
            **kwargs: Additional arguments for memory retrieval

        Returns:
            Retrieved value

        Raises:
            NotInitializedError: If memory is not initialized
        """
        if not self._memory or not self._memory.initialized:
            raise NotInitializedError(component_name=self._name, operation="retrieve_memory")

        # Extract valid MemoryRetrieveRequest parameters from kwargs
        valid_params = {}
        for k, v in kwargs.items():
            if k in ["context", "default", "metadata_only"]:
                valid_params[k] = v

        request = MemoryRetrieveRequest(key=key, **valid_params)

        return await self._memory.retrieve_with_model(request)

    async def search_memory(self, query: str, **kwargs: Any) -> list[Any]:
        """Search memory for relevant information.

        Args:
            query: Search query
            **kwargs: Additional arguments for memory search

        Returns:
            List of relevant memories

        Raises:
            NotInitializedError: If memory is not initialized
        """
        if not self._memory or not self._memory.initialized:
            raise NotInitializedError(component_name=self._name, operation="search_memory")

        # Extract valid MemorySearchRequest parameters from kwargs
        valid_params = {}
        for k, v in kwargs.items():
            if k in ["context", "limit", "threshold", "sort_by", "search_type", "metadata_filter"]:
                valid_params[k] = v

        request = MemorySearchRequest(query=query, **valid_params)

        result = await self._memory.search(request)
        return result

    async def get_memory_stats(self) -> dict[str, Any]:
        """Get memory system statistics.

        Returns:
            Dictionary of memory statistics
        """
        if not self._memory:
            return {"error": "Memory system not initialized"}

        stats = {
            "working_memory_items": 0,
            "vector_memory_items": 0,
            "knowledge_graph_nodes": 0,
            "knowledge_graph_edges": 0,
            "total_storage_mb": 0,
        }

        # Get working memory stats
        if hasattr(self._memory, "working_memory"):
            wm_items = await self._memory.working_memory.get_recent(1000)
            stats["working_memory_items"] = len(wm_items)

        # Get vector memory stats
        if hasattr(self._memory, "vector_memory"):
            vm_stats = await self._memory.vector_memory.get_stats()
            stats["vector_memory_items"] = vm_stats["count"] if "count" in vm_stats else 0

        # Get knowledge graph stats
        if hasattr(self._memory, "knowledge_graph"):
            kg_stats = await self._memory.knowledge_graph.get_stats()
            stats["knowledge_graph_nodes"] = kg_stats["nodes"] if "nodes" in kg_stats else 0
            stats["knowledge_graph_edges"] = kg_stats["edges"] if "edges" in kg_stats else 0

        return stats

    @property
    def memory(self) -> MemoryComponent | None:
        """Get the memory instance.

        Returns:
            MemoryComponent instance
        """
        return self._memory
