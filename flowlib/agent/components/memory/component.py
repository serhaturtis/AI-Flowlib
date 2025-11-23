"""
Agent Memory System.

Orchestrates interactions between different specialized memory components
(vector, knowledge graph, working memory) using modernized agent framework patterns.
"""

import asyncio
import logging
from typing import Any, cast

from pydantic import Field

from flowlib.agent.core.activity_stream import ActivityStream
from flowlib.core.models import StrictBaseModel
from flowlib.config.required_resources import RequiredAlias

# Import provider/registry for config-driven access
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.knowledge.plugin_manager import KnowledgePluginManager
from flowlib.providers.llm.base import LLMProvider, PromptTemplate
from flowlib.resources.registry.registry import resource_registry

from ...core.base import AgentComponent
from ...core.errors import MemoryError
from .knowledge import KnowledgeMemory, KnowledgeMemoryConfig
from .models import (
    MemoryItem,
    MemoryRetrieveRequest,
    MemorySearchRequest,
    MemorySearchResult,
    MemoryStoreRequest,
)
from .prompts import FusedMemoryResult

# Import modernized memory types
from .vector import VectorMemory, VectorMemoryConfig
from .working import WorkingMemory, WorkingMemoryConfig

logger = logging.getLogger(__name__)


class AgentMemoryConfig(StrictBaseModel):
    """Configuration for agent memory system."""

    working_memory: WorkingMemoryConfig = Field(
        default_factory=WorkingMemoryConfig,
        description="Configuration for working memory component",
    )
    vector_memory: VectorMemoryConfig = Field(
        default_factory=VectorMemoryConfig, description="Configuration for vector memory component"
    )
    knowledge_memory: KnowledgeMemoryConfig = Field(
        default_factory=KnowledgeMemoryConfig,
        description="Configuration for knowledge memory component",
    )
    fusion_llm_config: str = Field(
        default=RequiredAlias.DEFAULT_LLM.value, description="Provider config name for LLM fusion"
    )
    store_execution_history: bool = Field(
        default=True, description="Whether to store execution history in memory"
    )


class MemoryComponent(AgentComponent):
    """Agent memory system orchestrating specialized memory components."""

    def __init__(
        self,
        config: AgentMemoryConfig | None = None,
        activity_stream: ActivityStream | None = None,
        knowledge_plugins: KnowledgePluginManager | None = None,
    ) -> None:
        """Initialize agent memory with config-driven components."""
        super().__init__("agent_memory")
        self._config = config or AgentMemoryConfig()

        # Memory component instances
        self._vector_memory: VectorMemory | None = None
        self._knowledge_memory: KnowledgeMemory | None = None
        self._working_memory: WorkingMemory | None = None
        self._fusion_llm: LLMProvider | None = None

        # Optional components
        self._activity_stream = activity_stream
        self._knowledge_plugins = knowledge_plugins

        # State tracking
        self._contexts: set[str] = set()
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
            if (
                self._working_memory is None
                or self._vector_memory is None
                or self._knowledge_memory is None
            ):
                raise RuntimeError("Memory components not properly initialized")

            await asyncio.gather(
                self._working_memory.initialize(),
                self._vector_memory.initialize(),
                self._knowledge_memory.initialize(),
            )

            # Initialize LLM provider for fusion
            self._fusion_llm = cast(
                LLMProvider[Any],
                await provider_registry.get_by_config(self._config.fusion_llm_config),
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
                f"Comprehensive memory initialization failed: {str(e)}", cause=e
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

    async def _initialize_impl(self) -> None:
        """Implementation-specific initialization (not used - overridden initialize)."""
        pass

    async def _shutdown_impl(self) -> None:
        """Implementation-specific shutdown (not used - overridden shutdown)."""
        pass

    async def create_context(
        self, context_name: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """Create a memory context across all components."""
        if not self._initialized:
            raise MemoryError(
                "AgentMemory not initialized",
                operation="create_context",
                context_name=context_name,
            )

        logger.debug(f"Creating context: {context_name}")

        # Create context in all memory components
        if (
            self._working_memory is None
            or self._vector_memory is None
            or self._knowledge_memory is None
        ):
            raise RuntimeError("Memory components not initialized")

        tasks = [
            self._working_memory.create_context(context_name, metadata),
            self._vector_memory.create_context(context_name, metadata),
            self._knowledge_memory.create_context(context_name, metadata),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for errors
        errors = [res for res in results if isinstance(res, Exception)]
        if errors:
            logger.error(f"Errors creating context '{context_name}': {errors}")
            raise MemoryError(
                f"Failed to create context in all components: {errors[0]}"
            ) from errors[0]

        self._contexts.add(context_name)
        logger.debug(f"Context '{context_name}' created in all memory components")
        return context_name

    async def store(self, request: MemoryStoreRequest) -> str:
        """Store data across all appropriate memory components."""
        if not self._initialized:
            raise MemoryError(
                "AgentMemory not initialized",
                operation="store",
                key=request.key,
                context=request.context,
            )

        try:
            # Determine storage strategy based on content type
            from ....providers.graph.models import Entity

            is_entity = isinstance(request.value, Entity) or (
                hasattr(request.value, "model_dump")
                and isinstance(request.value.model_dump(), dict)
                and "type" in request.value.model_dump()
            )

            storage_tasks = []

            # 1. Store in Working Memory (for quick access with TTL)
            # Skip if execution history is disabled and this is execution-related content
            should_store_in_working = True
            if not self._config.store_execution_history:
                # Skip working memory for execution-related contexts
                execution_contexts = {
                    "agent_execution",
                    "execution",
                    "execution_step",
                    "execution_trace",
                }
                if request.context in execution_contexts:
                    should_store_in_working = False
                    logger.debug(
                        f"Skipping working memory for '{request.key}' (execution history disabled)"
                    )

            if (
                self._working_memory is None
                or self._vector_memory is None
                or self._knowledge_memory is None
            ):
                raise RuntimeError("Memory components not initialized")

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
                    f"Failed to store item in all memories: {errors[0]}",
                    operation="store",
                    key=request.key,
                    context=request.context,
                    cause=errors[0],
                ) from errors[0]

            # Stream storage activity
            if self._activity_stream:
                self._activity_stream.memory_store(
                    request.key,
                    request.value,
                    context=request.context,
                    importance=request.importance,
                )

            logger.debug(f"Successfully stored '{request.key}' across all memory components")
            return request.key

        except Exception as e:
            logger.error(f"Failed to store item '{request.key}': {e}")
            raise MemoryError(f"Comprehensive memory storage failed: {str(e)}", cause=e) from e

    async def store_with_model(self, request: MemoryStoreRequest) -> None:
        """Store data using structured request model.

        This method provides compatibility for code that expects the interface method.
        It delegates to the main store() method.

        Args:
            request: Memory store request with all parameters
        """
        await self.store(request)

    async def retrieve_with_model(self, request: MemoryRetrieveRequest) -> MemoryItem | None:
        """Retrieve data using structured request model.

        This method provides compatibility for code that expects the interface method.
        It delegates to the main retrieve() method.

        Args:
            request: Memory retrieve request with all parameters
        """
        return await self.retrieve(request)

    async def retrieve(self, request: MemoryRetrieveRequest) -> MemoryItem | None:
        """Retrieve by key, checking Working Memory first, then Knowledge Memory."""
        if not self._initialized:
            raise MemoryError(
                "AgentMemory not initialized",
                operation="retrieve",
                key=request.key,
                context=request.context,
            )

        # 1. Check Working Memory first (fastest access)
        logger.debug(f"Attempting to retrieve key '{request.key}' from working memory")

        if self._working_memory is None:
            raise RuntimeError("Working memory not initialized")

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

        if self._knowledge_memory is None:
            raise RuntimeError("Knowledge memory not initialized")

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

        if self._vector_memory is None:
            raise RuntimeError("Vector memory not initialized")

        try:
            item = await self._vector_memory.retrieve(request)
            if item is not None:
                logger.debug(f"Retrieved key '{request.key}' from vector memory")
                return item
        except Exception as e:
            logger.warning(f"Error retrieving from vector memory: {e}")

        logger.debug(f"Key '{request.key}' not found in any memory component")
        return None

    async def search(self, request: MemorySearchRequest) -> list[MemorySearchResult]:
        """Perform fused search across all memory components."""
        if not self._initialized:
            raise MemoryError(
                "AgentMemory not initialized",
                operation="search",
                query=request.query,
                context=request.context,
            )

        if not request.query.strip():
            logger.warning("Empty query provided")
            return []

        logger.debug(f"Performing fused search for query: '{request.query}'")

        if (
            self._working_memory is None
            or self._vector_memory is None
            or self._knowledge_memory is None
        ):
            raise RuntimeError("Memory components not initialized")

        try:
            # Execute searches across all memory components
            search_tasks = [
                self._working_memory.search(request),
                self._vector_memory.search(request),
                self._knowledge_memory.search(request),
            ]

            # Fail fast - don't capture exceptions
            results = await asyncio.gather(*search_tasks)

            # Combine results from all memory components
            combined_results = []

            for result in results:
                # Fail fast - no exception handling
                if not isinstance(result, list):
                    raise MemoryError(
                        f"Invalid search result type: expected list, got {type(result)}",
                        operation="search",
                        query=request.query,
                        context=request.context,
                        result_type=str(type(result)),
                    )
                combined_results.extend(result)

            # Sort by score if available
            combined_results.sort(
                key=lambda x: x.score if hasattr(x, "score") else 0.0, reverse=True
            )

            # Apply limit
            if request.limit:
                combined_results = combined_results[: request.limit]

            # Stream search activity
            if self._activity_stream:
                self._activity_stream.memory_retrieval(
                    request.query,
                    context=request.context,
                    search_type="fused",
                    results=[r.item.value[:100] for r in combined_results[:3]],
                )

            logger.debug(
                f"Found {len(combined_results)} total results across all memory components"
            )
            return combined_results

        except Exception as e:
            if self._activity_stream:
                self._activity_stream.error(f"Memory search failed: {str(e)}")
            raise MemoryError(
                f"Comprehensive memory search failed: {e}", operation="search", cause=e
) from e

    async def retrieve_relevant(
        self, query: str, context: str | None = None, limit: int = 10
    ) -> list[str]:
        """Retrieve relevant memories using LLM fusion across all memory components.
        
        This method performs intelligent fusion of results from:
        - Working memory (short-term, context-specific)
        - Vector memory (semantic search)
        - Knowledge memory (graph-based relationships)
        - Knowledge plugins (domain-specific databases)
        
        The fusion uses the MemoryFusionPrompt to synthesize results intelligently.
        
        Args:
            query: Search query string
            context: Optional context/namespace to search in
            limit: Maximum number of relevant items to return
            
        Returns:
            List of synthesized relevant memory items (from FusedMemoryResult)
            
        Raises:
            MemoryError: If memory system not initialized or fusion fails
            RuntimeError: If fusion LLM or prompt not available
        """
        if not self._initialized:
            raise MemoryError(
                "AgentMemory not initialized",
                operation="retrieve_relevant",
                query=query,
                context=context,
            )

        if not query.strip():
            logger.warning("Empty query provided to retrieve_relevant")
            return []

        if (
            self._working_memory is None
            or self._vector_memory is None
            or self._knowledge_memory is None
        ):
            raise RuntimeError("Memory components not initialized")

        if self._fusion_llm is None:
            raise RuntimeError(
                f"Fusion LLM not initialized: {self._config.fusion_llm_config}"
            )

        logger.debug(f"Performing fused memory retrieval for query: '{query}'")

        try:
            # Create search request for all memory components
            search_request = MemorySearchRequest(
                query=query,
                context=context,
                limit=limit,
                threshold=None,
                sort_by=None,
                search_type="hybrid",
                metadata_filter=None,
            )

            # Execute searches across all memory components concurrently
            search_tasks = [
                self._working_memory.search(search_request),
                self._vector_memory.search(search_request),
                self._knowledge_memory.search(search_request),
            ]

            # Fail fast - gather results without exception capture
            search_results = await asyncio.gather(*search_tasks)

            # Unpack results - fail fast if types are invalid
            working_results = search_results[0]
            vector_results = search_results[1]
            knowledge_results = search_results[2]

            if not isinstance(working_results, list):
                raise MemoryError(
                    f"Invalid working memory result type: expected list, got {type(working_results)}",
                    operation="retrieve_relevant",
                    query=query,
                )

            if not isinstance(vector_results, list):
                raise MemoryError(
                    f"Invalid vector memory result type: expected list, got {type(vector_results)}",
                    operation="retrieve_relevant",
                    query=query,
                )

            if not isinstance(knowledge_results, list):
                raise MemoryError(
                    f"Invalid knowledge memory result type: expected list, got {type(knowledge_results)}",
                    operation="retrieve_relevant",
                    query=query,
                )

            # Format results from each memory type
            working_results_text = self._format_working_memory_results(working_results)
            vector_results_text = self._format_vector_memory_results(vector_results)
            knowledge_results_text = self._format_knowledge_memory_results(knowledge_results)

            # Query knowledge plugins if available
            plugin_results_text = await self._query_knowledge_plugins(query, context)

            # Get fusion prompt from resource registry
            try:
                fusion_prompt_resource = resource_registry.get("memory_fusion")
                if not isinstance(fusion_prompt_resource, PromptTemplate):
                    raise TypeError(
                        f"memory_fusion prompt must be PromptTemplate, got {type(fusion_prompt_resource)}"
                    )
            except KeyError as e:
                raise RuntimeError(
                    "memory_fusion prompt not found in resource registry"
                ) from e

            # Perform LLM fusion
            fused_result = await self._fusion_llm.generate_structured(
                prompt=fusion_prompt_resource,
                output_type=FusedMemoryResult,
                model_name=RequiredAlias.DEFAULT_MODEL.value,
                prompt_variables={
                    "query": query,
                    "working_results": working_results_text,
                    "vector_results": vector_results_text,
                    "knowledge_results": knowledge_results_text,
                    "plugin_results": plugin_results_text,
                },
            )

            # Return synthesized relevant items
            logger.debug(
                f"Fusion completed: {len(fused_result.relevant_items)} items, summary: {fused_result.summary[:100]}"
            )

            if self._activity_stream:
                self._activity_stream.memory_retrieval(
                    query,
                    context=context,
                    search_type="fused_llm",
                    results=fused_result.relevant_items[:5],  # Top 5 items
                )

            return fused_result.relevant_items

        except Exception as e:
            if self._activity_stream:
                self._activity_stream.error(f"Memory fusion failed: {str(e)}")
            raise MemoryError(
                f"Memory fusion retrieval failed: {e}",
                operation="retrieve_relevant",
                query=query,
                context=context,
                cause=e,
            ) from e

    def _format_working_memory_results(self, results: list[MemorySearchResult]) -> str:
        """Format working memory search results for fusion prompt.
        
        Args:
            results: List of memory search results from working memory
            
        Returns:
            Formatted string representation of working memory results
        """
        if not results:
            return "No relevant working memory found."

        formatted_items = []
        for result in results:
            item = result.item
            value_str = str(item.value) if hasattr(item, "value") else str(item)
            # Truncate long values for prompt efficiency
            if len(value_str) > 200:
                value_str = value_str[:200] + "..."
            formatted_items.append(f"- {item.key}: {value_str} (score: {result.score:.2f})")

        return "\n".join(formatted_items)

    def _format_vector_memory_results(self, results: list[MemorySearchResult]) -> str:
        """Format vector memory search results for fusion prompt.
        
        Args:
            results: List of memory search results from vector memory
            
        Returns:
            Formatted string representation of vector memory results
        """
        if not results:
            return "No relevant semantic search results found."

        formatted_items = []
        for result in results:
            item = result.item
            value_str = str(item.value) if hasattr(item, "value") else str(item)
            # Truncate long values for prompt efficiency
            if len(value_str) > 200:
                value_str = value_str[:200] + "..."
            formatted_items.append(
                f"- {item.key}: {value_str} (similarity: {result.score:.2f})"
            )

        return "\n".join(formatted_items)

    def _format_knowledge_memory_results(self, results: list[MemorySearchResult]) -> str:
        """Format knowledge memory search results for fusion prompt.
        
        Args:
            results: List of memory search results from knowledge memory
            
        Returns:
            Formatted string representation of knowledge memory results
        """
        if not results:
            return "No relevant knowledge graph results found."

        formatted_items = []
        for result in results:
            item = result.item
            value_str = str(item.value) if hasattr(item, "value") else str(item)
            # Truncate long values for prompt efficiency
            if len(value_str) > 200:
                value_str = value_str[:200] + "..."
            formatted_items.append(
                f"- {item.key}: {value_str} (relevance: {result.score:.2f})"
            )

        return "\n".join(formatted_items)

    async def _query_knowledge_plugins(self, query: str, context: str | None) -> str:
        """Query knowledge plugins and format results for fusion prompt.
        
        Args:
            query: Search query string
            context: Optional context/domain to search in
            
        Returns:
            Formatted string representation of plugin results
        """
        if not self._knowledge_plugins:
            return "No knowledge plugins available."

        if not self._knowledge_plugins._initialized:
            await self._knowledge_plugins.initialize()

        # Use context as domain if available, otherwise try common domains
        domain = context or "general"

        try:
            plugin_results = await self._knowledge_plugins.query_domain(
                domain=domain, query=query, limit=10
            )

            if not plugin_results:
                return f"No relevant knowledge plugin results found for domain '{domain}'."

            formatted_items = []
            for knowledge in plugin_results:
                content_str = str(knowledge.content)
                # Truncate long content for prompt efficiency
                if len(content_str) > 200:
                    content_str = content_str[:200] + "..."
                confidence = knowledge.confidence if hasattr(knowledge, "confidence") else 1.0
                formatted_items.append(f"- {content_str} (confidence: {confidence:.2f})")

            return "\n".join(formatted_items)

        except Exception as e:
            logger.warning(f"Knowledge plugin query failed: {e}")
            return f"Knowledge plugin query failed: {str(e)}"

    async def wipe_context(self, context: str) -> None:
        """Wipe a specific context across all memory components."""
        if not self._initialized:
            raise MemoryError("AgentMemory not initialized")

        logger.warning(f"Wiping memory context '{context}' across all components...")

        if (
            self._working_memory is None
            or self._vector_memory is None
            or self._knowledge_memory is None
        ):
            raise RuntimeError("Memory components not initialized")

        wipe_tasks = [
            self._working_memory.wipe_context(context),
            self._vector_memory.wipe_context(context),
            self._knowledge_memory.wipe_context(context),
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

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {
            "initialized": self._initialized,
            "context_count": len(self._contexts),
            "contexts": list(self._contexts),
            "config": self._config.model_dump(),
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
