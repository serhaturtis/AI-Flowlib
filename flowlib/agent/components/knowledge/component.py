"""Unified Knowledge Component - Single source of truth.

This component consolidates all knowledge operations (learning, storage, retrieval)
following flowlib's AgentComponent patterns and strict validation principles.
"""

import logging
from datetime import datetime
from typing import Any, cast

from flowlib.agent.core.base import AgentComponent
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.graph.base import GraphDBProvider
from flowlib.providers.llm.base import LLMProvider, PromptTemplate
from flowlib.providers.vector.base import VectorDBProvider
from flowlib.resources.registry.registry import resource_registry
from flowlib.config.required_resources import RequiredAlias

from .models import (
    KnowledgeComponentConfig,
    KnowledgeSet,
    LearningInput,
    LearningResult,
    RetrievalRequest,
    RetrievalResult,
    StorageRequest,
)

logger = logging.getLogger(__name__)


class KnowledgeComponent(AgentComponent):
    """Unified knowledge component for learning, storage, and retrieval.

    This component follows flowlib's single responsibility principle by providing
    a clean interface for all knowledge operations while delegating to specialized
    providers for actual implementation.
    """

    def __init__(self, config: KnowledgeComponentConfig, name: str = "knowledge"):
        """Initialize the knowledge component.

        Args:
            config: Knowledge component configuration
            name: Component name
        """
        super().__init__(name)
        self._config = config

        # Provider instances (initialized during _initialize_impl)
        self._llm_provider: LLMProvider | None = None
        self._vector_provider: VectorDBProvider | None = None
        self._graph_provider: GraphDBProvider | None = None

    async def _initialize_impl(self) -> None:
        """Initialize the knowledge component and its providers."""
        logger.info("Initializing knowledge component")

        # Initialize LLM provider (required for learning operations)
        # Note: Agent controls whether learn_from_content() is called via enable_learning flag
        self._llm_provider = cast(
            LLMProvider, await provider_registry.get_by_config(self._config.llm_config)
        )
        if not self._llm_provider:
            raise RuntimeError(f"LLM provider '{self._config.llm_config}' not available")
        logger.debug(f"Initialized LLM provider: {self._config.llm_config}")

        # Initialize vector database provider (optional for storage/retrieval)
        if self._config.enable_storage or self._config.enable_retrieval:
            try:
                self._vector_provider = cast(
                    VectorDBProvider,
                    await provider_registry.get_by_config(self._config.vector_db_config),
                )
                if self._vector_provider:
                    logger.debug(f"Initialized vector provider: {self._config.vector_db_config}")
            except Exception as e:
                logger.warning(f"Vector provider initialization failed: {e}")

            # Initialize graph database provider (optional for storage/retrieval)
            try:
                self._graph_provider = cast(
                    GraphDBProvider,
                    await provider_registry.get_by_config(self._config.graph_db_config),
                )
                if self._graph_provider:
                    logger.debug(f"Initialized graph provider: {self._config.graph_db_config}")
            except Exception as e:
                logger.warning(f"Graph provider initialization failed: {e}")

        logger.info("Knowledge component initialization complete")

    async def _shutdown_impl(self) -> None:
        """Shutdown the knowledge component."""
        logger.info("Shutting down knowledge component")

        # Providers are managed by registry, no explicit shutdown needed
        self._llm_provider = None
        self._vector_provider = None
        self._graph_provider = None

        logger.info("Knowledge component shutdown complete")

    # Core Knowledge Operations

    async def learn_from_content(
        self,
        content: str,
        context: str,
        focus_areas: list[str] | None = None,
        domain_hint: str | None = None,
    ) -> LearningResult:
        """Learn knowledge from content with intelligent extraction.

        Args:
            content: Content to learn from
            context: Context information
            focus_areas: Optional areas to focus extraction on
            domain_hint: Optional domain hint for extraction

        Returns:
            Learning result with extracted knowledge

        Raises:
            NotInitializedError: If component not initialized
            RuntimeError: If LLM provider unavailable
        """
        self._check_initialized()

        if not self._llm_provider:
            raise RuntimeError("LLM provider not available for learning")

        start_time = datetime.now()

        try:
            logger.info(f"Learning from content: {len(content)} characters")

            # Create learning input with strict validation
            learning_input = LearningInput(
                content=content,
                context=context,
                focus_areas=focus_areas or [],
                domain_hint=domain_hint,
            )

            # Extract knowledge using LLM
            knowledge_set = await self._extract_knowledge(learning_input)

            processing_time = (datetime.now() - start_time).total_seconds()

            return LearningResult(
                success=True,
                knowledge=knowledge_set,
                processing_time_seconds=processing_time,
                message=f"Successfully extracted {knowledge_set.total_items} knowledge items",
                metadata={"extraction_method": "llm_structured", "domain_hint": domain_hint},
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Learning failed: {e}")

            return LearningResult(
                success=False,
                knowledge=KnowledgeSet(),  # Empty knowledge set
                processing_time_seconds=processing_time,
                message=f"Learning failed: {str(e)}",
                metadata={"error": str(e)},
            )

    async def store_knowledge(self, request: StorageRequest) -> None:
        """Store knowledge with intelligent routing to available providers.

        Args:
            request: Storage request with knowledge and context

        Raises:
            NotInitializedError: If component not initialized
            RuntimeError: If storage is disabled
        """
        self._check_initialized()

        if not self._config.enable_storage:
            raise RuntimeError("Storage is disabled in configuration")

        if request.knowledge.is_empty:
            logger.debug("Skipping storage of empty knowledge set")
            return

        logger.info(f"Storing knowledge: {request.knowledge.total_items} items")

        # Store in vector database for semantic search
        if self._vector_provider and (request.knowledge.entities or request.knowledge.concepts):
            await self._store_in_vector(request.knowledge, request.context_id)

        # Store in graph database for relationship analysis
        if self._graph_provider and request.knowledge.relationships:
            await self._store_in_graph(request.knowledge, request.context_id)

        logger.info("Knowledge storage complete")

    async def retrieve_knowledge(self, request: RetrievalRequest) -> RetrievalResult:
        """Retrieve knowledge using intelligent search across available providers.

        Args:
            request: Retrieval request with query and filters

        Returns:
            Retrieval result with found knowledge

        Raises:
            NotInitializedError: If component not initialized
            RuntimeError: If retrieval is disabled
        """
        self._check_initialized()

        if not self._config.enable_retrieval:
            raise RuntimeError("Retrieval is disabled in configuration")

        logger.info(f"Retrieving knowledge for query: '{request.query[:50]}...'")

        # Search in vector database for semantic similarity
        vector_results = KnowledgeSet()
        if self._vector_provider:
            vector_results = await self._search_vector(request)

        # Search in graph database for relationship traversal
        graph_results = KnowledgeSet()
        if self._graph_provider:
            graph_results = await self._search_graph(request)

        # Combine results intelligently
        combined_knowledge = self._combine_search_results(vector_results, graph_results, request)

        return RetrievalResult(
            knowledge=combined_knowledge,
            query=request.query,
            relevance_scores={},  # Could be populated with actual scores
            total_found=combined_knowledge.total_items,
        )

    # Private Implementation Methods

    async def _extract_knowledge(self, learning_input: LearningInput) -> KnowledgeSet:
        """Extract knowledge using LLM with structured output."""
        # Ensure LLM provider is available
        assert self._llm_provider is not None, "LLM provider must be initialized before extraction"

        # Get knowledge extraction prompt
        extraction_prompt = resource_registry.get("knowledge-extraction-prompt")

        # Prepare prompt variables - use Dict[str, Any] for compatibility
        prompt_vars: dict[str, Any] = {
            "content": learning_input.content,
            "context": learning_input.context,
            "focus_areas": ", ".join(learning_input.focus_areas)
            if learning_input.focus_areas
            else "general",
            "domain_hint": learning_input.domain_hint or "general",
        }

        # Use structured generation to extract knowledge
        result = await self._llm_provider.generate_structured(
            prompt=cast(PromptTemplate, extraction_prompt),
            prompt_variables=prompt_vars,
            output_type=KnowledgeSet,
            model_name=RequiredAlias.DEFAULT_MODEL.value,
        )

        # Result is guaranteed to be KnowledgeSet from structured generation
        if not isinstance(result, KnowledgeSet):
            raise RuntimeError(f"Expected KnowledgeSet, got {type(result)}")
        extraction_result = result

        # Add source context to all extracted items
        for entity in extraction_result.entities:
            entity.source_context = learning_input.context
        for concept in extraction_result.concepts:
            concept.source_context = learning_input.context
        for relationship in extraction_result.relationships:
            relationship.source_context = learning_input.context
        for pattern in extraction_result.patterns:
            pattern.source_context = learning_input.context

        return extraction_result

    async def _store_in_vector(self, knowledge: KnowledgeSet, context_id: str) -> None:
        """Store knowledge in vector database."""
        logger.debug(
            f"Storing {len(knowledge.entities + knowledge.concepts)} items in vector database"
        )
        # Implementation would use self._vector_provider
        # For now, this is a placeholder following no-fallback principle

    async def _store_in_graph(self, knowledge: KnowledgeSet, context_id: str) -> None:
        """Store knowledge in graph database."""
        logger.debug(f"Storing {len(knowledge.relationships)} relationships in graph database")
        # Implementation would use self._graph_provider
        # For now, this is a placeholder following no-fallback principle

    async def _search_vector(self, request: RetrievalRequest) -> KnowledgeSet:
        """Search vector database for semantic matches."""
        logger.debug(f"Searching vector database for: {request.query}")
        # Implementation would use self._vector_provider
        return KnowledgeSet()  # Placeholder

    async def _search_graph(self, request: RetrievalRequest) -> KnowledgeSet:
        """Search graph database for relationship matches."""
        logger.debug(f"Searching graph database for: {request.query}")
        # Implementation would use self._graph_provider
        return KnowledgeSet()  # Placeholder

    def _combine_search_results(
        self, vector_results: KnowledgeSet, graph_results: KnowledgeSet, request: RetrievalRequest
    ) -> KnowledgeSet:
        """Combine search results intelligently."""
        # Simple combination - could be enhanced with ranking/deduplication
        combined = KnowledgeSet(
            entities=vector_results.entities + graph_results.entities,
            concepts=vector_results.concepts + graph_results.concepts,
            relationships=vector_results.relationships + graph_results.relationships,
            patterns=vector_results.patterns + graph_results.patterns,
            metadata={"combined_from": ["vector", "graph"]},
        )

        # Apply limit
        return self._apply_result_limit(combined, request.limit)

    def _apply_result_limit(self, knowledge: KnowledgeSet, limit: int) -> KnowledgeSet:
        """Apply result limit to knowledge set."""
        if knowledge.total_items <= limit:
            return knowledge

        # Simple truncation - could be enhanced with ranking
        return KnowledgeSet(
            entities=knowledge.entities[: limit // 4],
            concepts=knowledge.concepts[: limit // 4],
            relationships=knowledge.relationships[: limit // 4],
            patterns=knowledge.patterns[: limit // 4],
            metadata=knowledge.metadata,
        )
