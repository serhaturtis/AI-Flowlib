"""
Modern Knowledge Memory Component.

Provides structured knowledge storage using graph databases with semantic
relationships. Uses the modernized agent framework patterns with config-driven providers.
"""

import logging
from datetime import datetime
from typing import Any, cast

from pydantic import Field

from flowlib.core.models import StrictBaseModel
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.graph.base import GraphDBProvider

from ....providers.graph.models import Entity, EntityAttribute, EntityRelationship
from ...core.errors import MemoryError

# Import MemoryItem and MemorySearchResult from the proper location
from .models import (
    MemoryItem,
    MemoryRetrieveRequest,
    MemorySearchRequest,
    MemorySearchResult,
    MemoryStoreRequest,
)

logger = logging.getLogger(__name__)


class KnowledgeMemoryConfig(StrictBaseModel):
    """Configuration for knowledge memory."""

    graph_provider_config: str = Field(
        default="default-graph-db", description="Provider config name for graph database"
    )
    default_importance: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Default importance score for new entities"
    )
    max_search_results: int = Field(
        default=50, ge=1, description="Maximum results to return from knowledge search"
    )
    default_context: str = Field(
        default="knowledge", description="Default context for knowledge storage"
    )


class KnowledgeMemory:
    """Modern knowledge memory implementation with graph-based storage."""

    def __init__(self, config: KnowledgeMemoryConfig | None = None):
        """Initialize knowledge memory with config-driven providers."""
        self._config = config or KnowledgeMemoryConfig()

        # Provider instance (resolved during initialization)
        self._graph_provider: GraphDBProvider | None = None

        # Contexts tracking
        self._contexts: set[str] = set()
        self._initialized = False

        logger.info(f"Initialized KnowledgeMemory with config: {self._config}")

    @property
    def initialized(self) -> bool:
        """Check if memory is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize knowledge memory and providers."""
        if self._initialized:
            return

        logger.info("Initializing KnowledgeMemory...")

        try:
            # Get graph provider using config-driven approach
            provider = await provider_registry.get_by_config(self._config.graph_provider_config)
            if not provider:
                raise MemoryError(
                    f"Graph provider not found: {self._config.graph_provider_config}",
                    operation="initialize",
                    provider_config=self._config.graph_provider_config,
                )
            self._graph_provider = cast(GraphDBProvider, provider)

            self._initialized = True
            logger.info("KnowledgeMemory initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeMemory: {e}")
            raise MemoryError(
                f"Knowledge memory initialization failed: {str(e)}", operation="initialize", cause=e
            ) from e

    async def shutdown(self) -> None:
        """Shutdown knowledge memory."""
        if not self._initialized:
            return

        logger.info("Shutting down KnowledgeMemory...")

        # Providers are managed by the registry, no explicit cleanup needed
        self._graph_provider = None
        self._contexts.clear()

        self._initialized = False
        logger.info("KnowledgeMemory shutdown completed")

    async def create_context(
        self, context_name: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """Create a memory context."""
        if not self._initialized:
            raise MemoryError("KnowledgeMemory not initialized")

        logger.debug(f"Creating context: {context_name}")

        # Knowledge memory doesn't need explicit context creation
        # Contexts are managed through entity types and metadata
        self._contexts.add(context_name)

        logger.debug(f"Context '{context_name}' registered")
        return context_name

    async def store(self, request: MemoryStoreRequest) -> str:
        """Store knowledge as entities and relationships in the graph database."""
        if not self._initialized:
            raise MemoryError("KnowledgeMemory not initialized")

        try:
            # Extract content for entity creation
            content = self._extract_content(request.value)

            # Handle Entity objects directly
            if isinstance(request.value, Entity):
                entity = request.value
                if not entity.id:
                    entity.id = request.key

                if self._graph_provider is None:
                    raise RuntimeError("Graph provider not initialized")
                await self._graph_provider.add_entity(entity)
                logger.debug(f"Stored entity with ID '{entity.id}' of type '{entity.type}'")
                return request.key

            # Handle dictionary with entity information
            if hasattr(request.value, "model_dump") and isinstance(
                request.value.model_dump(), dict
            ):
                item_dict = request.value.model_dump()
            elif isinstance(request.value, dict):
                item_dict = request.value
            else:
                item_dict = {"content": content}

            # Create entity from content
            # Fail fast - no fallbacks allowed
            if "type" not in item_dict:
                raise ValueError("Memory item must contain 'type' field")
            entity_type = item_dict["type"]

            # Get or create entity
            if self._graph_provider is None:
                raise RuntimeError("Graph provider not initialized")
            existing_entity = await self._graph_provider.get_entity(request.key)

            if not existing_entity:
                entity = Entity(
                    id=request.key,
                    type=entity_type,
                    importance=self._config.default_importance,
                    source="conversation",
                    vector_id=None,
                )
            else:
                entity = existing_entity

            # Add content as attribute
            entity.attributes["content"] = EntityAttribute(
                name="content", value=content, confidence=0.9, source="memory_store"
            )

            # Add metadata as attributes
            if request.metadata:
                for key, value in request.metadata.items():
                    entity.attributes[key] = EntityAttribute(
                        name=key, value=str(value), confidence=0.9, source="metadata"
                    )

            # Store timestamp
            if hasattr(request.value, "timestamp"):
                entity.attributes["timestamp"] = EntityAttribute(
                    name="timestamp",
                    value=request.value.timestamp.isoformat(),
                    confidence=1.0,
                    source="system",
                )

            # Update entity
            await self._graph_provider.add_entity(entity)
            logger.debug(f"Stored knowledge entity '{request.key}' in context '{request.context}'")
            return request.key

        except Exception as e:
            raise MemoryError(
                f"Failed to store knowledge item '{request.key}': {str(e)}",
                operation="store",
                context=request.context,
                key=request.key,
                cause=e,
            ) from e

    async def retrieve(self, request: MemoryRetrieveRequest) -> MemoryItem | None:
        """Retrieve a knowledge entity by key."""
        if not self._initialized:
            raise MemoryError("KnowledgeMemory not initialized")

        try:
            # Get entity by ID
            if self._graph_provider is None:
                raise RuntimeError("Graph provider not initialized")
            entity = await self._graph_provider.get_entity(request.key)

            if not entity:
                logger.debug(f"Entity with ID '{request.key}' not found")
                return None

            # Create memory item from entity
            content = (
                entity.attributes["content"]
                if "content" in entity.attributes
                else EntityAttribute(
                    name="content", value=str(entity), confidence=1.0, source="system"
                )
            ).value

            # Note: MemoryItem uses default MemoryItemMetadata

            item = MemoryItem(
                key=request.key,
                value=content,
                context=request.context or "knowledge",
                updated_at=datetime.now(),
            )

            logger.debug(
                f"Retrieved knowledge entity '{request.key}' from context '{request.context}'"
            )
            return item

        except Exception as e:
            raise MemoryError(
                f"Knowledge retrieval failed: {e}", operation="retrieve", key=request.key, cause=e
) from e

    async def search(self, request: MemorySearchRequest) -> list[MemorySearchResult]:
        """Search for knowledge entities using graph database capabilities."""
        if not self._initialized:
            raise MemoryError("KnowledgeMemory not initialized")

        try:
            # Determine entity type filter
            entity_type = None
            if request.context:
                entity_type = request.context

            search_results: list[MemorySearchResult] = []

            # Use graph provider search if available
            if self._graph_provider is None:
                raise RuntimeError("Graph provider not initialized")
            if hasattr(self._graph_provider, "search_entities"):
                search_result = await self._graph_provider.search_entities(
                    query=request.query,
                    entity_type=entity_type,
                    limit=min(
                        request.limit or self._config.max_search_results,
                        self._config.max_search_results,
                    ),
                )

                for entity in search_result.entities:
                    # Extract content from entity
                    content = (
                        entity.attributes["content"]
                        if "content" in entity.attributes
                        else EntityAttribute(
                            name="content", value=str(entity), confidence=1.0, source="system"
                        )
                    ).value

                    # Create memory item
                    item = MemoryItem(
                        key=entity.id,
                        value=content,
                        context=entity_type or request.context or "knowledge",
                        updated_at=datetime.now(),
                    )

                    # Calculate relevance score (simplified)
                    score = entity.importance  # Use entity importance as base score

                    # Create search metadata with proper typing
                    from .models import MemorySearchMetadata

                    search_metadata = MemorySearchMetadata(
                        search_query=request.query,
                        search_type="graph_search",
                        total_results=len(search_result.entities),
                        result_rank=len(search_results) + 1,
                    )
                    search_results.append(
                        MemorySearchResult(item=item, score=score, metadata=search_metadata)
                    )
            else:
                # Fail fast - no fallbacks allowed
                raise NotImplementedError(
                    f"Graph provider '{self._graph_provider}' does not support search_entities operation"
                )

            logger.debug(
                f"Found {len(search_results)} knowledge entities for query '{request.query}'"
            )
            return search_results

        except Exception as e:
            raise MemoryError(f"Knowledge search failed: {e}", operation="search", cause=e) from e

    async def retrieve_relevant(
        self, query: str, context: str | None = None, limit: int = 10
    ) -> list[str]:
        """Retrieve relevant knowledge based on semantic relationships."""
        if not self._initialized:
            raise MemoryError("KnowledgeMemory not initialized")

        search_request = MemorySearchRequest(
            query=query,
            context=context,
            limit=limit,
            threshold=None,
            sort_by=None,
            search_type="hybrid",
            metadata_filter=None,
        )

        search_results = await self.search(search_request)

        # Convert to string list format
        return [f"{result.item.key}: {result.item.value}" for result in search_results]

    async def wipe_context(self, context: str) -> None:
        """Remove all entities from a specific context/type."""
        if not self._initialized:
            raise MemoryError("KnowledgeMemory not initialized")

        try:
            # Search for all entities of this type/context
            search_request = MemorySearchRequest(
                query="*",  # Match all
                context=context,
                limit=10000,  # Large number to get all
                threshold=None,
                sort_by=None,
                search_type="hybrid",
                metadata_filter=None,
            )

            results = await self.search(search_request)

            if self._graph_provider is None:
                raise RuntimeError("Graph provider not initialized")
            if hasattr(self._graph_provider, "delete_entity"):
                for result in results:
                    await self._graph_provider.delete_entity(result.item.key)

                logger.info(f"Wiped knowledge context '{context}', removed {len(results)} entities")
            else:
                logger.warning("Graph provider does not support entity deletion")

            self._contexts.discard(context)

        except Exception as e:
            raise MemoryError(
                f"Failed to wipe knowledge context '{context}': {str(e)}",
                operation="wipe",
                context=context,
                cause=e,
            ) from e

    async def add_relationship(
        self,
        source_id: str,
        relation_type: str,
        target_id: str,
        confidence: float = 0.9,
        source: str = "system",
    ) -> None:
        """Add a relationship between knowledge entities."""
        if not self._initialized:
            raise MemoryError("KnowledgeMemory not initialized")

        try:
            # Get source entity
            if self._graph_provider is None:
                raise RuntimeError("Graph provider not initialized")
            source_entity = await self._graph_provider.get_entity(source_id)
            if not source_entity:
                raise MemoryError(f"Source entity '{source_id}' not found")

            # Create relationship
            relationship = EntityRelationship(
                relation_type=relation_type,
                target_entity=target_id,
                confidence=confidence,
                source=source,
            )

            # Add to source entity
            source_entity.relationships.append(relationship)

            # Update entity
            await self._graph_provider.add_entity(source_entity)
            logger.debug(
                f"Added relationship '{relation_type}' from '{source_id}' to '{target_id}'"
            )

        except Exception as e:
            raise MemoryError(
                f"Failed to add relationship: {str(e)}",
                operation="add_relationship",
                source_id=source_id,
                target_id=target_id,
                cause=e,
            ) from e

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get a knowledge entity by ID."""
        if not self._initialized:
            raise MemoryError("KnowledgeMemory not initialized")

        try:
            if self._graph_provider is None:
                raise RuntimeError("Graph provider not initialized")
            result = await self._graph_provider.get_entity(entity_id)
            # Result is guaranteed to be Entity or None from graph provider
            if result is None:
                return None
            if not isinstance(result, Entity):
                raise RuntimeError(f"Graph provider returned unexpected type: {type(result)}")
            return result
        except Exception as e:
            raise MemoryError(
                f"Failed to get entity '{entity_id}': {e}",
                operation="get_entity",
                key=entity_id,
                cause=e,
) from e

    def _extract_content(self, item: Any) -> str:
        """Extract content from any item for entity storage."""
        if hasattr(item, "content"):
            return str(item.content)
        elif hasattr(item, "value"):
            return str(item.value)
        elif hasattr(item, "text"):
            return str(item.text)
        elif hasattr(item, "message"):
            return str(item.message)
        else:
            return str(item)

    def get_stats(self) -> dict[str, Any]:
        """Get knowledge memory statistics."""
        return {
            "initialized": self._initialized,
            "context_count": len(self._contexts),
            "contexts": list(self._contexts),
            "config": self._config.model_dump(),
            "provider": {"graph": self._config.graph_provider_config},
        }
