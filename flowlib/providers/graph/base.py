"""Graph database provider base class.

This module defines the base class for graph database providers,
establishing a common interface for entity and relationship operations.
"""

import logging
from abc import abstractmethod
from typing import Dict, List, Optional, Any

from flowlib.providers.core.base import Provider
from ...flows.base import FlowSettings
from .models import (
    Entity, GraphQueryResult, EntitySearchResult, RelationshipSearchResult, 
    GraphStoreResult, GraphDeleteResult, GraphUpdateResult, GraphStats, 
    GraphHealthResult, EntityRelationship, GraphQueryParams, TraversalParams
)

logger = logging.getLogger(__name__)

class GraphDBProviderSettings(FlowSettings):
    """Settings for graph database providers.
    
    Attributes:
        max_retries: Maximum retries for graph operations
        retry_delay_seconds: Delay between retries in seconds
        timeout_seconds: Operation timeout in seconds
        max_batch_size: Maximum batch size for operations
    """
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 30.0
    max_batch_size: int = 100

class GraphDBProvider(Provider):
    """Base class for graph database providers.
    
    This class defines the common interface for all graph database providers,
    with methods for entity and relationship operations.
    """
    
    def __init__(self, name: str = "graph", provider_type: str = "graph_db", settings: Optional[GraphDBProviderSettings] = None):
        """Initialize graph database provider.
        
        Args:
            name: Unique provider name
            provider_type: Provider type
            settings: Optional provider settings
        """
        super().__init__(name=name, provider_type=provider_type, settings=settings)
        
    @abstractmethod
    async def add_entity(self, entity: Entity) -> GraphStoreResult:
        """Add or update an entity node.
        
        Args:
            entity: Entity to add or update (must be Entity model)
            
        Returns:
            GraphStoreResult
            
        Raises:
            ProviderError: If entity creation fails
        """
        raise NotImplementedError("Subclasses must implement add_entity()")
        
    @abstractmethod
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID.
        
        Args:
            entity_id: Unique identifier of the entity
            
        Returns:
            Entity object if found, None otherwise
            
        Raises:
            ProviderError: If entity retrieval fails
        """
        raise NotImplementedError("Subclasses must implement get_entity()")
        
    @abstractmethod
    async def add_relationship(
        self,
        source_id: str,
        target_entity: str,
        relation_type: str,
        relationship: EntityRelationship
    ) -> None:
        """Add a relationship between two entities.
        
        Args:
            source_id: ID of the source entity
            target_entity: ID of the target entity
            relation_type: Type of relationship
            relationship: EntityRelationship model with properties
            
        Examples:
            >>> # Create a simple relationship
            >>> rel = EntityRelationship(
            ...     relation_type="knows",
            ...     target_entity="person456",
            ...     confidence=0.9,
            ...     source="conversation"
            ... )
            >>> await provider.add_relationship(
            ...     "person123",
            ...     "person456", 
            ...     "knows",
            ...     rel
            ... )
            
            >>> # Create a relationship with custom timestamp
            >>> from datetime import datetime
            >>> rel = EntityRelationship(
            ...     relation_type="employed_by",
            ...     target_entity="company789",
            ...     confidence=1.0,
            ...     source="resume",
            ...     timestamp=datetime(2023, 1, 15).isoformat()
            ... )
            >>> await provider.add_relationship(
            ...     "person123",
            ...     "company789",
            ...     "employed_by",
            ...     rel
            ... )
            
        Raises:
            ProviderError: If relationship creation fails
        """
        raise NotImplementedError("Subclasses must implement add_relationship()")
        
    @abstractmethod
    async def query_relationships(
        self, 
        entity_id: str, 
        relation_type: Optional[str] = None, 
        direction: str = "outgoing"
    ) -> RelationshipSearchResult:
        """Query relationships for an entity.
        
        Args:
            entity_id: ID of the entity
            relation_type: Optional type to filter by
            direction: 'outgoing' or 'incoming'
            
        Returns:
            RelationshipSearchResult
            
        Raises:
            ProviderError: If relationship query fails
        """
        raise NotImplementedError("Subclasses must implement query_relationships()")
        
    @abstractmethod
    async def traverse(
        self, 
        start_id: str, 
        relation_types: Optional[List[str]] = None, 
        max_depth: int = 2
    ) -> List[Entity]:
        """Traverse the graph starting from an entity.
        
        Args:
            start_id: ID of the starting entity
            relation_types: Optional list of relation types to traverse
            max_depth: Maximum traversal depth
            
        Returns:
            List of Entity objects found during traversal
            
        Examples:
            >>> # Traverse all relationships up to 2 levels deep
            >>> entities = await provider.traverse("person123")
            
            >>> # Traverse only 'knows' and 'works_with' relationships
            >>> entities = await provider.traverse(
            ...     "person123",
            ...     relation_types=["knows", "works_with"],
            ...     max_depth=3
            ... )
            
            >>> # Single-level traversal (direct connections only)
            >>> direct_connections = await provider.traverse(
            ...     "company456",
            ...     relation_types=["employs"],
            ...     max_depth=1
            ... )
            
        Raises:
            ProviderError: If traversal fails
        """
        raise NotImplementedError("Subclasses must implement traverse()")
        
    @abstractmethod
    async def query(
        self, 
        query: str, 
        params: Optional[GraphQueryParams] = None
    ) -> GraphQueryResult:
        """Execute a native query in the graph database.
        
        Args:
            query: Query string in the native query language
            params: Optional query parameters (base GraphQueryParams or provider-specific subclass)
            
        Returns:
            GraphQueryResult with nodes, edges, and metadata
            
        Note:
            Each provider implementation should accept its specific parameter type
            (e.g., Neo4jQueryParams for Neo4j, ArangoQueryParams for ArangoDB).
            The base GraphQueryParams is provided here for interface definition only.
            
        Examples:
            >>> # Simple query without parameters
            >>> result = await provider.query("MATCH (n:Person) RETURN n LIMIT 10")
            
            >>> # Query with base parameters
            >>> params = GraphQueryParams(limit=5, timeout_ms=5000)
            >>> result = await provider.query(
            ...     "MATCH (p:Person)-[:KNOWS]->(friend) RETURN p, friend",
            ...     params=params
            ... )
            
        Raises:
            ProviderError: If query execution fails
        """
        raise NotImplementedError("Subclasses must implement query()")
        
    async def get_health(self) -> GraphHealthResult:
        """Get provider health information.
        
        Returns:
            Structured health check result
            
        Raises:
            ProviderError: If health check fails
        """
        try:
            return GraphHealthResult(
                healthy=True,
                connection_status="connected",
                database_info={
                    "provider": self.name,
                    "provider_type": self.provider_type,
                    "initialized": self.initialized
                }
            )
        except Exception as e:
            # Safely handle provider name in error case
            provider_name = "unknown"
            provider_type = "unknown"
            try:
                provider_name = self.name
                provider_type = self.provider_type
            except AttributeError as attr_error:
                # Keep defaults if provider attributes are not accessible
                logger.debug(f"Could not access provider attributes during error logging: {attr_error}")
                
            logger.error(f"Error checking health for provider '{provider_name}': {str(e)}")
            return GraphHealthResult(
                healthy=False,
                connection_status="error",
                error_message=str(e),
                database_info={
                    "provider": provider_name,
                    "provider_type": provider_type
                }
            )
        
    @abstractmethod
    async def delete_entity(self, entity_id: str) -> GraphDeleteResult:
        """Delete an entity by ID.
        
        Args:
            entity_id: Unique identifier of the entity
            
        Returns:
            GraphDeleteResult
            
        Raises:
            ProviderError: If entity deletion fails
        """
        raise NotImplementedError("Subclasses must implement delete_entity()")
    
    @abstractmethod
    async def delete_relationship(
        self, 
        source_id: str, 
        target_entity: str, 
        relation_type: Optional[str] = None
    ) -> bool:
        """Delete a relationship between two entities.
        
        Args:
            source_id: ID of the source entity
            target_entity: ID of the target entity
            relation_type: Optional type of relationship
            
        Returns:
            True if deleted, False otherwise
            
        Raises:
            ProviderError: If relationship deletion fails
        """
        raise NotImplementedError("Subclasses must implement delete_relationship()")
    
    @abstractmethod
    async def bulk_add_entities(self, entities: List[Entity]) -> GraphStoreResult:
        """Bulk add entities.
        
        Args:
            entities: List of Entity models
            
        Returns:
            GraphStoreResult
            
        Raises:
            ProviderError: If bulk add fails
        """
        raise NotImplementedError("Subclasses must implement bulk_add_entities()")
    
    @abstractmethod
    async def search_entities(
        self,
        query: Optional[str] = None,
        entity_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10
    ) -> EntitySearchResult:
        """Search for entities.
        
        Args:
            query: Optional search query
            entity_type: Optional entity type
            tags: Optional list of tags
            limit: Maximum number of results
            
        Returns:
            EntitySearchResult
            
        Raises:
            ProviderError: If search fails
        """
        raise NotImplementedError("Subclasses must implement search_entities()")

    @abstractmethod
    async def remove_relationship(
        self,
        source_id: str,
        target_entity: str,
        relation_type: str
    ) -> None:
        """Remove a relationship between two entities.
        
        Args:
            source_id: ID of the source entity
            target_entity: ID of the target entity
            relation_type: Type of relationship
            
        Raises:
            ProviderError: If removal fails
        """
        raise NotImplementedError("Subclasses must implement remove_relationship()")
    
    async def get_stats(self) -> GraphStats:
        """Get graph database statistics.
        
        Returns:
            Graph statistics including entity/relationship counts
            
        Raises:
            ProviderError: If stats retrieval fails
        """
        raise NotImplementedError("Subclasses must implement get_stats()") 