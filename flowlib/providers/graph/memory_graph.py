"""In-memory graph database implementation with provider interface.

This module provides a complete in-memory graph database that implements the GraphDBProvider
interface. It is a real graph database that stores data in memory - suitable for testing,
development, and small-scale applications where persistence is not required.

WARNING: This is an actual in-memory database, not just a provider client.
All data is lost when the process terminates.
"""

import asyncio
import logging
import copy
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

from pydantic import ValidationError

from flowlib.core.errors.errors import ProviderError, ErrorContext
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.core.decorators import provider
from flowlib.providers.graph.base import GraphDBProvider, GraphDBProviderSettings
from flowlib.providers.graph.models import (
    Entity, EntityAttribute, EntityRelationship, EntitySearchResult, 
    GraphStoreResult, GraphQueryParams, GraphQueryResult,
    RelationshipSearchResult, GraphDeleteResult, GraphStats, TraversalParams)

logger = logging.getLogger(__name__)


class InMemoryGraphDatabase:
    """Pure in-memory graph database implementation.
    
    This is a complete graph database that stores all data in memory.
    It provides thread-safe operations and maintains graph structure integrity.
    
    WARNING: All data is volatile and lost when the process terminates.
    """
    
    def __init__(self):
        self._entities: Dict[str, Entity] = {}
        self._relationships: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
    
    async def store_entity(self, entity: Entity) -> str:
        """Store an entity in the database."""
        async with self._lock:
            entity_copy = copy.deepcopy(entity)
            self._entities[entity.id] = entity_copy
            
            # Initialize relationship storage for this entity
            if entity.id not in self._relationships:
                self._relationships[entity.id] = []
                
            return entity.id
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity from the database."""
        async with self._lock:
            entity = self._entities[entity_id] if entity_id in self._entities else None
            return copy.deepcopy(entity) if entity else None
    
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and all its relationships."""
        async with self._lock:
            if entity_id not in self._entities:
                return False
                
            # Remove entity
            del self._entities[entity_id]
            
            # Remove all relationships involving this entity
            if entity_id in self._relationships:
                del self._relationships[entity_id]
                
            # Remove relationships where this entity is the target
            for source_id, rels in self._relationships.items():
                self._relationships[source_id] = [
                    rel for rel in rels if (rel["target"] if "target" in rel else None) != entity_id
                ]
            
            return True
    
    async def add_relationship(self, source_id: str, target_id: str, relation_type: str, properties: Dict[str, Any]) -> None:
        """Add a relationship between entities."""
        async with self._lock:
            if source_id not in self._entities:
                raise ValueError(f"Source entity {source_id} does not exist")
            if target_id not in self._entities:
                raise ValueError(f"Target entity {target_id} does not exist")
                
            if source_id not in self._relationships:
                self._relationships[source_id] = []
                
            # Check for duplicate relationship
            for existing_rel in self._relationships[source_id]:
                if (existing_rel["target"] == target_id and 
                    existing_rel["relation_type"] == relation_type):
                    # Update existing relationship with new properties
                    existing_rel["properties"].update(properties)
                    return
                
            relationship = {
                "target": target_id,
                "relation_type": relation_type,
                "properties": properties.copy()
            }
            
            self._relationships[source_id].append(relationship)
    
    async def query_relationships(self, entity_id: str, relation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query relationships for an entity."""
        async with self._lock:
            if entity_id not in self._relationships:
                return []
                
            relationships = self._relationships[entity_id]
            
            if relation_type:
                relationships = [r for r in relationships if r["relation_type"] == relation_type]
                
            return [copy.deepcopy(rel) for rel in relationships]

    async def remove_relationship(self, source_id: str, target_id: str, relation_type: str) -> None:
        """Remove a specific relationship between entities."""
        async with self._lock:
            if source_id not in self._relationships:
                return  # No relationships to remove
                
            # Filter out the specific relationship
            self._relationships[source_id] = [
                rel for rel in self._relationships[source_id] 
                if not (rel["target"] == target_id and rel["relation_type"] == relation_type)
            ]
    
    async def search_entities(self, query: Optional[str] = None, entity_type: Optional[str] = None, 
                            tags: Optional[List[str]] = None, limit: int = 10) -> List[Entity]:
        """Search entities by criteria."""
        async with self._lock:
            results = []
            
            for entity in self._entities.values():
                if len(results) >= limit:
                    break
                    
                # Type filter
                if entity_type and entity.type != entity_type:
                    continue
                    
                # Tag filter
                if tags and not any(tag in entity.tags for tag in tags):
                    continue
                    
                # Text query filter (search in ID and attribute values)
                if query:
                    query_lower = query.lower()
                    match_found = False
                    
                    if query_lower in entity.id.lower():
                        match_found = True
                    else:
                        for attr in entity.attributes.values():
                            if query_lower in str(attr.value).lower():
                                match_found = True
                                break
                    
                    if not match_found:
                        continue
                
                results.append(copy.deepcopy(entity))
            
            return results

    async def traverse(self, start_id: str, relation_types: Optional[List[str]] = None, max_depth: int = 2) -> List[Entity]:
        """Traverse the graph starting from an entity."""
        async with self._lock:
            if start_id not in self._entities:
                return []
            
            visited = set()
            result = []
            queue = [(start_id, 0)]  # (entity_id, depth)
            
            while queue:
                current_id, depth = queue.pop(0)
                
                if current_id in visited or depth > max_depth:
                    continue
                    
                visited.add(current_id)
                
                if current_id in self._entities:
                    result.append(copy.deepcopy(self._entities[current_id]))
                
                # Add connected entities to queue
                if depth < max_depth and current_id in self._relationships:
                    for rel in self._relationships[current_id]:
                        if relation_types is None or rel["relation_type"] in relation_types:
                            target_id = rel["target"]
                            if target_id not in visited:
                                queue.append((target_id, depth + 1))
            
            return result


@provider(provider_type="graph_db", name="memory-graph", settings_class=GraphDBProviderSettings)
class MemoryGraphProvider(GraphDBProvider):
    """Provider interface to an in-memory graph database.
    
    This provider connects to an embedded in-memory graph database.
    It is a real graph database implementation, not a mock or stub.
    
    Use cases:
    - Testing and development
    - Small applications where persistence is not required
    - Prototyping and demos
    
    WARNING: This is a volatile database - all data is lost when the process terminates.
    """
    
    def __init__(self, name: str = "memory-graph", provider_type: str = "graph_db", settings: Optional[GraphDBProviderSettings] = None):
        """Initialize provider with embedded in-memory database.
        
        Args:
            name: Unique provider name
            provider_type: Provider type
            settings: Optional provider settings
        """
        settings = settings or GraphDBProviderSettings()
        super().__init__(name=name, provider_type=provider_type, settings=settings)
        self._database: Optional[InMemoryGraphDatabase] = None
    
    async def initialize(self) -> None:
        """Initialize the in-memory database."""
        self._database = InMemoryGraphDatabase()
        self._initialized = True
        logger.info(f"In-memory graph database '{self.name}' initialized (volatile storage)")
    
    async def shutdown(self) -> None:
        """Shutdown the database and clear all data."""
        if self._database:
            # Clear all data
            self._database._entities.clear()
            self._database._relationships.clear()
            self._database = None
        
        self._initialized = False
        logger.info(f"In-memory graph database '{self.name}' shutdown (all data lost)")
    
    async def add_entity(self, entity: Entity) -> GraphStoreResult:
        """Add or update an entity in the database."""
        if not self._initialized or not self._database:
            raise ProviderError(
                message="Provider not initialized",
                context=ErrorContext.create(
                    flow_name="memory_graph",
                    error_type="InitializationError",
                    error_location="add_entity",
                    component="MemoryGraphProvider",
                    operation="add_entity"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="add_entity",
                    retry_count=0
                ),
                cause=None
            )
        try:
            # Store the entity first
            entity_id = await self._database.store_entity(entity)
            stored_relationships = []
            
            # Add relationships
            for rel in entity.relationships:
                await self.add_relationship(
                    entity.id,
                    rel.target_entity,
                    rel.relation_type,
                    rel  # Pass the EntityRelationship model directly
                )
                stored_relationships.append(f"{entity.id}-{rel.relation_type}->{rel.target_entity}")
            
            return GraphStoreResult(
                success=True,
                stored_entities=[entity_id],
                stored_relationships=stored_relationships,
                failed_entities=[],
                failed_relationships=[],
                error_details={}
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to add entity: {str(e)}",
                context=ErrorContext.create(
                    flow_name="memory_graph",
                    error_type="EntityAddError",
                    error_location="add_entity",
                    component="MemoryGraphProvider",
                    operation="add_entity"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="add_entity",
                    retry_count=0
                ),
                cause=e
            )
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        if not self._initialized or not self._database:
            raise ProviderError(
                message="Provider not initialized",
                context=ErrorContext.create(
                    flow_name="memory_graph",
                    error_type="InitializationError",
                    error_location="get_entity",
                    component="MemoryGraphProvider",
                    operation="get_entity"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="get_entity",
                    retry_count=0
                ),
                cause=None
            )
        
        try:
            return await self._database.get_entity(entity_id)
        except Exception as e:
            raise ProviderError(
                message=f"Failed to get entity: {str(e)}",
                context=ErrorContext.create(
                    flow_name="memory_graph",
                    error_type="EntityGetError",
                    error_location="get_entity",
                    component="MemoryGraphProvider",
                    operation="get_entity"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="get_entity",
                    retry_count=0
                ),
                cause=e
            )
    
    async def delete_entity(self, entity_id: str) -> GraphDeleteResult:
        """Delete an entity and all its relationships."""
        if not self._initialized or not self._database:
            raise ProviderError(
                message="Provider not initialized",
                context=ErrorContext.create(
                    flow_name="memory_graph",
                    error_type="InitializationError",
                    error_location="delete_entity",
                    component="MemoryGraphProvider",
                    operation="delete_entity"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="delete_entity",
                    retry_count=0
                ),
                cause=None
            )
        
        try:
            deleted = await self._database.delete_entity(entity_id)
            if deleted:
                return GraphDeleteResult(
                    success=True,
                    deleted_entities=[entity_id],
                    deleted_relationships=[],  # Relationships are deleted internally
                    not_found_entities=[],
                    not_found_relationships=[],
                    error_details={}
                )
            else:
                return GraphDeleteResult(
                    success=False,
                    deleted_entities=[],
                    deleted_relationships=[],
                    not_found_entities=[entity_id],
                    not_found_relationships=[],
                    error_details={"reason": "Entity not found"}
                )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete entity: {str(e)}",
                context=ErrorContext.create(
                    flow_name="memory_graph",
                    error_type="EntityDeleteError",
                    error_location="delete_entity",
                    component="MemoryGraphProvider",
                    operation="delete_entity"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="delete_entity",
                    retry_count=0
                ),
                cause=e
            )
    
    async def add_relationship(self, source_id: str, target_entity: str, relation_type: str, relationship: EntityRelationship) -> None:
        """Add a relationship between two entities."""
        if not self._initialized or not self._database:
            raise ProviderError(
                message="Provider not initialized",
                context=ErrorContext.create(
                    flow_name="memory_graph",
                    error_type="InitializationError",
                    error_location="add_relationship",
                    component="MemoryGraphProvider",
                    operation="add_relationship"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="add_relationship",
                    retry_count=0
                ),
                cause=None
            )
        try:
            # Check that both source and target exist
            source = await self._database.get_entity(source_id)
            if not source:
                raise ProviderError(
                    message=f"Source entity {source_id} does not exist",
                    context=ErrorContext.create(
                        flow_name="memory_graph",
                        error_type="EntityNotFoundError",
                        error_location="add_relationship",
                        component="MemoryGraphProvider",
                        operation="add_relationship"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="graph_db",
                        operation="add_relationship",
                        retry_count=0
                    ),
                    cause=None
                )
            target = await self._database.get_entity(target_entity)
            if not target:
                raise ProviderError(
                    message=f"Target entity {target_entity} does not exist",
                    context=ErrorContext.create(
                        flow_name="memory_graph",
                        error_type="EntityNotFoundError",
                        error_location="add_relationship",
                        component="MemoryGraphProvider",
                        operation="add_relationship"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="graph_db",
                        operation="add_relationship",
                        retry_count=0
                    ),
                    cause=None
                )
            # Convert EntityRelationship model to properties dict
            properties = {
                "confidence": relationship.confidence,
                "source": relationship.source,
                "timestamp": relationship.timestamp
            }
            await self._database.add_relationship(source_id, target_entity, relation_type, properties)
        except Exception as e:
            raise ProviderError(
                message=f"Failed to add relationship: {str(e)}",
                context=ErrorContext.create(
                    flow_name="memory_graph",
                    error_type="RelationshipAddError",
                    error_location="add_relationship",
                    component="MemoryGraphProvider",
                    operation="add_relationship"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="add_relationship",
                    retry_count=0
                ),
                cause=e
            )
    
    async def query_relationships(self, entity_id: str, relation_type: Optional[str] = None, direction: str = "outgoing") -> RelationshipSearchResult:
        """Query relationships for an entity."""
        if not self._initialized or not self._database:
            raise ProviderError(
                message="Provider not initialized",
                context=ErrorContext.create(
                    flow_name="memory_graph",
                    error_type="InitializationError",
                    error_location="query_relationships",
                    component="MemoryGraphProvider",
                    operation="query_relationships"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="query_relationships",
                    retry_count=0
                ),
                cause=None
            )
        try:
            relationships = []
            if direction == "outgoing":
                rels = await self._database.query_relationships(entity_id, relation_type)
                for rel in rels:
                    relationships.append(EntityRelationship(
                        relation_type=rel["relation_type"],
                        target_entity=rel["target"],
                        confidence=rel["properties"]["confidence"] if "confidence" in rel["properties"] else 0.8,
                        source=rel["properties"]["source"] if "source" in rel["properties"] else "",
                        timestamp=rel["properties"]["timestamp"] if "timestamp" in rel["properties"] else datetime.now().isoformat()
                    ))
            elif direction == "incoming":
                async with self._database._lock:
                    for source_id, rels in self._database._relationships.items():
                        for rel in rels:
                            if rel["target"] == entity_id:
                                if relation_type is None or rel["relation_type"] == relation_type:
                                    relationships.append(EntityRelationship(
                                        relation_type=rel["relation_type"],
                                        target_entity=entity_id,
                                        confidence=rel["properties"]["confidence"] if "confidence" in rel["properties"] else 0.8,
                                        source=rel["properties"]["source"] if "source" in rel["properties"] else "",
                                        timestamp=rel["properties"]["timestamp"] if "timestamp" in rel["properties"] else datetime.now().isoformat()
                                    ))
            else:
                raise ProviderError(
                    message=f"Unsupported direction: {direction}",
                    context=ErrorContext.create(
                        flow_name="memory_graph",
                        error_type="InvalidParameterError",
                        error_location="query_relationships",
                        component="MemoryGraphProvider",
                        operation="query_relationships"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="graph_db",
                        operation="query_relationships",
                        retry_count=0
                    ),
                    cause=None
                )
            return RelationshipSearchResult(
                success=True,
                relationships=relationships,
                total_count=len(relationships),
                source_entity=entity_id if direction == "outgoing" else None,
                target_entity=entity_id if direction == "incoming" else None,
                execution_time_ms=None,
                metadata={"direction": direction, "relation_type": relation_type}
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to query relationships: {str(e)}",
                context=ErrorContext.create(
                    flow_name="memory_graph",
                    error_type="RelationshipQueryError",
                    error_location="query_relationships",
                    component="MemoryGraphProvider",
                    operation="query_relationships"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="query_relationships",
                    retry_count=0
                ),
                cause=e
            )

    async def delete_relationship(self, source_id: str, target_entity: str, relation_type: Optional[str] = None) -> bool:
        """Delete relationship(s) between entities."""
        if not self._initialized or not self._database:
            raise ProviderError(
                message="Provider not initialized",
                context=ErrorContext.create(
                    flow_name="memory_graph",
                    error_type="InitializationError",
                    error_location="delete_relationship",
                    component="MemoryGraphProvider",
                    operation="delete_relationship"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="delete_relationship",
                    retry_count=0
                ),
                cause=None
            )
        
        try:
            async with self._database._lock:
                if source_id not in self._database._relationships:
                    return False
                
                original_count = len(self._database._relationships[source_id])
                
                if relation_type:
                    # Remove specific relationship type
                    self._database._relationships[source_id] = [
                        rel for rel in self._database._relationships[source_id]
                        if not (rel["target"] == target_entity and rel["relation_type"] == relation_type)
                    ]
                else:
                    # Remove all relationships to target
                    self._database._relationships[source_id] = [
                        rel for rel in self._database._relationships[source_id]
                        if rel["target"] != target_entity
                    ]
                
                return len(self._database._relationships[source_id]) < original_count
                
        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete relationship: {str(e)}",
                context=ErrorContext.create(
                    flow_name="memory_graph",
                    error_type="RelationshipDeleteError",
                    error_location="delete_relationship",
                    component="MemoryGraphProvider",
                    operation="delete_relationship"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="delete_relationship",
                    retry_count=0
                ),
                cause=e
            )

    async def remove_relationship(self, source_id: str, target_entity: str, relation_type: str) -> None:
        """Remove a relationship between entities."""
        if not self._initialized or not self._database:
            raise ProviderError(
                message="Provider not initialized",
                context=ErrorContext.create(
                    flow_name="memory_graph",
                    error_type="InitializationError",
                    error_location="remove_relationship",
                    component="MemoryGraphProvider",
                    operation="remove_relationship"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="remove_relationship",
                    retry_count=0
                ),
                cause=None
            )
        
        try:
            await self._database.remove_relationship(source_id, target_entity, relation_type)
        except Exception as e:
            raise ProviderError(
                message=f"Failed to remove relationship: {str(e)}",
                context=ErrorContext.create(
                    flow_name="memory_graph",
                    error_type="RelationshipRemoveError",
                    error_location="remove_relationship",
                    component="MemoryGraphProvider",
                    operation="remove_relationship"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="remove_relationship",
                    retry_count=0
                ),
                cause=e
            )
    
    async def traverse(self, start_id: str, relation_types: Optional[List[str]] = None, max_depth: int = 2) -> List[Entity]:
        """Traverse the graph starting from an entity."""
        if not self._initialized or not self._database:
            raise ProviderError(
                message="Provider not initialized",
                context=ErrorContext.create(
                    flow_name="memory_graph",
                    error_type="InitializationError",
                    error_location="traverse",
                    component="MemoryGraphProvider",
                    operation="traverse"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="traverse",
                    retry_count=0
                ),
                cause=None
            )
        
        try:
            return await self._database.traverse(start_id, relation_types, max_depth)
        except Exception as e:
            raise ProviderError(
                message=f"Failed to traverse graph: {str(e)}",
                context=ErrorContext.create(
                    flow_name="memory_graph",
                    error_type="TraversalError",
                    error_location="traverse",
                    component="MemoryGraphProvider",
                    operation="traverse"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="traverse",
                    retry_count=0
                ),
                cause=e
            )
    
    async def search_entities(self, query: Optional[str] = None, entity_type: Optional[str] = None, 
                            tags: Optional[List[str]] = None, limit: int = 10) -> List[Entity]:
        """Search for entities based on criteria."""
        if not self._initialized or not self._database:
            raise ProviderError(
                message="Provider not initialized",
                context=ErrorContext.create(
                    flow_name="memory_graph",
                    error_type="InitializationError",
                    error_location="search_entities",
                    component="MemoryGraphProvider",
                    operation="search_entities"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="search_entities",
                    retry_count=0
                ),
                cause=None
            )
        
        try:
            return await self._database.search_entities(query, entity_type, tags, limit)
        except Exception as e:
            raise ProviderError(
                message=f"Failed to search entities: {str(e)}",
                context=ErrorContext.create(
                    flow_name="memory_graph",
                    error_type="EntitySearchError",
                    error_location="search_entities",
                    component="MemoryGraphProvider",
                    operation="search_entities"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="search_entities",
                    retry_count=0
                ),
                cause=e
            )
    
    async def query(self, query: str, params: Optional[GraphQueryParams] = None) -> GraphQueryResult:
        """Execute a simple query against the in-memory graph."""
        if not self._initialized or not self._database:
            raise ProviderError(
                message="Provider not initialized",
                context=ErrorContext.create(
                    flow_name="memory_graph",
                    error_type="InitializationError",
                    error_location="query",
                    component="MemoryGraphProvider",
                    operation="query"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="query",
                    retry_count=0
                ),
                cause=None
            )
        try:
            nodes = []
            edges = []
            
            # Parse simple query patterns for the in-memory implementation
            if query.startswith("find_entities"):
                entity_type = None
                name = None
                # Parse query string for type and name
                parts = query.split()
                for part in parts[1:]:
                    if part.startswith("type="):
                        entity_type = part.split("=", 1)[1]
                    elif part.startswith("name="):
                        name = part.split("=", 1)[1]
                
                # Apply params if provided
                limit = 10
                if params:
                    if params.limit:
                        limit = params.limit
                    # Check extra_params for additional filters
                    if 'entity_type' in params.extra_params:
                        entity_type = params.extra_params['entity_type']
                    if 'name' in params.extra_params:
                        name = params.extra_params['name']
                
                # Search entities
                entities = await self._database.search_entities(
                    query=name, 
                    entity_type=entity_type,
                    limit=limit
                )
                
                # Convert entities to node format
                for entity in entities:
                    nodes.append({
                        "id": entity.id,
                        "type": entity.type,
                        "attributes": {k: v.model_dump() for k, v in entity.attributes.items()},
                        "importance": entity.importance,
                        "source": entity.source
                    })
                    
            elif query.startswith("get_relationships"):
                # Parse entity_id from query
                parts = query.split()
                entity_id = None
                for part in parts[1:]:
                    if part.startswith("entity_id="):
                        entity_id = part.split("=", 1)[1]
                        
                if entity_id:
                    # Get relationships for entity
                    result = await self.query_relationships(entity_id)
                    for rel in result.relationships:
                        edges.append({
                            "source": entity_id,
                            "target": rel.target_entity,
                            "type": rel.relation_type,
                            "confidence": rel.confidence
                        })
            else:
                # For unsupported queries, return empty result
                logger.warning(f"Unsupported query pattern: {query}")
                
            return GraphQueryResult(
                success=True,
                nodes=nodes,
                edges=edges,
                metadata={
                    "query": query,
                    "provider": "memory_graph"
                },
                execution_time_ms=None,
                total_count=len(nodes) + len(edges)
            )
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to execute query: {str(e)}",
                context=ErrorContext.create(
                    flow_name="memory_graph",
                    error_type="QueryExecutionError",
                    error_location="query",
                    component="MemoryGraphProvider",
                    operation="query"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="query",
                    retry_count=0
                ),
                cause=e
            )

    async def bulk_add_entities(self, entities: List[Entity]) -> GraphStoreResult:
        """Add multiple entities in bulk to the in-memory graph."""
        if not self._initialized or not self._database:
            raise ProviderError(
                message="Provider not initialized",
                context=ErrorContext.create(
                    flow_name="memory_graph",
                    error_type="InitializationError",
                    error_location="bulk_add_entities",
                    component="MemoryGraphProvider",
                    operation="bulk_add_entities"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="bulk_add_entities",
                    retry_count=0
                ),
                cause=None
            )
        stored_entities = []
        failed_entities = []
        all_relationships = []
        for entity in entities:
            try:
                result = await self.add_entity(entity)
                if result.success:
                    stored_entities.extend(result.stored_entities)
                    all_relationships.extend(result.stored_relationships)
                else:
                    failed_entities.append(entity.id)
            except Exception:
                failed_entities.append(entity.id)
        return GraphStoreResult(
            success=len(failed_entities) == 0,
            stored_entities=stored_entities,
            stored_relationships=all_relationships,
            failed_entities=failed_entities,
            failed_relationships=[],
            error_details={"failed_entities": failed_entities} if failed_entities else {},
            execution_time_ms=None
        )