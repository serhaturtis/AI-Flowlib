from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field

from flowlib.core.models import StrictBaseModel


class EntityAttribute(StrictBaseModel):
    """An attribute of an entity."""
    # Inherits strict configuration from StrictBaseModel

    name: str = Field(..., description="Name of this attribute (e.g., 'full_name', 'age')")
    value: str = Field(..., description="Value of this attribute")
    confidence: float = Field(0.9, ge=0.0, le=1.0, description="Confidence score (0-1)")
    source: str = Field("conversation", description="Source of this information")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(),
                          description="When this attribute was recorded")

class EntityRelationship(StrictBaseModel):
    """A relationship between entities."""
    # Inherits strict configuration from StrictBaseModel

    relation_type: str = Field(..., description="Type of relationship (e.g., 'friend_of')")
    target_entity: str = Field(..., description="Name or identifier of the target entity")
    confidence: float = Field(0.9, ge=0.0, le=1.0, description="Confidence in this relationship")
    source: str = Field("conversation", description="Source of this relationship")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(),
                          description="When this relationship was established")

class Entity(StrictBaseModel):
    """An entity in the knowledge graph."""
    # Inherits strict configuration from StrictBaseModel

    id: str = Field(..., description="Unique identifier for this entity")
    type: str = Field(..., description="Type of entity (person, location, event, etc.)")
    attributes: Dict[str, EntityAttribute] = Field(default_factory=dict)
    relationships: List[EntityRelationship] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    source: str = Field("conversation", description="Source of this entity information")
    importance: float = Field(0.7, ge=0.0, le=1.0, description="Overall importance of this entity")
    vector_id: Optional[str] = Field(None, description="ID in vector store if applicable")
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat(),
                             description="When this entity was last updated")

    def to_memory_item(self, attribute_name: Optional[str] = None) -> Dict[str, Any]:
        """Convert entity to a memory item for storage.
        
        Args:
            attribute_name: Optional specific attribute to convert
            
        Returns:
            Dictionary representation for memory storage
        """
        if attribute_name and attribute_name in self.attributes:
            # Return specific attribute as memory item
            attr = self.attributes[attribute_name]
            return {
                "entity_id": self.id,
                "entity_type": self.type,
                "attribute": attribute_name,
                "value": attr.value,
                "confidence": attr.confidence,
                "importance": self.importance,
                "source": attr.source,
                "tags": self.tags,
                "relationships": [{"type": r.relation_type, "target": r.target_entity} for r in self.relationships],
                "timestamp": attr.timestamp
            }
        else:
            # Return entity overview
            return {
                "entity_id": self.id,
                "entity_type": self.type,
                "attribute": "summary",
                "value": f"{self.type} with {len(self.attributes)} attributes and {len(self.relationships)} relationships",
                "confidence": 1.0,
                "importance": self.importance,
                "source": "system",
                "tags": self.tags,
                "relationships": [{"type": r.relation_type, "target": r.target_entity} for r in self.relationships],
                "timestamp": self.last_updated
            }

    def get_formatted_view(self) -> str:
        """Get a human-readable formatted view of the entity.
        
        Returns:
            String representation with attributes and relationships
        """
        lines = [f"Entity: {self.id} (Type: {self.type})"]

        # Add attributes
        if self.attributes:
            lines.append("Attributes:")
            for name, attr in self.attributes.items():
                lines.append(f"  {name}: {attr.value} (confidence: {attr.confidence:.2f})")

        # Add relationships
        if self.relationships:
            lines.append("Relationships:")
            for rel in self.relationships:
                lines.append(f"  {rel.relation_type} {rel.target_entity} (confidence: {rel.confidence:.2f})")

        # Add tags
        if self.tags:
            lines.append(f"Tags: {', '.join(self.tags)}")

        return "\n".join(lines)

class RelationshipUpdate(StrictBaseModel):
    """A relationship to be added or updated."""
    # Inherits strict configuration from StrictBaseModel

    type: str = Field(..., description="Type of relationship")
    target: str = Field(..., description="ID of the target entity")


class GraphQueryResult(StrictBaseModel):
    """Result from a graph query operation."""
    # Inherits strict configuration from StrictBaseModel

    success: bool = Field(..., description="Whether the query was successful")
    nodes: List[Dict[str, Any]] = Field(default_factory=list, description="Nodes returned by the query")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="Edges returned by the query")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional query metadata")
    execution_time_ms: Optional[float] = Field(None, description="Query execution time in milliseconds")
    total_count: Optional[int] = Field(None, description="Total count of results (for pagination)")


class EntitySearchResult(StrictBaseModel):
    """Result from entity search operation."""
    # Inherits strict configuration from StrictBaseModel

    success: bool = Field(..., description="Whether the search was successful")
    entities: List[Entity] = Field(default_factory=list, description="Found entities")
    total_count: int = Field(0, description="Total number of matching entities")
    search_query: str = Field("", description="Original search query")
    execution_time_ms: Optional[float] = Field(None, description="Search execution time in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional search metadata")


class RelationshipSearchResult(StrictBaseModel):
    """Result from relationship search operation."""
    # Inherits strict configuration from StrictBaseModel

    success: bool = Field(..., description="Whether the search was successful")
    relationships: List[EntityRelationship] = Field(default_factory=list, description="Found relationships")
    total_count: int = Field(0, description="Total number of matching relationships")
    source_entity: Optional[str] = Field(None, description="Source entity for relationship search")
    target_entity: Optional[str] = Field(None, description="Target entity for relationship search")
    execution_time_ms: Optional[float] = Field(None, description="Search execution time in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional search metadata")


class GraphStoreResult(StrictBaseModel):
    """Result from graph store operation."""
    # Inherits strict configuration from StrictBaseModel

    success: bool = Field(..., description="Whether the store operation was successful")
    stored_entities: List[str] = Field(default_factory=list, description="IDs of successfully stored entities")
    stored_relationships: List[str] = Field(default_factory=list, description="IDs of successfully stored relationships")
    failed_entities: List[str] = Field(default_factory=list, description="IDs of entities that failed to store")
    failed_relationships: List[str] = Field(default_factory=list, description="IDs of relationships that failed to store")
    error_details: Dict[str, Any] = Field(default_factory=dict, description="Details about any errors")
    execution_time_ms: Optional[float] = Field(None, description="Store execution time in milliseconds")


class GraphDeleteResult(StrictBaseModel):
    """Result from graph delete operation."""
    # Inherits strict configuration from StrictBaseModel

    success: bool = Field(..., description="Whether the delete operation was successful")
    deleted_entities: List[str] = Field(default_factory=list, description="IDs of successfully deleted entities")
    deleted_relationships: List[str] = Field(default_factory=list, description="IDs of successfully deleted relationships")
    not_found_entities: List[str] = Field(default_factory=list, description="IDs of entities that were not found")
    not_found_relationships: List[str] = Field(default_factory=list, description="IDs of relationships that were not found")
    error_details: Dict[str, Any] = Field(default_factory=dict, description="Details about any errors")
    execution_time_ms: Optional[float] = Field(None, description="Delete execution time in milliseconds")


class GraphUpdateResult(StrictBaseModel):
    """Result from graph update operation."""
    # Inherits strict configuration from StrictBaseModel

    success: bool = Field(..., description="Whether the update operation was successful")
    updated_entities: List[str] = Field(default_factory=list, description="IDs of successfully updated entities")
    updated_relationships: List[str] = Field(default_factory=list, description="IDs of successfully updated relationships")
    created_entities: List[str] = Field(default_factory=list, description="IDs of newly created entities")
    created_relationships: List[str] = Field(default_factory=list, description="IDs of newly created relationships")
    error_details: Dict[str, Any] = Field(default_factory=dict, description="Details about any errors")
    execution_time_ms: Optional[float] = Field(None, description="Update execution time in milliseconds")


class GraphStats(StrictBaseModel):
    """Statistics about a graph database."""
    # Inherits strict configuration from StrictBaseModel

    total_entities: int = Field(0, description="Total number of entities in the graph")
    total_relationships: int = Field(0, description="Total number of relationships in the graph")
    entity_types: Dict[str, int] = Field(default_factory=dict, description="Count of entities by type")
    relationship_types: Dict[str, int] = Field(default_factory=dict, description="Count of relationships by type")
    avg_degree: Optional[float] = Field(None, description="Average degree (connections) per entity")
    max_degree: Optional[int] = Field(None, description="Maximum degree of any entity")
    database_size_bytes: Optional[int] = Field(None, description="Size of the database in bytes")
    last_updated: Optional[str] = Field(None, description="When the stats were last updated")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional statistics metadata")


class GraphHealthResult(StrictBaseModel):
    """Result from graph database health check."""
    # Inherits strict configuration from StrictBaseModel

    healthy: bool = Field(..., description="Whether the graph database is healthy")
    connection_status: str = Field(..., description="Status of database connection")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if unhealthy")
    database_info: Dict[str, Any] = Field(default_factory=dict, description="Database version and configuration info")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="When the health check was performed")


class GraphQueryParams(StrictBaseModel):
    """Parameters for graph database queries.
    
    This model provides type-safe parameter passing for native graph queries.
    Each provider can extend this with provider-specific parameters.
    """
    # Inherits strict configuration from StrictBaseModel

    # Common query parameters
    limit: Optional[int] = Field(None, ge=1, description="Maximum number of results to return")
    offset: Optional[int] = Field(None, ge=0, description="Number of results to skip (for pagination)")
    timeout_ms: Optional[int] = Field(None, ge=0, description="Query timeout in milliseconds")

    # Additional parameters as a dict for flexibility
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific parameters")


class Neo4jQueryParams(GraphQueryParams):
    """Neo4j-specific query parameters."""
    # Note: Inherits strict configuration from GraphQueryParams/StrictBaseModel

    # Neo4j-specific parameters
    access_mode: Optional[str] = Field(None, description="Access mode: 'READ' or 'WRITE'")
    database: Optional[str] = Field(None, description="Target database name")
    bookmarks: Optional[List[str]] = Field(None, description="Transaction bookmarks for causal consistency")


class ArangoQueryParams(GraphQueryParams):
    """ArangoDB-specific query parameters."""
    # Note: Inherits strict configuration from GraphQueryParams/StrictBaseModel

    # ArangoDB-specific parameters
    batch_size: Optional[int] = Field(None, ge=1, description="Number of results per batch")
    ttl: Optional[int] = Field(None, ge=0, description="Time-to-live for query cursor in seconds")
    memory_limit: Optional[int] = Field(None, ge=0, description="Memory limit for query in bytes")
    max_runtime: Optional[float] = Field(None, ge=0, description="Maximum query runtime in seconds")


class JanusGraphQueryParams(GraphQueryParams):
    """JanusGraph-specific query parameters."""
    # Note: Inherits strict configuration from GraphQueryParams/StrictBaseModel

    # JanusGraph/Gremlin-specific parameters
    profile: Optional[bool] = Field(False, description="Enable query profiling")
    iterate: Optional[bool] = Field(True, description="Whether to iterate results automatically")
    batch_size: Optional[int] = Field(None, ge=1, description="Gremlin result batch size")


class TraversalParams(StrictBaseModel):
    """Parameters for graph traversal operations.
    
    This model standardizes traversal parameters across providers.
    """
    # Inherits strict configuration from StrictBaseModel

    start_id: str = Field(..., description="ID of the starting entity")
    relation_types: Optional[List[str]] = Field(None, description="Types of relationships to follow")
    max_depth: int = Field(2, ge=1, le=10, description="Maximum traversal depth")
    direction: str = Field("outgoing", pattern="^(outgoing|incoming|both)$", description="Traversal direction")
    include_start: bool = Field(True, description="Whether to include the starting node in results")
    limit_per_level: Optional[int] = Field(None, ge=1, description="Max nodes to visit per depth level")
