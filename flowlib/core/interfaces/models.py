"""Strict Pydantic models for interface contracts.

Replaces all Dict[str, Any] patterns in interfaces with typed models.
"""

from typing import Any, Optional, List
from pydantic import Field, ConfigDict
from flowlib.core.models import StrictBaseModel


class VectorMetadata(StrictBaseModel):
    """Strict metadata model for vector operations."""
    # Inherits strict configuration from StrictBaseModel
    
    source: str = Field(description="Source of the vector data")
    document_id: str = Field(description="Document identifier")
    chunk_id: Optional[str] = Field(default=None, description="Chunk identifier if applicable")
    timestamp: str = Field(description="ISO timestamp when vector was created")
    confidence: float = Field(default=1.0, description="Confidence score for the vector")


class VectorSearchResult(StrictBaseModel):
    """Strict search result model for vector operations."""
    # Inherits strict configuration from StrictBaseModel
    
    id: str = Field(description="Vector identifier")
    score: float = Field(description="Similarity score")
    document: str = Field(description="Document content")
    metadata: VectorMetadata = Field(description="Vector metadata")


class EntityData(StrictBaseModel):
    """Strict entity data model for graph operations."""
    # Inherits strict configuration from StrictBaseModel
    
    entity_type: str = Field(description="Type of entity")
    entity_id: str = Field(description="Unique entity identifier")
    name: str = Field(description="Entity name")
    description: Optional[str] = Field(default=None, description="Entity description")
    attributes: StrictBaseModel = Field(description="Entity attributes as structured model")
    tags: List[str] = Field(default_factory=list, description="Entity tags")
    source: str = Field(description="Source of entity data")
    confidence: float = Field(default=1.0, description="Confidence in entity data")


class RelationshipProperties(StrictBaseModel):
    """Strict relationship properties model."""
    # Inherits strict configuration from StrictBaseModel
    
    relationship_type: str = Field(description="Type of relationship")
    confidence: float = Field(default=1.0, description="Confidence in relationship")
    source: str = Field(description="Source of relationship data")
    timestamp: str = Field(description="ISO timestamp when relationship was created")
    bidirectional: bool = Field(default=False, description="Whether relationship is bidirectional")


class QueryParameters(StrictBaseModel):
    """Strict query parameters model."""
    # Inherits strict configuration from StrictBaseModel
    
    query: str = Field(description="Query string")
    limit: Optional[int] = Field(default=None, description="Maximum number of results")
    offset: Optional[int] = Field(default=0, description="Result offset for pagination")
    filters: Optional[StrictBaseModel] = Field(default=None, description="Query filters as structured model")


class QueryResult(StrictBaseModel):
    """Strict query result model."""
    # Inherits strict configuration from StrictBaseModel
    
    success: bool = Field(description="Whether query succeeded")
    result_count: int = Field(description="Number of results returned")
    data: StrictBaseModel = Field(description="Query result data as structured model")
    execution_time_ms: Optional[float] = Field(default=None, description="Query execution time")
    query_id: Optional[str] = Field(default=None, description="Query identifier for tracking")


class ComponentMetadata(StrictBaseModel):
    """Strict component metadata model."""
    # Inherits strict configuration from StrictBaseModel
    
    component_type: str = Field(description="Type of component")
    component_name: str = Field(description="Component name")
    version: str = Field(description="Component version")
    capabilities: List[str] = Field(description="List of component capabilities")
    dependencies: List[str] = Field(default_factory=list, description="Component dependencies")
    status: str = Field(default="unknown", description="Component status")


class ComponentSettings(StrictBaseModel):
    """Strict component settings model."""
    # Inherits strict configuration from StrictBaseModel
    
    component_name: str = Field(description="Component name")
    settings_data: StrictBaseModel = Field(description="Settings as structured model")
    last_updated: str = Field(description="ISO timestamp of last update")
    is_valid: bool = Field(description="Whether settings are valid")


class SchemaDefinition(StrictBaseModel):
    """Strict schema definition model."""
    # Inherits strict configuration from StrictBaseModel
    
    schema_name: str = Field(description="Schema name")
    schema_type: str = Field(description="Schema type (input/output)")
    model_class: str = Field(description="Pydantic model class name")
    fields: List[str] = Field(description="List of field names")
    required_fields: List[str] = Field(description="List of required field names")
    is_valid: bool = Field(description="Whether schema is valid")