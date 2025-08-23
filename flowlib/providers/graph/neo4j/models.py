"""Strict Pydantic models for Neo4j query results and operations.

No fallbacks, no defaults, no optional fields unless explicitly required.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import Field, ConfigDict
from flowlib.core.models import StrictBaseModel


class Neo4jAttributeData(StrictBaseModel):
    """Strict Neo4j attribute data model."""
    # Inherits strict configuration from StrictBaseModel
    
    name: str = Field(..., description="Attribute name")
    value: str = Field(..., description="Attribute value")
    confidence: float = Field(..., description="Confidence score")
    source: str = Field(..., description="Attribute source")
    timestamp: str = Field(..., description="Timestamp in ISO format")


class Neo4jNodeData(StrictBaseModel):
    """Strict Neo4j node data model."""
    # Inherits strict configuration from StrictBaseModel
    
    id: str = Field(..., description="Node ID")
    type: str = Field(..., description="Node type")
    attributes: Dict[str, Neo4jAttributeData] = Field(..., description="Node attributes")
    tags: List[str] = Field(..., description="Node tags")
    source: str = Field(..., description="Node source")
    importance: float = Field(..., description="Node importance score")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Update timestamp")


class Neo4jRelationshipData(StrictBaseModel):
    """Strict Neo4j relationship data model."""
    # Inherits strict configuration from StrictBaseModel
    
    id: str = Field(..., description="Relationship ID")
    type: str = Field(..., description="Relationship type")
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    properties: Dict[str, Any] = Field(..., description="Relationship properties")
    confidence: float = Field(..., description="Relationship confidence")
    source: str = Field(..., description="Relationship source")
    timestamp: str = Field(..., description="Timestamp")


class Neo4jQueryResult(StrictBaseModel):
    """Strict Neo4j query result model."""
    # Inherits strict configuration from StrictBaseModel
    
    nodes: List[Neo4jNodeData] = Field(..., description="Result nodes")
    relationships: List[Neo4jRelationshipData] = Field(..., description="Result relationships")
    total_count: int = Field(..., description="Total result count")
    query_time: float = Field(..., description="Query execution time")


class Neo4jEntitySearchResult(StrictBaseModel):
    """Strict Neo4j entity search result."""
    # Inherits strict configuration from StrictBaseModel
    
    entity_id: str = Field(..., description="Entity ID")
    entity_type: str = Field(..., description="Entity type")
    score: float = Field(..., description="Search relevance score")
    properties: Dict[str, Any] = Field(..., description="Entity properties")


class Neo4jBatchOperation(StrictBaseModel):
    """Strict Neo4j batch operation model."""
    # Inherits strict configuration from StrictBaseModel
    
    operation_type: str = Field(..., description="Type of operation")
    entity_data: Dict[str, Any] = Field(..., description="Entity data")
    result: Optional[Dict[str, Any]] = Field(None, description="Operation result")
    success: bool = Field(..., description="Whether operation succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class Neo4jConnectionInfo(StrictBaseModel):
    """Strict Neo4j connection information."""
    # Inherits strict configuration from StrictBaseModel
    
    uri: str = Field(..., description="Neo4j URI")
    database: str = Field(..., description="Database name")
    username: str = Field(..., description="Username")
    driver_version: str = Field(..., description="Driver version")
    server_version: str = Field(..., description="Server version")


class Neo4jConstraintInfo(StrictBaseModel):
    """Strict Neo4j constraint information."""
    # Inherits strict configuration from StrictBaseModel
    
    name: str = Field(..., description="Constraint name")
    type: str = Field(..., description="Constraint type")
    entity_type: str = Field(..., description="Entity type")
    properties: List[str] = Field(..., description="Constrained properties")