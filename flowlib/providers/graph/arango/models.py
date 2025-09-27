"""Strict Pydantic models for ArangoDB query results and operations.

No fallbacks, no defaults, no optional fields unless explicitly required.
"""

from typing import Any, Dict, List, Optional
from pydantic import Field
from flowlib.core.models import StrictBaseModel


class ArangoAttributeData(StrictBaseModel):
    """Strict ArangoDB attribute data model."""
    value: str = Field(..., description="Attribute value")
    confidence: float = Field(..., description="Confidence score")
    source: str = Field(..., description="Attribute source")
    timestamp: str = Field(..., description="Timestamp in ISO format")
    
    # Inherits strict configuration from StrictBaseModel


class ArangoDocumentData(StrictBaseModel):
    """Strict ArangoDB document data model."""
    id: str = Field(..., description="Document ID")
    type: str = Field(..., description="Document type")
    attributes: Dict[str, ArangoAttributeData] = Field(..., description="Document attributes")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    source: str = Field(..., description="Document source")
    importance: float = Field(..., description="Document importance score")
    last_updated: str = Field(..., description="Last update timestamp")
    
    # Inherits strict configuration from StrictBaseModel


class ArangoRelationshipData(StrictBaseModel):
    """Strict ArangoDB relationship data model."""
    relation_type: str = Field(..., description="Relationship type")
    target_id: str = Field(..., description="Target entity ID")
    confidence: float = Field(..., description="Relationship confidence")
    source: str = Field(..., description="Relationship source")
    timestamp: str = Field(..., description="Timestamp")
    
    # Inherits strict configuration from StrictBaseModel


class ArangoQueryResult(StrictBaseModel):
    """Strict ArangoDB query result model."""
    documents: List[ArangoDocumentData] = Field(..., description="Result documents")
    total_count: int = Field(..., description="Total result count")
    query_time: float = Field(..., description="Query execution time")
    
    # Inherits strict configuration from StrictBaseModel


class ArangoEntitySearchResult(StrictBaseModel):
    """Strict ArangoDB entity search result."""
    entity_id: str = Field(..., description="Entity ID")
    entity_type: str = Field(..., description="Entity type")
    score: float = Field(..., description="Search relevance score")
    properties: Dict[str, Any] = Field(..., description="Entity properties")
    
    # Inherits strict configuration from StrictBaseModel


class ArangoBatchOperation(StrictBaseModel):
    """Strict ArangoDB batch operation model."""
    operation_type: str = Field(..., description="Type of operation")
    entity_data: Dict[str, Any] = Field(..., description="Entity data")
    result: Optional[Dict[str, Any]] = Field(None, description="Operation result")
    success: bool = Field(..., description="Whether operation succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    # Inherits strict configuration from StrictBaseModel


class ArangoConnectionInfo(StrictBaseModel):
    """Strict ArangoDB connection information."""
    url: str = Field(..., description="ArangoDB URL")
    database: str = Field(..., description="Database name")
    username: str = Field(..., description="Username")
    version: str = Field(..., description="Server version")
    
    # Inherits strict configuration from StrictBaseModel


class ArangoCollectionInfo(StrictBaseModel):
    """Strict ArangoDB collection information."""
    name: str = Field(..., description="Collection name")
    type: str = Field(..., description="Collection type")
    status: str = Field(..., description="Collection status")
    count: int = Field(..., description="Document count")
    
    # Inherits strict configuration from StrictBaseModel