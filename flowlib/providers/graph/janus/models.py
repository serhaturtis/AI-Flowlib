"""Strict Pydantic models for JanusGraph query results and operations.

No fallbacks, no defaults, no optional fields unless explicitly required.
"""

from typing import Any, Dict, List

from pydantic import Field

from flowlib.core.models import StrictBaseModel


class JanusAttributeData(StrictBaseModel):
    """Strict JanusGraph attribute data model."""
    # Inherits strict configuration from StrictBaseModel

    name: str = Field(..., description="Attribute name")
    value: str = Field(..., description="Attribute value")
    confidence: float = Field(..., description="Confidence score")
    source: str = Field(..., description="Attribute source")
    timestamp: str = Field(..., description="Timestamp in ISO format")


class JanusEntityData(StrictBaseModel):
    """Strict JanusGraph entity data model."""
    id: str = Field(..., description="Entity ID")
    type: str = Field(..., description="Entity type")
    attributes: Dict[str, JanusAttributeData] = Field(..., description="Entity attributes")
    source: str = Field(..., description="Entity source")
    importance: float = Field(..., description="Entity importance score")
    last_updated: str = Field(..., description="Last update timestamp")

    # Inherits strict configuration from StrictBaseModel


class JanusRelationshipData(StrictBaseModel):
    """Strict JanusGraph relationship data model."""
    relation_type: str = Field(..., description="Relationship type")
    target_id: str = Field(..., description="Target entity ID")
    confidence: float = Field(..., description="Relationship confidence")
    source: str = Field(..., description="Relationship source")
    timestamp: str = Field(..., description="Timestamp")

    # Inherits strict configuration from StrictBaseModel


class JanusVertexData(StrictBaseModel):
    """Strict JanusGraph vertex data model."""
    id: str = Field(..., description="Vertex ID")
    label: str = Field(..., description="Vertex label")
    properties: Dict[str, Any] = Field(..., description="Vertex properties")

    # Inherits strict configuration from StrictBaseModel


class JanusEdgeData(StrictBaseModel):
    """Strict JanusGraph edge data model."""
    id: str = Field(..., description="Edge ID")
    label: str = Field(..., description="Edge label")
    source_id: str = Field(..., description="Source vertex ID")
    target_id: str = Field(..., description="Target vertex ID")
    properties: Dict[str, Any] = Field(..., description="Edge properties")

    # Inherits strict configuration from StrictBaseModel


class JanusQueryResult(StrictBaseModel):
    """Strict JanusGraph query result model."""
    vertices: List[JanusVertexData] = Field(..., description="Result vertices")
    edges: List[JanusEdgeData] = Field(..., description="Result edges")
    total_count: int = Field(..., description="Total result count")
    query_time: float = Field(..., description="Query execution time")

    # Inherits strict configuration from StrictBaseModel


class JanusTraversalResult(StrictBaseModel):
    """Strict JanusGraph traversal result model."""
    result_data: List[Dict[str, Any]] = Field(..., description="Traversal result data")
    execution_time: float = Field(..., description="Execution time in milliseconds")

    # Inherits strict configuration from StrictBaseModel


class JanusConnectionInfo(StrictBaseModel):
    """Strict JanusGraph connection information."""
    url: str = Field(..., description="JanusGraph URL")
    graph_name: str = Field(..., description="Graph name")
    traversal_source: str = Field(..., description="Traversal source name")
    connection_pool_size: int = Field(..., description="Connection pool size")

    # Inherits strict configuration from StrictBaseModel
