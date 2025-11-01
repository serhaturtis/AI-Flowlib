"""Models for graph storage flow."""

from typing import Any

from pydantic import BaseModel, Field

from flowlib.knowledge.models import (
    DocumentContent,
    Entity,
    GraphEdge,
    GraphNode,
    GraphStatistics,
    GraphStoreInput,
    GraphStoreOutput,
    Relationship,
)


class GraphEntityAttribute(BaseModel):
    """Attribute for a graph entity."""

    name: str = Field(..., description="Attribute name")
    value: Any = Field(..., description="Attribute value")
    type: str = Field(default="string", description="Attribute type")


class GraphEntity(BaseModel):
    """Entity representation for graph database."""

    id: str = Field(..., description="Entity ID")
    type: str = Field(..., description="Entity type")
    label: str = Field(..., description="Entity label")
    attributes: list[GraphEntityAttribute] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class GraphRelationship(BaseModel):
    """Relationship representation for graph database."""

    id: str = Field(..., description="Relationship ID")
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    type: str = Field(..., description="Relationship type")
    label: str = Field(..., description="Relationship label")
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, description="Confidence score")


__all__ = [
    "GraphStoreInput",
    "GraphStoreOutput",
    "GraphNode",
    "GraphEdge",
    "GraphStatistics",
    "Entity",
    "Relationship",
    "DocumentContent",
    # Graph-specific models
    "GraphEntityAttribute",
    "GraphEntity",
    "GraphRelationship",
]
