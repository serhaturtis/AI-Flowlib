"""Knowledge component models - Single source of truth.

This module defines all models for the unified knowledge component following
flowlib's strict Pydantic contract principles.
"""

from enum import Enum
from typing import Any

from pydantic import Field

from flowlib.core.models import MutableStrictBaseModel, StrictBaseModel


class KnowledgeType(str, Enum):
    """Types of knowledge that can be processed."""

    ENTITY = "entity"
    CONCEPT = "concept"
    RELATIONSHIP = "relationship"
    PATTERN = "pattern"
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    PERSONAL = "personal"


class ConfidenceLevel(str, Enum):
    """Standard confidence levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class Entity(MutableStrictBaseModel):
    """Entity representation with strict validation."""

    name: str = Field(..., min_length=1, description="Entity name")
    type: str = Field(
        ..., min_length=1, description="Entity type (person, organization, location, etc.)"
    )
    description: str | None = Field(None, description="Entity description")
    properties: dict[str, Any] = Field(default_factory=dict, description="Entity properties")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    source_context: str = Field(..., description="Source context where entity was found")
    aliases: list[str] = Field(default_factory=list, description="Alternative names")


class Concept(MutableStrictBaseModel):
    """Concept representation with strict validation."""

    name: str = Field(..., min_length=1, description="Concept name")
    definition: str = Field(..., min_length=1, description="Concept definition")
    category: str = Field(..., min_length=1, description="Concept category")
    related_terms: list[str] = Field(default_factory=list, description="Related terminology")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    source_context: str = Field(..., description="Source context")


class Relationship(MutableStrictBaseModel):
    """Relationship representation with strict validation."""

    source: str = Field(..., min_length=1, description="Source entity/concept")
    target: str = Field(..., min_length=1, description="Target entity/concept")
    relationship_type: str = Field(..., min_length=1, description="Type of relationship")
    description: str = Field(..., description="Relationship description")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    source_context: str = Field(..., description="Source context")


class Pattern(MutableStrictBaseModel):
    """Pattern representation with strict validation."""

    name: str = Field(..., min_length=1, description="Pattern name")
    description: str = Field(..., min_length=1, description="Pattern description")
    frequency: int = Field(..., ge=1, description="Pattern frequency")
    examples: list[str] = Field(default_factory=list, description="Pattern examples")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    source_context: str = Field(..., description="Source context")


class KnowledgeSet(StrictBaseModel):
    """Collection of knowledge items with strict validation."""

    entities: list[Entity] = Field(default_factory=list, description="Extracted entities")
    concepts: list[Concept] = Field(default_factory=list, description="Extracted concepts")
    relationships: list[Relationship] = Field(
        default_factory=list, description="Extracted relationships"
    )
    patterns: list[Pattern] = Field(default_factory=list, description="Extracted patterns")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Knowledge set metadata")

    @property
    def total_items(self) -> int:
        """Total number of knowledge items."""
        return (
            len(self.entities) + len(self.concepts) + len(self.relationships) + len(self.patterns)
        )

    @property
    def is_empty(self) -> bool:
        """Check if knowledge set is empty."""
        return self.total_items == 0


class LearningInput(StrictBaseModel):
    """Input for knowledge learning operations."""

    content: str = Field(..., min_length=1, description="Content to learn from")
    context: str = Field(..., description="Context information")
    focus_areas: list[str] = Field(default_factory=list, description="Areas to focus extraction on")
    domain_hint: str | None = Field(None, description="Domain hint for extraction")


class LearningResult(StrictBaseModel):
    """Result of knowledge learning operations."""

    success: bool = Field(..., description="Whether learning succeeded")
    knowledge: KnowledgeSet = Field(..., description="Extracted knowledge")
    processing_time_seconds: float = Field(..., ge=0.0, description="Processing time")
    message: str = Field(..., description="Result message")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Processing metadata")


class StorageRequest(StrictBaseModel):
    """Request for knowledge storage operations."""

    knowledge: KnowledgeSet = Field(..., description="Knowledge to store")
    context_id: str = Field(..., min_length=1, description="Context identifier")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Storage metadata")


class RetrievalRequest(StrictBaseModel):
    """Request for knowledge retrieval operations."""

    query: str = Field(..., min_length=1, description="Search query")
    knowledge_types: list[KnowledgeType] = Field(
        default_factory=list, description="Types to retrieve"
    )
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")
    context_filter: str | None = Field(None, description="Context filter")


class RetrievalResult(StrictBaseModel):
    """Result of knowledge retrieval operations."""

    knowledge: KnowledgeSet = Field(..., description="Retrieved knowledge")
    query: str = Field(..., description="Original query")
    relevance_scores: dict[str, float] = Field(default_factory=dict, description="Relevance scores")
    total_found: int = Field(..., ge=0, description="Total items found")


class KnowledgeComponentConfig(StrictBaseModel):
    """Configuration for knowledge component."""

    enable_storage: bool = Field(default=True, description="Enable storage operations")
    enable_retrieval: bool = Field(default=True, description="Enable retrieval operations")
    llm_config: str = Field(default="default-llm", description="LLM configuration name")
    vector_db_config: str = Field(
        default="default-vector-db", description="Vector DB configuration name"
    )
    graph_db_config: str = Field(
        default="default-graph-db", description="Graph DB configuration name"
    )
    learning_batch_size: int = Field(default=100, ge=1, description="Learning batch size")
    max_storage_items: int = Field(default=10000, ge=1, description="Maximum items to store")
