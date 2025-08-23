"""Models for entity analysis flow."""

from typing import List, Dict, Any, Optional
from pydantic import Field, ConfigDict
from flowlib.core.models import StrictBaseModel
from enum import Enum


class EntityExtractionStrategy(str, Enum):
    """Strategy for entity extraction."""
    LLM_BASED = "llm_based"
    REGEX_BASED = "regex_based"
    HYBRID = "hybrid"


class LLMEntityExtractionRequest(StrictBaseModel):
    """Request for LLM entity extraction."""
    text: str = Field(description="Text to extract entities from")
    domain: str = Field(description="Technical domain (e.g., avionics, game_dev)")
    context: Optional[str] = Field(None, description="Additional context about the document")


class LLMEntityResult(StrictBaseModel):
    """Single entity extracted by LLM."""
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type")
    description: Optional[str] = Field(None, description="Entity description")
    importance: str = Field("medium", description="Importance level: critical/high/medium/low")


class LLMEntityExtractionResult(StrictBaseModel):
    """Result of LLM entity extraction."""
    entities: List[LLMEntityResult] = Field(
        default_factory=list,
        description="Extracted entities with type, name, description, and importance"
    )


class LLMRelationshipResult(StrictBaseModel):
    """Single relationship extracted by LLM."""
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    type: str = Field(description="Relationship type")
    description: Optional[str] = Field(None, description="Relationship description")
    confidence: float = Field(0.8, ge=0.0, le=1.0, description="Confidence score")


class LLMRelationshipExtractionResult(StrictBaseModel):
    """Result of LLM relationship extraction."""
    relationships: List[LLMRelationshipResult] = Field(
        default_factory=list,
        description="Extracted relationships between entities"
    )


class LLMConceptResult(BaseModel):
    """Key concept extracted by LLM."""
    concept: str = Field(description="Concept name")
    abbreviation: Optional[str] = Field(None, description="Common abbreviation")
    explanation: str = Field(description="Concept explanation")
    importance: str = Field("medium", description="Importance: high/medium/low")
    related_concepts: List[str] = Field(default_factory=list, description="Related concepts")


class LLMConceptExtractionResult(BaseModel):
    """Result of LLM concept extraction."""
    concepts: List[LLMConceptResult] = Field(
        default_factory=list,
        description="Key technical concepts with explanations"
    )