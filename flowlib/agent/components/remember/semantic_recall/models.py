"""Models specific to semantic recall flow."""

from typing import List
from pydantic import Field
from flowlib.core.models import StrictBaseModel

# Import parent models
from ..models import RecallRequest, RecallResponse, RecallStrategy, MemoryMatch


class SemanticAnalysis(StrictBaseModel):
    """Analysis of semantic aspects of a query."""
    
    key_concepts: List[str] = Field(default_factory=list, description="Key concepts identified in the query")
    semantic_relationships: List[str] = Field(default_factory=list, description="Semantic relationships found")
    contextual_meaning: str = Field(..., description="Contextual meaning of the query")
    topic_categories: List[str] = Field(default_factory=list, description="Topic categories the query relates to")
    confidence: float = Field(..., description="Confidence in the semantic analysis (0.0 to 1.0)")


class SemanticRecallRequest(RecallRequest):
    """Specialized request for semantic recall."""
    
    similarity_threshold: float = Field(0.7, description="Minimum similarity threshold for matches")
    include_related_concepts: bool = Field(True, description="Whether to include related concepts in search")


class SemanticRecallResponse(RecallResponse):
    """Specialized response for semantic recall."""
    
    semantic_clusters: List[dict] = Field(default_factory=list, description="Semantic clusters of results")
    concept_coverage: dict = Field(default_factory=dict, description="Coverage of query concepts in results")


# Export the models
__all__ = [
    "SemanticAnalysis",
    "SemanticRecallRequest",
    "SemanticRecallResponse",
    "RecallRequest",
    "RecallResponse", 
    "RecallStrategy",
    "MemoryMatch"
]