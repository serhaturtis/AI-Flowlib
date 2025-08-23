"""Models specific to entity recall flow."""

from typing import Optional
from pydantic import Field
from flowlib.core.models import StrictBaseModel

# Entity recall uses the same models as the parent remember module
# Import them for consistency
from ..models import RecallRequest, RecallResponse, RecallStrategy, MemoryMatch

# If entity recall needs specific models, they would be defined here
# For example:

class EntityRecallRequest(RecallRequest):
    """Specialized request for entity recall with additional entity-specific fields."""
    entity_type: Optional[str] = Field(None, description="Type of entity being recalled")
    include_relationships: bool = Field(True, description="Whether to include entity relationships")


class EntityRecallResponse(RecallResponse):
    """Specialized response for entity recall with additional entity-specific data."""
    entity_properties: dict = Field(default_factory=dict, description="Entity properties retrieved")
    relationship_count: int = Field(0, description="Number of relationships found")

# Export the models
__all__ = [
    "EntityRecallRequest",
    "EntityRecallResponse",
    "RecallRequest", 
    "RecallResponse",
    "RecallStrategy",
    "MemoryMatch"
]