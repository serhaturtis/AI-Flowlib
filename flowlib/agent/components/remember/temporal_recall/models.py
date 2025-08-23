"""Models specific to temporal recall flow."""

from typing import Optional, List
from datetime import datetime
from pydantic import Field
from flowlib.core.models import StrictBaseModel

# Import parent models
from ..models import RecallRequest, RecallResponse, RecallStrategy, MemoryMatch


class TemporalRecallRequest(RecallRequest):
    """Specialized request for temporal recall."""
    time_range_start: Optional[datetime] = Field(None, description="Start of time range for recall")
    time_range_end: Optional[datetime] = Field(None, description="End of time range for recall")
    temporal_relation: Optional[str] = Field(None, description="Temporal relationship (before, after, during)")


class TemporalRecallResponse(RecallResponse):
    """Specialized response for temporal recall."""
    timeline: List[dict] = Field(default_factory=list, description="Timeline of retrieved memories")
    temporal_clusters: List[dict] = Field(default_factory=list, description="Temporal clusters of results")


# Export the models
__all__ = [
    "TemporalRecallRequest",
    "TemporalRecallResponse",
    "RecallRequest",
    "RecallResponse",
    "RecallStrategy", 
    "MemoryMatch"
]