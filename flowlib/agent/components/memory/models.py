"""
Memory models for the agent system.

This module provides Pydantic models for memory operations to ensure
consistent, type-safe interactions with memory systems. These models
follow Flowlib's strict typing principles and eliminate metadata abuse.
"""

from typing import Any, Dict, List, Optional, Literal
from datetime import datetime
from pydantic import Field, field_validator
from flowlib.core.models import StrictBaseModel

# Import Entity from the graph models - this is referenced by other modules


class MemoryItemMetadata(StrictBaseModel):
    """Strict Pydantic model for memory item metadata.
    
    This contains true metadata about the memory item (not domain data).
    Domain-specific fields should be in specialized MemoryItem subclasses.
    """
    
    source: str = Field(default="user", description="Source of the memory item")
    item_type: str = Field(default="general", description="Type of memory item")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in the information")
    last_accessed: Optional[datetime] = Field(default=None, description="Last access timestamp")
    access_count: int = Field(default=0, ge=0, description="Number of times accessed")
    related_items: List[str] = Field(default_factory=list, description="Related memory item keys")
    version: int = Field(default=1, ge=1, description="Version number of the item")


class MemorySearchMetadata(StrictBaseModel):
    """Strict Pydantic model for memory search metadata."""
    
    search_query: str = Field(..., description="Original search query")
    search_type: str = Field(default="semantic", description="Type of search performed")
    search_time_ms: float = Field(default=0.0, ge=0.0, description="Search execution time in milliseconds")
    total_results: int = Field(default=0, ge=0, description="Total number of results found")
    result_rank: int = Field(default=1, ge=1, description="Rank of this result in the search")
    filter_applied: List[str] = Field(default_factory=list, description="Filters applied to the search")


class MemoryItem(StrictBaseModel):
    """Base memory item model representing stored information.
    
    Uses proper typed metadata instead of generic Dict[str, Any].
    Domain-specific data should be in specialized subclasses, not in metadata.
    """
    
    key: str = Field(..., min_length=1, description="Unique identifier for this memory item")
    value: Any = Field(..., description="The stored value/content")
    context: str = Field("default", description="Context/namespace for this memory")
    created_at: datetime = Field(default_factory=datetime.now, description="When this memory was created")
    updated_at: Optional[datetime] = Field(None, description="When this memory was last updated")
    metadata: MemoryItemMetadata = Field(default_factory=MemoryItemMetadata, description="Typed metadata about this memory")
    
    def update_value(self, new_value: Any) -> 'MemoryItem':
        """Update the value and timestamp, returning a new instance.
        
        Args:
            new_value: The new value to set
            
        Returns:
            New MemoryItem instance with updated value and timestamp
        """
        return self.model_copy(update={
            "value": new_value,
            "updated_at": datetime.now()
        })


class ExecutionMemoryItem(MemoryItem):
    """Memory item for execution-related data.
    
    Replaces the anti-pattern of metadata.get("type") == "execution_step"
    with proper typed fields.
    """
    
    execution_type: Literal["step", "result", "error", "trace"] = Field(
        ..., description="Type of execution data"
    )
    step_number: Optional[int] = Field(None, description="Step number in execution sequence")
    parent_execution_id: Optional[str] = Field(None, description="Parent execution identifier")


class EntityMemoryItem(MemoryItem):
    """Memory item for entity/knowledge data.
    
    Replaces the anti-pattern of metadata["entity_type"] = "person"
    with proper typed fields.
    """
    
    entity_type: str = Field(..., description="Type of entity (person, organization, concept, etc.)")
    entity_id: str = Field(..., description="Unique identifier for the entity")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Importance score of the entity")


class MemoryStoreRequest(StrictBaseModel):
    """Request model for storing items in memory."""
    
    key: str = Field(..., min_length=1, description="Key to store the memory under")
    value: Any = Field(..., description="Value to store")
    context: Optional[str] = Field(None, description="Context to store the memory in (namespace)")
    ttl: Optional[int] = Field(None, description="Time-to-live in seconds (None = no expiration)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata to store (temporary - will be eliminated)")
    importance: float = Field(0.5, description="Importance of the memory (0.0-1.0)")
    original_importance: Optional[float] = Field(None, description="Original importance value before clamping")
    
    @field_validator('importance', mode='before')
    @classmethod
    def clamp_importance(cls, v: float) -> float:
        """Clamp importance value to valid range [0.0, 1.0]."""
        return max(0.0, min(1.0, v))
    
    def __init__(self, **data: Any) -> None:
        # Store original importance before validation
        if 'importance' in data:
            data['original_importance'] = data['importance']
        super().__init__(**data)


class MemoryRetrieveRequest(StrictBaseModel):
    """Request model for retrieving items from memory."""
    
    key: str = Field(..., min_length=1, description="Key to retrieve")
    context: Optional[str] = Field(None, description="Context to retrieve from")
    default: Optional[Any] = Field(None, description="Default value if key not found")
    metadata_only: bool = Field(False, description="Whether to return only metadata without the value")


class MemorySearchRequest(StrictBaseModel):
    """Request model for searching memory."""
    
    query: str = Field(..., min_length=1, description="Search query (text, embedding, or hybrid)")
    context: Optional[str] = Field(None, description="Context to search in")
    limit: int = Field(10, gt=0, description="Maximum number of results to return")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity threshold (0.0-1.0)")
    sort_by: Optional[Literal["relevance", "created_at", "updated_at"]] = Field(None, description="Field to sort results by")
    search_type: Literal["semantic", "keyword", "hybrid"] = Field("hybrid", description="Type of search performed")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Filter results based on metadata key-value pairs")


class MemorySearchResult(StrictBaseModel):
    """Result model for single memory search result with typed metadata."""
    
    item: MemoryItem = Field(..., description="The memory item found")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score for this result")
    metadata: MemorySearchMetadata = Field(default_factory=lambda: MemorySearchMetadata(search_query=""), description="Search metadata")


class MemorySearchResultCollection(StrictBaseModel):
    """Collection of memory search results."""
    
    items: List[MemorySearchResult] = Field(default_factory=list, description="Matching memory search results")
    total_count: int = Field(0, ge=0, description="Total number of matching items")
    query: str = Field("", description="Original search query")
    context: Optional[str] = Field(None, description="Context that was searched")
    search_time_ms: float = Field(default=0.0, ge=0.0, description="Total search time in milliseconds")


class MemoryContext(StrictBaseModel):
    """Model representing a memory context (namespace)."""
    
    name: str = Field(..., min_length=1, description="Name of the context")
    path: str = Field(..., min_length=1, description="Full path of the context")
    parent: Optional[str] = Field(None, description="Parent context path")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Context metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="When this context was created")