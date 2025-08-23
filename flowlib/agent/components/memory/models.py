"""
Memory models for the agent system.

This module provides Pydantic models for memory operations to ensure
consistent, type-safe interactions with memory systems.
"""

from typing import Any, Dict, List, Optional, Union, Literal
from datetime import datetime
from pydantic import Field, model_validator, field_validator, ConfigDict
from flowlib.core.models import StrictBaseModel, MutableStrictBaseModel

# Import Entity from the graph models - this is referenced by other modules
from flowlib.providers.graph.models import Entity


class MemoryItem(StrictBaseModel):
    """Base memory item model representing stored information."""
    # Inherits strict configuration from StrictBaseModel
    
    key: str = Field(..., min_length=1, description="Unique identifier for this memory item")
    value: Any = Field(..., description="The stored value/content")
    context: str = Field("default", description="Context/namespace for this memory")
    created_at: datetime = Field(default_factory=datetime.now, description="When this memory was created")
    updated_at: Optional[datetime] = Field(None, description="When this memory was last updated")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about this memory")
    
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


class MemoryStoreRequest(StrictBaseModel):
    """Request model for storing items in memory."""
    # Inherits strict configuration from StrictBaseModel
    
    key: str = Field(..., min_length=1, description="Key to store the memory under")
    value: Any = Field(..., description="Value to store")
    context: Optional[str] = Field(None, description="Context to store the memory in (namespace)")
    ttl: Optional[int] = Field(None, description="Time-to-live in seconds (None = no expiration)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata to store")
    importance: float = Field(0.5, description="Importance of the memory (0.0-1.0)")
    original_importance: Optional[float] = Field(None, description="Original importance value before clamping")
    
    @field_validator('importance', mode='before')
    @classmethod
    def clamp_importance(cls, v: float) -> float:
        """Clamp importance value to valid range [0.0, 1.0]."""
        return max(0.0, min(1.0, v))
    
    def __init__(self, **data):
        # Store original importance before validation
        if 'importance' in data:
            data['original_importance'] = data['importance']
        super().__init__(**data)


class MemoryRetrieveRequest(StrictBaseModel):
    """Request model for retrieving items from memory."""
    # Inherits strict configuration from StrictBaseModel
    
    key: str = Field(..., min_length=1, description="Key to retrieve")
    context: Optional[str] = Field(None, description="Context to retrieve from")
    default: Optional[Any] = Field(None, description="Default value if key not found")
    metadata_only: bool = Field(False, description="Whether to return only metadata without the value")


class MemorySearchRequest(StrictBaseModel):
    """Request model for searching memory."""
    # Inherits strict configuration from StrictBaseModel
    
    query: str = Field(..., min_length=1, description="Search query (text, embedding, or hybrid)")
    context: Optional[str] = Field(None, description="Context to search in")
    limit: int = Field(10, gt=0, description="Maximum number of results to return")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity threshold (0.0-1.0)")
    sort_by: Optional[Literal["relevance", "created_at", "updated_at"]] = Field(None, description="Field to sort results by (relevance, created_at, updated_at)")
    search_type: Literal["semantic", "keyword", "hybrid"] = Field("hybrid", description="Type of search: 'semantic', 'keyword', or 'hybrid'")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Filter results based on metadata key-value pairs")


class MemorySearchResult(StrictBaseModel):
    """Result model for memory search operations."""
    # Inherits strict configuration from StrictBaseModel
    
    items: List[MemoryItem] = Field(default_factory=list, description="Matching memory items")
    count: int = Field(0, ge=0, description="Total number of matching items")
    query: str = Field("", description="Original search query")
    context: Optional[str] = Field(None, description="Context that was searched")


class MemoryContext(StrictBaseModel):
    """Model representing a memory context (namespace)."""
    # Inherits strict configuration from StrictBaseModel
    
    name: str = Field(..., min_length=1, description="Name of the context")
    path: str = Field(..., min_length=1, description="Full path of the context")
    parent: Optional[str] = Field(None, description="Parent context path")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Context metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="When this context was created")
