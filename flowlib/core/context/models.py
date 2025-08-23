"""Strict Pydantic models for context state management.

No fallbacks, no Dict[str, Any] patterns, strict contracts only.
"""

from typing import Any
from pydantic import Field, ConfigDict
from flowlib.core.models import StrictBaseModel


class ContextSnapshot(StrictBaseModel):
    """Strict snapshot model for context state."""
    # Inherits strict configuration from StrictBaseModel
    
    snapshot_id: int = Field(description="Unique snapshot identifier")
    data: StrictBaseModel = Field(description="Snapshot of model state")
    timestamp: str = Field(description="ISO timestamp when snapshot was created")
    
    @classmethod
    def create(cls, snapshot_id: int, model_data: StrictBaseModel) -> 'ContextSnapshot':
        """Create a new context snapshot."""
        from datetime import datetime
        return cls(
            snapshot_id=snapshot_id,
            data=model_data,
            timestamp=datetime.now().isoformat()
        )


class ContextData(StrictBaseModel):
    """Strict context data model."""
    # Inherits strict configuration from StrictBaseModel
    
    model_data: StrictBaseModel = Field(description="The actual model data")
    model_type_name: str = Field(description="Name of the model type")
    
    @classmethod
    def create(cls, model_data: StrictBaseModel) -> 'ContextData':
        """Create context data from model."""
        return cls(
            model_data=model_data,
            model_type_name=model_data.__class__.__name__
        )


class ContextMergeRequest(StrictBaseModel):
    """Strict model for context merge operations."""
    # Inherits strict configuration from StrictBaseModel
    
    primary_context: ContextData = Field(description="Primary context data")
    secondary_context: ContextData = Field(description="Secondary context to merge")
    merge_strategy: str = Field(default="overwrite", description="Merge strategy to use")
    
    @classmethod
    def create(cls, primary: StrictBaseModel, secondary: StrictBaseModel, strategy: str = "overwrite") -> 'ContextMergeRequest':
        """Create merge request from two models."""
        return cls(
            primary_context=ContextData.create(primary),
            secondary_context=ContextData.create(secondary),
            merge_strategy=strategy
        )


class ContextState(StrictBaseModel):
    """Strict model for context internal state."""
    # Inherits strict configuration from StrictBaseModel
    
    has_data: bool = Field(description="Whether context contains data")
    model_type_name: str = Field(description="Name of the model type")
    snapshot_count: int = Field(description="Number of snapshots stored")
    last_modified: str = Field(description="ISO timestamp of last modification")
    
    @classmethod
    def create(cls, has_data: bool, model_type_name: str, snapshot_count: int) -> 'ContextState':
        """Create context state snapshot."""
        from datetime import datetime
        return cls(
            has_data=has_data,
            model_type_name=model_type_name,
            snapshot_count=snapshot_count,
            last_modified=datetime.now().isoformat()
        )