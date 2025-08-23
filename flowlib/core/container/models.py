"""Strict Pydantic models for container metadata and registry entries.

No fallbacks, no defaults, no optional fields unless explicitly required.
"""

from typing import Any, Callable, Optional
from pydantic import Field, ConfigDict
from flowlib.core.models import StrictBaseModel


class ProviderMetadata(StrictBaseModel):
    """Strict metadata for provider registry entries."""
    # Inherits strict configuration from StrictBaseModel
    
    provider_type: str = Field(..., description="Type of provider (llm, storage, etc)")
    provider_class: str = Field(..., description="Provider class name")
    settings_class: str = Field(..., description="Settings class name")


class ResourceMetadata(StrictBaseModel):
    """Strict metadata for resource registry entries."""
    # Inherits strict configuration from StrictBaseModel
    
    resource_type: str = Field(..., description="Type of resource (config, model, etc)")
    resource_class: str = Field(..., description="Resource class name")


class FlowMetadata(StrictBaseModel):
    """Strict metadata for flow registry entries."""
    # Inherits strict configuration from StrictBaseModel
    
    flow_class: str = Field(..., description="Flow class name")
    flow_category: str = Field(..., description="Category of flow")
    input_type: str = Field(..., description="Expected input type")
    output_type: str = Field(..., description="Expected output type")


class ConfigMetadata(StrictBaseModel):
    """Strict metadata for configuration registry entries."""
    # Inherits strict configuration from StrictBaseModel
    
    config_type: str = Field(..., description="Type of configuration")
    config_class: str = Field(..., description="Configuration class name")


class RegistryEntryData(StrictBaseModel):
    """Strict registry entry model replacing Dict metadata."""
    # Inherits strict configuration from StrictBaseModel
    
    name: str = Field(..., description="Entry name")
    item_type: str = Field(..., description="Type of item")
    
    # Metadata based on item type
    provider_metadata: Optional[ProviderMetadata] = Field(None, description="Provider metadata")
    resource_metadata: Optional[ResourceMetadata] = Field(None, description="Resource metadata") 
    flow_metadata: Optional[FlowMetadata] = Field(None, description="Flow metadata")
    config_metadata: Optional[ConfigMetadata] = Field(None, description="Config metadata")
    
    initialized: bool = Field(default=False, description="Whether entry is initialized")
    
    def get_metadata_for_type(self) -> StrictBaseModel:
        """Get the appropriate metadata based on item type."""
        if self.item_type == "provider":
            if not self.provider_metadata:
                raise ValueError(f"Provider entry {self.name} missing provider_metadata")
            return self.provider_metadata
        elif self.item_type in ("resource", "config"):
            if not self.resource_metadata:
                raise ValueError(f"Resource entry {self.name} missing resource_metadata")
            return self.resource_metadata
        elif self.item_type == "flow":
            if not self.flow_metadata:
                raise ValueError(f"Flow entry {self.name} missing flow_metadata")
            return self.flow_metadata
        else:
            raise ValueError(f"Unknown item type: {self.item_type}")


class ContainerStats(StrictBaseModel):
    """Strict container statistics model."""
    # Inherits strict configuration from StrictBaseModel
    
    total_entries: int = Field(..., description="Total number of entries")
    provider_count: int = Field(..., description="Number of providers")
    resource_count: int = Field(..., description="Number of resources")
    flow_count: int = Field(..., description="Number of flows")
    config_count: int = Field(..., description="Number of configs")
    initialized_count: int = Field(..., description="Number of initialized entries")