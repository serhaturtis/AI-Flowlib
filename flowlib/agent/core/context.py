"""
Flow execution context system for clean dependency injection.

This module provides a clean way to inject providers and configuration
into flows without polluting input models with configuration concerns.
"""

from typing import Dict, Any, Optional, Protocol
from pydantic import BaseModel, Field, ConfigDict
from dataclasses import dataclass

from flowlib.providers.core.registry import provider_registry
from flowlib.resources.registry.registry import resource_registry


class FlowProvider(Protocol):
    """Protocol for providers that can be injected into flows."""
    
    async def initialize(self) -> None:
        """Initialize the provider if needed."""
        ...


@dataclass
class FlowContext:
    """Execution context for flows with automatic provider resolution.
    
    This class provides clean dependency injection for flows without
    polluting input models with configuration concerns.
    """
    
    # Provider instances (lazy-loaded)
    _llm_provider: Optional[FlowProvider] = None
    _graph_provider: Optional[FlowProvider] = None  
    _vector_provider: Optional[FlowProvider] = None
    _cache_provider: Optional[FlowProvider] = None
    
    # Processing configuration
    confidence_threshold: float = 0.7
    model_preference: str = "balanced"  # "fast", "balanced", "quality"
    
    async def llm(self) -> FlowProvider:
        """Get LLM provider using config-driven resolution."""
        if self._llm_provider is None:
            config_name = self._get_llm_config_name()
            self._llm_provider = await provider_registry.get_by_config(config_name)
        return self._llm_provider
    
    async def graph(self) -> FlowProvider:
        """Get graph database provider using config-driven resolution."""
        if self._graph_provider is None:
            self._graph_provider = await provider_registry.get_by_config("default-graph-db")
        return self._graph_provider
    
    async def vector(self) -> FlowProvider:
        """Get vector database provider using config-driven resolution."""
        if self._vector_provider is None:
            self._vector_provider = await provider_registry.get_by_config("default-vector-db")
        return self._vector_provider
    
    async def cache(self) -> FlowProvider:
        """Get cache provider using config-driven resolution."""
        if self._cache_provider is None:
            self._cache_provider = await provider_registry.get_by_config("default-cache")
        return self._cache_provider
    
    def _get_llm_config_name(self) -> str:
        """Get appropriate LLM config based on model preference."""
        if self.model_preference == "fast":
            return "fast-llm"
        elif self.model_preference == "quality":
            return "quality-llm"
        else:  # balanced
            return "default-llm"


class ProcessingOptions(BaseModel):
    """Optional processing configuration that can be passed to flows.
    
    This replaces the configuration fields that were scattered throughout
    input models, providing clean defaults and clear documentation.
    """
    
    # Quality settings
    confidence_threshold: float = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0,
        description="Minimum confidence threshold for results"
    )
    
    # Performance settings  
    model_preference: str = Field(
        default="balanced",
        description="Model preference: 'fast' (speed), 'balanced' (default), 'quality' (accuracy)"
    )
    
    # Limits
    max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return"
    )
    
    # Timeout settings
    timeout_seconds: int = Field(
        default=60,
        ge=1,
        le=300,
        description="Maximum processing time in seconds"
    )
    
    def create_context(self) -> FlowContext:
        """Create a flow context from these processing options."""
        return FlowContext(
            confidence_threshold=self.confidence_threshold,
            model_preference=self.model_preference
        )


# Default context for flows that don't need custom configuration
DEFAULT_CONTEXT = FlowContext()


def get_default_processing_options() -> ProcessingOptions:
    """Get default processing options for flows."""
    return ProcessingOptions()


async def create_flow_context(
    processing_options: Optional[ProcessingOptions] = None
) -> FlowContext:
    """Create a flow context with optional processing configuration.
    
    Args:
        processing_options: Optional processing configuration
        
    Returns:
        FlowContext ready for use in flows
    """
    if processing_options is None:
        processing_options = get_default_processing_options()
    
    return processing_options.create_context()