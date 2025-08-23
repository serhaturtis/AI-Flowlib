"""
Models for agent decorators component.

This module contains Pydantic models used by the decorators component.
"""

from typing import Dict, Any, Optional, List
from pydantic import Field
from flowlib.core.models import StrictBaseModel


class DecoratorConfig(StrictBaseModel):
    """Configuration for agent decorators."""
    
    enabled: bool = Field(default=True, description="Whether decorators are enabled")
    registry_name: str = Field(default="default", description="Registry name to use")


class AgentRegistrationData(StrictBaseModel):
    """Data model for agent registration."""
    
    agent_id: str = Field(description="Unique agent identifier")
    agent_type: str = Field(description="Type of agent (e.g., 'dual_path')")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DecoratorResult(StrictBaseModel):
    """Result from decorator operations."""
    model_config = ConfigDict(extra="forbid")
    
    success: bool = Field(description="Whether operation was successful")
    agent_id: Optional[str] = Field(default=None, description="Agent ID if applicable")
    message: Optional[str] = Field(default=None, description="Result message")
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional result data")