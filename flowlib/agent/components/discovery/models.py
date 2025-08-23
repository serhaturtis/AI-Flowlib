"""
Models for flow discovery component.

This module contains Pydantic models used by the flow discovery system.
"""

from typing import Dict, Any, Optional, List, Set
from pydantic import Field
from flowlib.core.models import StrictBaseModel
from enum import Enum


class DiscoveryStatus(str, Enum):
    """Status of discovery operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class FlowDiscoveryConfig(StrictBaseModel):
    """Configuration for flow discovery."""
    
    enabled: bool = Field(default=True, description="Whether discovery is enabled")
    scan_paths: List[str] = Field(default_factory=list, description="Paths to scan for flows")
    exclude_patterns: List[str] = Field(default_factory=list, description="Patterns to exclude")
    cache_enabled: bool = Field(default=True, description="Whether to cache results")


class FlowDiscoveryRequest(StrictBaseModel):
    """Request for flow discovery operations."""
    
    scan_paths: Optional[List[str]] = Field(default=None, description="Specific paths to scan")
    force_rescan: bool = Field(default=False, description="Force rescan even if cached")
    flow_patterns: Optional[List[str]] = Field(default=None, description="Specific flow patterns to find")


class DiscoveredFlow(StrictBaseModel):
    """Information about a discovered flow."""
    
    flow_name: str = Field(description="Name of the discovered flow")
    flow_class: str = Field(description="Full class name")
    module_path: str = Field(description="Module path where flow is defined")
    file_path: str = Field(description="File path containing the flow")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Flow metadata")
    dependencies: List[str] = Field(default_factory=list, description="Flow dependencies")


class FlowDiscoveryResult(StrictBaseModel):
    """Result of flow discovery operations."""
    
    status: DiscoveryStatus = Field(description="Discovery status")
    discovered_flows: List[DiscoveredFlow] = Field(default_factory=list, description="Found flows")
    scan_paths: List[str] = Field(default_factory=list, description="Paths that were scanned")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    total_flows: int = Field(default=0, description="Total number of flows discovered")
    scan_duration: Optional[float] = Field(default=None, description="Time taken for scan in seconds")