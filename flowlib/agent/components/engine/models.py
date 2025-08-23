"""
Models for agent execution engine component.

This module contains Pydantic models used by the agent execution engine.
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import Field, ConfigDict
from enum import Enum
from datetime import datetime
from flowlib.core.models import StrictBaseModel


class ExecutionMode(str, Enum):
    """Execution mode for the engine."""
    SINGLE_CYCLE = "single_cycle"
    TODO_DRIVEN = "todo_driven"
    CONTINUOUS = "continuous"


class ExecutionStatus(str, Enum):
    """Status of execution operations."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class EngineConfig(StrictBaseModel):
    """Configuration for the agent execution engine."""
    # Inherits strict configuration from StrictBaseModel
    
    execution_mode: ExecutionMode = Field(default=ExecutionMode.SINGLE_CYCLE, description="Default execution mode")
    max_cycles: int = Field(default=10, description="Maximum execution cycles")
    cycle_timeout: float = Field(default=300.0, description="Timeout per cycle in seconds")
    enable_reflection: bool = Field(default=True, description="Whether to enable reflection")
    enable_planning: bool = Field(default=True, description="Whether to enable planning")


class ExecutionRequest(StrictBaseModel):
    """Request for engine execution."""
    # Inherits strict configuration from StrictBaseModel
    
    task_description: str = Field(description="Description of task to execute")
    execution_mode: Optional[ExecutionMode] = Field(default=None, description="Override execution mode")
    max_cycles: Optional[int] = Field(default=None, description="Override max cycles")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional execution context")


class ExecutionStep(StrictBaseModel):
    """Information about a single execution step."""
    # Inherits strict configuration from StrictBaseModel
    
    step_id: str = Field(description="Unique step identifier")
    flow_name: str = Field(description="Name of flow executed")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Inputs to the flow")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Outputs from the flow")
    status: ExecutionStatus = Field(description="Step execution status")
    start_time: Optional[datetime] = Field(default=None, description="Step start time")
    end_time: Optional[datetime] = Field(default=None, description="Step end time")
    duration: Optional[float] = Field(default=None, description="Step duration in seconds")
    error: Optional[str] = Field(default=None, description="Error message if step failed")


class ExecutionResult(StrictBaseModel):
    """Result of engine execution."""
    # Inherits strict configuration from StrictBaseModel
    
    status: ExecutionStatus = Field(description="Overall execution status")
    steps: List[ExecutionStep] = Field(default_factory=list, description="Execution steps")
    total_cycles: int = Field(default=0, description="Total cycles executed")
    total_duration: Optional[float] = Field(default=None, description="Total execution time")
    final_output: Dict[str, Any] = Field(default_factory=dict, description="Final execution output")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EngineStats(StrictBaseModel):
    """Engine performance statistics."""
    # Inherits strict configuration from StrictBaseModel
    
    total_executions: int = Field(default=0, description="Total number of executions")
    successful_executions: int = Field(default=0, description="Number of successful executions")
    failed_executions: int = Field(default=0, description="Number of failed executions")
    average_duration: float = Field(default=0.0, description="Average execution duration")
    average_cycles: float = Field(default=0.0, description="Average cycles per execution")
    last_execution: Optional[datetime] = Field(default=None, description="Last execution timestamp")