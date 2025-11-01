"""
Models for agent execution engine component.

This module contains Pydantic models used by the agent execution engine.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import Field

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


class ExecutionContext(StrictBaseModel):
    """Typed execution context replacing Dict[str, Any] context fields."""

    # Request identification
    request_id: str | None = Field(default=None, description="Unique request identifier")
    session_id: str | None = Field(default=None, description="Session identifier")
    user_id: str | None = Field(default=None, description="User identifier")

    # Execution environment
    working_directory: str = Field(default=".", description="Working directory")
    environment_type: Literal["development", "testing", "production"] = Field(
        default="development", description="Execution environment"
    )

    # Agent configuration
    agent_name: str | None = Field(default=None, description="Agent name/identifier")
    agent_version: str | None = Field(default=None, description="Agent version")

    # Performance settings
    max_execution_time: float | None = Field(default=300.0, description="Maximum execution time")
    max_memory_usage: int | None = Field(default=None, description="Maximum memory usage in MB")

    # Feature flags
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    verbose_logging: bool = Field(default=False, description="Enable verbose logging")

    # Resource constraints
    max_llm_calls: int | None = Field(default=None, description="Maximum LLM calls allowed")
    max_tool_calls: int | None = Field(default=None, description="Maximum tool calls allowed")

    # Escape hatch for additional context
    metadata: dict[str, str] = Field(default_factory=dict, description="Additional string metadata")


class FlowInput(StrictBaseModel):
    """Typed inputs for flow execution replacing Dict[str, Any]."""

    # Core task data
    task_description: str = Field(..., description="Task to execute")
    user_message: str | None = Field(default=None, description="Original user message")

    # Context information
    working_directory: str | None = Field(default=None, description="Working directory")
    session_id: str | None = Field(default=None, description="Session identifier")

    # Agent context
    agent_persona: str | None = Field(default=None, description="Agent persona/personality")
    conversation_history: list[dict[str, Any]] | None = Field(
        default_factory=list, description="Conversation history (legacy format)"
    )

    # Execution parameters
    max_cycles: int | None = Field(default=None, description="Maximum execution cycles")
    timeout: float | None = Field(default=None, description="Execution timeout in seconds")

    # Tool and flow context
    available_tools: list[str] | None = Field(
        default_factory=list, description="Available tool names"
    )
    flow_name: str | None = Field(default=None, description="Target flow name")

    # Escape hatch for flow-specific parameters
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Additional string metadata for specific flows"
    )


class FlowOutput(StrictBaseModel):
    """Typed outputs from flow execution replacing Dict[str, Any]."""

    # Execution results
    success: bool = Field(description="Whether execution succeeded")
    result: str | None = Field(default=None, description="Primary execution result")
    error_message: str | None = Field(default=None, description="Error message if failed")

    # Generated content
    response: str | None = Field(default=None, description="Agent response/output")
    generated_tasks: list[str] | None = Field(
        default_factory=list, description="Generated task list"
    )

    # Execution metadata
    execution_time_ms: float | None = Field(
        default=None, description="Execution time in milliseconds"
    )
    cycles_executed: int | None = Field(default=None, description="Number of cycles executed")
    token_usage: int | None = Field(default=None, description="LLM token usage")

    # Tool execution results
    tools_called: list[str] | None = Field(
        default_factory=list, description="Tools that were called"
    )
    tool_results: list[str] | None = Field(
        default_factory=list, description="Results from tool calls"
    )

    # State information
    final_state: str | None = Field(default=None, description="Final execution state")
    intermediate_results: list[str] | None = Field(
        default_factory=list, description="Intermediate execution results"
    )

    # Escape hatch for flow-specific outputs
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Additional string metadata for specific outputs"
    )


class EngineConfig(StrictBaseModel):
    """Configuration for the agent execution engine."""

    # Inherits strict configuration from StrictBaseModel

    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.SINGLE_CYCLE, description="Default execution mode"
    )
    max_cycles: int = Field(default=1, description="Maximum execution cycles")
    cycle_timeout: float = Field(default=300.0, description="Timeout per cycle in seconds")


class ExecutionRequest(StrictBaseModel):
    """Request for engine execution."""

    # Inherits strict configuration from StrictBaseModel

    task_description: str = Field(description="Description of task to execute")
    execution_mode: ExecutionMode | None = Field(
        default=None, description="Override execution mode"
    )
    max_cycles: int | None = Field(default=None, description="Override max cycles")
    context: ExecutionContext = Field(
        default_factory=ExecutionContext, description="Typed execution context"
    )


class ExecutionStep(StrictBaseModel):
    """Information about a single execution step."""

    # Inherits strict configuration from StrictBaseModel

    step_id: str = Field(description="Unique step identifier")
    flow_name: str = Field(description="Name of flow executed")
    inputs: FlowInput = Field(
        default_factory=lambda: FlowInput(task_description=""),
        description="Typed inputs to the flow",
    )
    outputs: FlowOutput = Field(
        default_factory=lambda: FlowOutput(success=False), description="Typed outputs from the flow"
    )
    status: ExecutionStatus = Field(description="Step execution status")
    start_time: datetime | None = Field(default=None, description="Step start time")
    end_time: datetime | None = Field(default=None, description="Step end time")
    duration: float | None = Field(default=None, description="Step duration in seconds")
    error: str | None = Field(default=None, description="Error message if step failed")


class EngineStats(StrictBaseModel):
    """Engine performance statistics."""

    # Inherits strict configuration from StrictBaseModel

    total_executions: int = Field(default=0, description="Total number of executions")
    successful_executions: int = Field(default=0, description="Number of successful executions")
    failed_executions: int = Field(default=0, description="Number of failed executions")
    average_duration: float = Field(default=0.0, description="Average execution duration")
    average_cycles: float = Field(default=0.0, description="Average cycles per execution")
    last_execution: datetime | None = Field(default=None, description="Last execution timestamp")
