"""Tool calling models following flowlib's strict patterns.

This module defines the data models used for structured tool calling,
with strict validation and no fallbacks following CLAUDE.md principles.
"""

from typing import Any, Dict, List, Optional, Literal
from datetime import datetime
from pydantic import Field

from flowlib.core.models import StrictBaseModel


class ToolCall(StrictBaseModel):
    """Structured tool call from LLM with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    tool_name: str = Field(description="Name of the tool to execute")
    parameters: Dict[str, Any] = Field(description="Tool parameters")
    reasoning: Optional[str] = Field(default=None, description="LLM reasoning for tool selection")
    call_id: Optional[str] = Field(default=None, description="Unique identifier for this call")


class ToolResult(StrictBaseModel):
    """Result from tool execution with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    tool_name: str = Field(description="Name of the executed tool")
    status: Literal["success", "error", "warning"] = Field(description="Execution status")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Tool execution result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time_ms: Optional[float] = Field(default=None, description="Execution time in milliseconds")
    call_id: Optional[str] = Field(default=None, description="Matching call identifier")


class ToolCallRequest(StrictBaseModel):
    """Request for agent tool calling with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    task_description: str = Field(description="Description of the task requiring tool execution")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context for tool selection")
    max_tools: int = Field(default=5, description="Maximum number of tools to select")
    working_directory: str = Field(default=".", description="Working directory for tool execution")


class ToolExecutionResult(StrictBaseModel):
    """Complete result from tool calling flow execution."""
    # Inherits strict configuration from StrictBaseModel
    
    request: ToolCallRequest = Field(description="Original request")
    selected_tools: List[ToolCall] = Field(description="Tools selected by LLM")
    results: List[ToolResult] = Field(description="Execution results")
    total_execution_time_ms: float = Field(description="Total execution time")
    success_count: int = Field(description="Number of successful tool executions")
    error_count: int = Field(description="Number of failed tool executions")
    timestamp: datetime = Field(default_factory=datetime.now, description="Execution timestamp")


class ToolExecutionContext(StrictBaseModel):
    """Context provided to tools during execution."""
    # Inherits strict configuration from StrictBaseModel
    
    working_directory: str = Field(description="Working directory for tool execution")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional execution metadata")