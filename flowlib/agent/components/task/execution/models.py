"""Core models for the agent tool system.

This module provides the fundamental base models that all tools must use.
Following flowlib's single source of truth principle - only generic models here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from flowlib.core.models import MutableStrictBaseModel, StrictBaseModel

# Import from parent task models
from ..core.context import RequestContext
from ..core.todo import TodoItem

# TaskThought removed - no longer used in Plan-Execute-Evaluate architecture


class ToolStatus(str, Enum):
    """Tool execution status."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class PresentationMode(str, Enum):
    """How tool output should be presented to users."""
    DIRECT = "direct"           # Use tool output directly without LLM processing
    TRANSFORM = "transform"     # Apply LLM presentation layer to transform output
    HYBRID = "hybrid"          # Tool provides guidance for presentation layer


class ToolPresentationPreference(StrictBaseModel):
    """Tool's declaration of how its output should be presented."""

    mode: PresentationMode = Field(description="Presentation mode for this tool")
    reason: str = Field(description="Why this tool uses this presentation mode")

    class Config:
        extra = "forbid"


# Removed SecurityLevel and PermissionLevel enums - using simple string-based system instead


# Hardcoded mappings removed - these are now managed through the resource registry
# Use RoleConfigResource and ToolCategoryConfigResource for dynamic configuration


# --- TODO Execution Models (moved from task/models.py) ---


class TodoExecutionContext(StrictBaseModel):
    """Structured execution context for TODO items.

    Used in Plan-Execute-Evaluate architecture for tool execution.
    """

    tool_name: str = Field(description="Tool assigned for execution")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool-specific parameters")
    timeout_seconds: Optional[int] = Field(default=None, description="Execution timeout in seconds")
    retry_policy: Literal["none", "simple", "exponential"] = Field(
        default="simple", description="Retry policy for failed execution"
    )
    dependencies_met: bool = Field(default=True, description="Whether all dependencies are satisfied")
    working_directory: Optional[str] = Field(default=None, description="Working directory for execution")
    environment_variables: Dict[str, str] = Field(
        default_factory=dict, description="Environment variables for execution"
    )
    execution_priority: Literal["low", "normal", "high", "urgent"] = Field(
        default="normal", description="Execution priority level"
    )
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional string metadata")


class TodoExecutionResult(StrictBaseModel):
    """Structured result from TODO execution."""

    status: Literal["success", "error", "timeout", "cancelled", "partial"] = Field(
        description="Execution result status"
    )
    output_message: str = Field(description="Primary output or result message")
    files_created: List[str] = Field(default_factory=list, description="Files created during execution")
    files_modified: List[str] = Field(default_factory=list, description="Files modified during execution")
    execution_time_ms: float = Field(description="Execution time in milliseconds")
    exit_code: Optional[int] = Field(default=None, description="Exit code if applicable")

    # Tool-specific outputs
    tool_output: Optional[str] = Field(default=None, description="Raw tool output")
    tool_errors: Optional[str] = Field(default=None, description="Tool error messages")

    # Performance metrics
    memory_used_mb: Optional[float] = Field(default=None, description="Memory usage in MB")
    cpu_time_ms: Optional[float] = Field(default=None, description="CPU time in milliseconds")

    # Recovery and debugging info
    recovery_actions: List[str] = Field(
        default_factory=list, description="Suggested actions if execution failed"
    )
    debug_info: Dict[str, str] = Field(
        default_factory=dict, description="Debug information for troubleshooting"
    )

    # Preserve escape hatch for complex tool-specific data
    tool_specific_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Tool-specific data that doesn't fit typed fields"
    )




class ToolParameters(StrictBaseModel):
    """Base class for all tool parameters.
    
    All tool-specific parameter classes must inherit from this.
    Provides strict validation and forbids extra fields.
    """

    class Config:
        extra = "forbid"  # Strict validation - no unknown fields allowed


class ToolResult(StrictBaseModel):
    """Base class for all tool results.
    
    All tool-specific result classes must inherit from this.
    Provides common result fields and protocol methods.
    """

    status: ToolStatus = Field(description="Execution status")
    message: Optional[str] = Field(default=None, description="Human-readable result message")
    execution_time_ms: Optional[float] = Field(default=None, description="Execution time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the result was generated")

    def get_display_content(self) -> str:
        """Get displayable content from this result.
        
        This is the protocol method that replaces isinstance checks.
        Tool-specific result classes should override this method.
        
        Returns:
            String representation of the result content
        """
        return self.message or ""

    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ToolStatus.SUCCESS

    def is_error(self) -> bool:
        """Check if execution failed."""
        return self.status == ToolStatus.ERROR


# --- New Typed Models for Tool Execution Context ---

class ToolExecutionSharedData(MutableStrictBaseModel):
    """Shared data between tools in execution chain.

    Replaces the generic Dict[str, Any] shared_data with typed fields.
    Must be mutable so tools can update shared state.
    """

    variables: Dict[str, str] = Field(default_factory=dict, description="String variables shared between tools")
    files_created: List[str] = Field(default_factory=list, description="Files created during execution")
    files_modified: List[str] = Field(default_factory=list, description="Files modified during execution")
    working_directory: Optional[str] = Field(default=None, description="Current working directory")
    execution_state: Literal["initial", "in_progress", "completed", "failed"] = Field(
        default="initial", description="Overall execution state"
    )
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional string metadata")
    domain_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific state for specialized tools (e.g., music composition session)"
    )


class ToolExecutionResult(StrictBaseModel):
    """Structured result from previous tool execution.
    
    Replaces the generic Dict[str, Any] in previous_results with typed fields.
    """

    tool_name: str = Field(description="Name of the tool that produced this result")
    status: ToolStatus = Field(description="Execution status of the tool")
    output: str = Field(description="Primary output/message from the tool")
    files_affected: List[str] = Field(default_factory=list, description="Files created or modified by this tool")
    execution_time_ms: float = Field(description="Tool execution time in milliseconds")
    timestamp: datetime = Field(description="When this result was produced")
    tool_specific_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Tool-specific data that doesn't fit typed fields"
    )


class ToolErrorContext(StrictBaseModel):
    """Structured error context information.
    
    Replaces the generic Dict[str, Any] context in ToolExecutionError with typed fields.
    """

    operation: str = Field(description="The operation that was being performed when error occurred")
    attempted_values: Dict[str, str] = Field(
        default_factory=dict, description="Parameter values that were attempted"
    )
    system_state: Optional[str] = Field(default=None, description="Relevant system state information")
    recovery_suggestions: List[str] = Field(
        default_factory=list, description="Suggested recovery actions"
    )
    related_files: List[str] = Field(default_factory=list, description="Files related to the error")
    environment_info: Dict[str, str] = Field(
        default_factory=dict, description="Relevant environment information"
    )


class ToolExecutionContext(StrictBaseModel):
    """Context information provided to tools during execution.
    
    Contains environment and execution metadata that tools may need.
    Now uses typed models instead of generic Dict[str, Any].
    """

    # Execution environment
    working_directory: str = Field(description="Working directory for tool execution")
    timeout_seconds: Optional[int] = Field(default=None, description="Execution timeout")

    # Agent context
    agent_id: str = Field(..., description="ID of the executing agent")
    agent_persona: str = Field(..., description="Agent's persona/personality")
    agent_role: Optional[str] = Field(default=None, description="Agent's role for permission checking")
    session_id: Optional[str] = Field(default=None, description="Agent session ID")
    task_id: Optional[str] = Field(default=None, description="Current task ID")

    # Original user message context
    original_user_message: Optional[str] = Field(default=None, description="Original user message that started this task")
    conversation_history: List[dict] = Field(default_factory=list, description="Recent conversation history for context")

    # Execution metadata
    execution_id: str = Field(description="Unique execution identifier")
    parent_execution_id: Optional[str] = Field(default=None, description="Parent execution if nested")
    execution_depth: int = Field(default=0, description="Execution nesting depth")

    # Tool chain context - now typed
    previous_results: List[ToolExecutionResult] = Field(
        default_factory=list, description="Results from previous tools"
    )
    shared_data: ToolExecutionSharedData = Field(
        description="Shared data between tools - must be explicitly provided to ensure session persistence"
    )

    # Safety and permissions
    permission_level: str = Field(default="ask", description="Permission level (ask/allow/deny)")
    safety_checks_enabled: bool = Field(default=True, description="Whether safety checks are enabled")


class ToolMetadata(StrictBaseModel):
    """Metadata about a tool for registration and discovery."""

    name: str = Field(description="Tool name")
    description: str = Field(description="Full tool description for documentation")
    planning_description: Optional[str] = Field(default=None, description="Concise description for planning prompts (defaults to first sentence of description)")
    tool_category: str = Field(default="generic", description="Tool category (generic, software, systems, etc.)")
    version: str = Field(default="1.0.0", description="Tool version")

    # Discovery
    aliases: List[str] = Field(default_factory=list, description="Alternative tool names")
    tags: List[str] = Field(default_factory=list, description="Tool tags for discovery")

    # Simple access control
    allowed_roles: List[str] = Field(default_factory=list, description="Agent roles allowed to use this tool (if empty, uses category-based access)")
    denied_roles: List[str] = Field(default_factory=list, description="Agent roles explicitly denied access")

    # Capabilities
    max_execution_time: Optional[int] = Field(default=None, description="Maximum execution time in seconds")
    requires_confirmation: bool = Field(default=False, description="Whether tool requires user confirmation before execution")


class ToolExecutionError(StrictBaseModel):
    """Structured error information from tool execution."""

    error_type: str = Field(description="Type of error that occurred")
    error_message: str = Field(description="Detailed error message")
    error_code: Optional[str] = Field(default=None, description="Tool-specific error code")
    context: ToolErrorContext = Field(default_factory=lambda: ToolErrorContext(operation="unknown"), description="Structured error context")
    recoverable: bool = Field(default=False, description="Whether the error can be recovered from")


class ToolPresentationInterface(ABC):
    """Interface that all tools must implement to declare presentation preferences.

    Following Flowlib's single source of truth principle - tools know best
    how their output should be presented to users.
    """

    @abstractmethod
    def get_presentation_preference(self) -> ToolPresentationPreference:
        """Get this tool's presentation preference.

        Returns:
            ToolPresentationPreference defining how this tool's output should be presented

        Raises:
            NotImplementedError: If tool hasn't implemented this method
        """
        raise NotImplementedError("Tools must declare their presentation preferences")


class AgentTaskRequest(StrictBaseModel):
    """Request from agent to execute a task through tools.
    
    This model represents a high-level task request from the agent
    that will be decomposed into TODOs and executed through tools.
    """

    task_description: str = Field(description="The task to be executed")
    context: RequestContext = Field(description="Request context with session, user, and conversation history")
    agent_id: str = Field(description="ID of the agent making the request")


class TaskExecutionResult(StrictBaseModel):
    """Result from executing a task through the tool system.
    
    Aggregates results from all TODO executions and provides
    a unified response to the agent.
    """

    task_description: str = Field(description="Original task that was executed")
    todos_executed: List[TodoItem] = Field(description="List of TodoItems executed")
    tool_results: List[ToolResult] = Field(description="Results from each tool execution")
    final_response: str = Field(description="Aggregated response for the user")
    success: bool = Field(description="Overall success status")
    execution_time_ms: float = Field(description="Total execution time in milliseconds")
    error_summary: Optional[str] = Field(default=None, description="Summary of any errors encountered")


