"""Core models for the agent tool system.

This module provides the fundamental base models that all tools must use.
Following flowlib's single source of truth principle - only generic models here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Literal
from pydantic import Field
from enum import Enum

from flowlib.core.models import StrictBaseModel

# Import from parent task models
from ..models import RequestContext, TodoItem


class ToolStatus(str, Enum):
    """Tool execution status."""
    SUCCESS = "success"
    ERROR = "error" 
    WARNING = "warning"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


# Removed SecurityLevel and PermissionLevel enums - using simple string-based system instead


# Agent role constants - simple strings for flexibility
AGENT_ROLES = {
    "general_purpose": "General purpose agent with basic access",
    "software_engineer": "Software development and coding tasks",
    "devops_engineer": "Infrastructure and deployment tasks", 
    "systems_engineer": "System administration and configuration",
    "security_analyst": "Security analysis and audit tasks",
    "data_engineer": "Data processing and analysis tasks",
    "qa_engineer": "Quality assurance and testing tasks",
    "admin": "Administrative tasks with full access",
    "audit": "Read-only access for compliance and auditing"
}

# Tool categories - simple strings for tool classification
TOOL_CATEGORIES = {
    "generic": "Basic tools available to all roles",
    "software": "Software development tools",
    "systems": "System administration tools", 
    "security": "Security analysis tools",
    "data": "Data processing tools",
    "testing": "Testing and QA tools",
    "admin": "Administrative tools"
}

# Role to tool category mappings - flexible and easy to configure
ROLE_TOOL_ACCESS = {
    "general_purpose": ["generic"],
    "software_engineer": ["generic", "software"],
    "devops_engineer": ["generic", "software", "systems"],
    "systems_engineer": ["generic", "systems"],
    "security_analyst": ["generic", "security"],
    "data_engineer": ["generic", "data"],
    "qa_engineer": ["generic", "software", "testing"],
    "admin": ["generic", "software", "systems", "security", "data", "testing", "admin"],
    "audit": ["generic"]  # Read-only access
}




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

class ToolExecutionSharedData(StrictBaseModel):
    """Shared data between tools in execution chain.
    
    Replaces the generic Dict[str, Any] shared_data with typed fields.
    """
    
    variables: Dict[str, str] = Field(default_factory=dict, description="String variables shared between tools")
    files_created: List[str] = Field(default_factory=list, description="Files created during execution")
    files_modified: List[str] = Field(default_factory=list, description="Files modified during execution")
    working_directory: Optional[str] = Field(default=None, description="Current working directory")
    execution_state: Literal["initial", "in_progress", "completed", "failed"] = Field(
        default="initial", description="Overall execution state"
    )
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional string metadata")


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
    
    # Execution metadata
    execution_id: str = Field(description="Unique execution identifier")
    parent_execution_id: Optional[str] = Field(default=None, description="Parent execution if nested")
    execution_depth: int = Field(default=0, description="Execution nesting depth")
    
    # Tool chain context - now typed
    previous_results: List[ToolExecutionResult] = Field(
        default_factory=list, description="Results from previous tools"
    )
    shared_data: ToolExecutionSharedData = Field(
        default_factory=ToolExecutionSharedData, description="Shared data between tools"
    )
    
    # Safety and permissions
    permission_level: str = Field(default="ask", description="Permission level (ask/allow/deny)")
    safety_checks_enabled: bool = Field(default=True, description="Whether safety checks are enabled")


class ToolMetadata(StrictBaseModel):
    """Metadata about a tool for registration and discovery."""
    
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description for LLM")
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


