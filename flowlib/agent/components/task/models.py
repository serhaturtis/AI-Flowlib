"""Common models for task management system.

This module contains shared models used by all task components (decomposition, execution, debriefing),
eliminating circular dependencies and maintaining single source of truth.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Type, Literal
from enum import Enum
from uuid import uuid4
from pydantic import Field, ConfigDict, BeforeValidator, field_validator
from typing_extensions import Annotated

from flowlib.core.models import StrictBaseModel, MutableStrictBaseModel


# --- Shared Context Models ---

class RequestContext(StrictBaseModel):
    """Context for task decomposition and execution requests."""
    
    session_id: Optional[str] = Field(default=None, description="Agent session ID")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    agent_name: Optional[str] = Field(default=None, description="Agent name")
    previous_messages: List[Any] = Field(default_factory=list, description="Previous conversation messages")
    working_directory: str = Field(default=".", description="Working directory")
    agent_persona: Optional[str] = Field(default=None, description="Agent's persona/personality")
    memory_context: Optional[str] = Field(default=None, description="Memory context string")


# --- TODO Models ---

class TodoStatus(Enum):
    """Status of a TODO item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    BLOCKED = "blocked"


class TodoPriority(Enum):
    """Priority levels for TODO items."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


def validate_todo_status(value) -> TodoStatus:
    """Convert string to TodoStatus enum."""
    if isinstance(value, TodoStatus):
        return value
    if isinstance(value, str):
        try:
            return TodoStatus(value.lower())
        except ValueError:
            return TodoStatus.PENDING
    return TodoStatus.PENDING


def validate_todo_priority(value) -> TodoPriority:
    """Convert string to TodoPriority enum."""
    if isinstance(value, TodoPriority):
        return value
    if isinstance(value, str):
        try:
            return TodoPriority(value.lower())
        except ValueError:
            return TodoPriority.MEDIUM
    return TodoPriority.MEDIUM


class TodoStatusSummary(StrictBaseModel):
    """Status summary for TODO items."""
    
    total: int = Field(0, ge=0, description="Total number of TODO items")
    pending: int = Field(0, ge=0, description="Number of pending items")
    in_progress: int = Field(0, ge=0, description="Number of in-progress items")
    completed: int = Field(0, ge=0, description="Number of completed items")
    cancelled: int = Field(0, ge=0, description="Number of cancelled items")
    failed: int = Field(0, ge=0, description="Number of failed items")
    blocked: int = Field(0, ge=0, description="Number of blocked items")


# --- New Typed Models for TODO Execution ---

class TodoExecutionContext(StrictBaseModel):
    """Structured execution context for TODO items.
    
    Replaces the generic Dict[str, Any] execution_context with typed fields.
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
    """Structured result from TODO execution.
    
    Replaces the generic Dict[str, Any] result with typed fields.
    """
    
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


class TodoItem(StrictBaseModel):
    """Individual TODO item with complete lifecycle management.
    
    This is the unified TODO model that handles both LLM generation
    and runtime management, following flowlib's single source of truth principle.
    """
    
    # Core identification
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier")
    content: str = Field(..., description="Description of the task")
    
    # Status and priority with validation
    status: Annotated[TodoStatus, BeforeValidator(validate_todo_status)] = Field(
        default=TodoStatus.PENDING, 
        description="Current status"
    )
    priority: Annotated[TodoPriority, BeforeValidator(validate_todo_priority)] = Field(
        default=TodoPriority.MEDIUM, 
        description="Task priority"
    )
    
    # Timestamps - system-managed fields
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")
    started_at: Optional[datetime] = Field(default=None, description="When task started")
    completed_at: Optional[datetime] = Field(default=None, description="When task completed")
    
    # Duration tracking
    estimated_duration: Optional[timedelta] = Field(default=None, description="Estimated completion time")
    actual_duration: Optional[timedelta] = Field(default=None, description="Actual completion time")
    
    # Dependencies and relationships
    depends_on: List[str] = Field(default_factory=list, description="TODO IDs this depends on")
    blocks: List[str] = Field(default_factory=list, description="TODO IDs this blocks")
    parent_id: Optional[str] = Field(default=None, description="Parent TODO ID if this is a subtask")
    children: List[str] = Field(default_factory=list, description="Child TODO IDs")
    
    # Execution details - now typed
    assigned_tool: Optional[str] = Field(default=None, description="Tool assigned for execution")
    execution_context: Optional[TodoExecutionContext] = Field(
        default=None, description="Structured context for tool execution"
    )
    result: Optional[TodoExecutionResult] = Field(default=None, description="Structured execution result")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Classification tags")
    notes: List[str] = Field(default_factory=list, description="Additional notes")
    
    def mark_in_progress(self) -> 'TodoItem':
        """Mark TODO as in progress."""
        now = datetime.now()
        return self.model_copy(update={
            'status': TodoStatus.IN_PROGRESS,
            'started_at': now,
            'updated_at': now
        })
    
    def mark_completed(self, result: Optional[TodoExecutionResult] = None) -> 'TodoItem':
        """Mark TODO as completed."""
        now = datetime.now()
        update_data = {
            'status': TodoStatus.COMPLETED,
            'completed_at': now,
            'updated_at': now
        }
        
        if result is not None:
            update_data['result'] = result
        
        if self.started_at:
            update_data['actual_duration'] = now - self.started_at
            
        return self.model_copy(update=update_data)
    
    def mark_failed(self, error: str) -> 'TodoItem':
        """Mark TODO as failed."""
        return self.model_copy(update={
            'status': TodoStatus.FAILED,
            'error_message': error,
            'updated_at': datetime.now()
        })
    
    def mark_blocked(self, reason: Optional[str] = None) -> 'TodoItem':
        """Mark TODO as blocked."""
        update_data = {
            'status': TodoStatus.BLOCKED,
            'updated_at': datetime.now()
        }
        
        if reason:
            update_data['error_message'] = reason
            
        return self.model_copy(update=update_data)
    
    def increment_retry(self) -> 'TodoItem':
        """Increment retry count."""
        return self.model_copy(update={
            'retry_count': self.retry_count + 1,
            'updated_at': datetime.now()
        })
    
    def can_retry(self) -> bool:
        """Check if TODO can be retried."""
        return self.retry_count < self.max_retries
    
    def is_ready_to_execute(self, completed_todos: List[str]) -> bool:
        """Check if TODO is ready for execution based on dependencies."""
        if self.status != TodoStatus.PENDING:
            return False
        
        # Check if all dependencies are completed
        return all(dep_id in completed_todos for dep_id in self.depends_on)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Use mode='json' to properly serialize enums and nested models
        data = self.model_dump(mode='json')
        
        # Convert datetime objects to ISO format
        for field in ['created_at', 'updated_at', 'started_at', 'completed_at']:
            if data.get(field):
                data[field] = data[field].isoformat() if isinstance(data[field], datetime) else data[field]
        
        # Convert timedelta objects from ISO format back to seconds for compatibility
        for field in ['estimated_duration', 'actual_duration']:
            if data.get(field) and isinstance(data[field], str) and data[field].startswith('PT'):
                # Convert ISO 8601 duration back to seconds
                # For simplicity, handle common patterns like PT5M, PT30S, PT1H30M
                duration_str = data[field]
                seconds = 0
                
                if 'H' in duration_str:
                    hours = int(duration_str.split('H')[0].split('T')[1])
                    seconds += hours * 3600
                    duration_str = duration_str.split('H')[1] if 'H' in duration_str else duration_str
                
                if 'M' in duration_str:
                    minutes = int(duration_str.split('M')[0].split('T')[-1])  
                    seconds += minutes * 60
                    duration_str = duration_str.split('M')[1] if 'M' in duration_str else duration_str
                
                if 'S' in duration_str:
                    secs = float(duration_str.split('S')[0].split('T')[-1])
                    seconds += secs
                
                data[field] = seconds
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TodoItem':
        """Create TodoItem from dictionary."""
        # Convert ISO datetime strings back to datetime objects
        for field in ['created_at', 'updated_at', 'started_at', 'completed_at']:
            if data.get(field) and isinstance(data[field], str):
                try:
                    data[field] = datetime.fromisoformat(data[field])
                except ValueError:
                    data[field] = None
        
        # Convert seconds back to timedelta objects
        for field in ['estimated_duration', 'actual_duration']:
            if data.get(field) and isinstance(data[field], (int, float)):
                data[field] = timedelta(seconds=data[field])
        
        # Handle typed execution models - convert dict to typed models
        if data.get('execution_context') and isinstance(data['execution_context'], dict):
            data['execution_context'] = TodoExecutionContext(**data['execution_context'])
        
        if data.get('result') and isinstance(data['result'], dict):
            data['result'] = TodoExecutionResult(**data['result'])
        
        return cls(**data)


class TodoList(MutableStrictBaseModel):
    """Collection of TODO items with operations."""
    
    items: List[TodoItem] = Field(default_factory=list, description="TODO items in this list")
    name: str = Field(default="default", description="List name")
    description: Optional[str] = Field(default=None, description="List description")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    def add_todo(self, todo: TodoItem) -> None:
        """Add a TODO item to the list."""
        self.items.append(todo)
    
    def remove_todo(self, todo_id: str) -> bool:
        """Remove a TODO item by ID."""
        for i, item in enumerate(self.items):
            if item.id == todo_id:
                self.items.pop(i)
                return True
        return False
    
    def get_todo(self, todo_id: str) -> Optional[TodoItem]:
        """Get TODO by ID."""
        for item in self.items:
            if item.id == todo_id:
                return item
        return None
    
    def update_todo(self, todo_id: str, updated_todo: TodoItem) -> bool:
        """Update a TODO item."""
        for i, item in enumerate(self.items):
            if item.id == todo_id:
                self.items[i] = updated_todo
                return True
        return False
    
    def get_by_status(self, status: TodoStatus) -> List[TodoItem]:
        """Get all TODOs with specified status."""
        return [item for item in self.items if item.status == status]
    
    def get_ready_to_execute(self) -> List[TodoItem]:
        """Get TODOs ready for execution (no unmet dependencies)."""
        completed_ids = [item.id for item in self.items if item.status == TodoStatus.COMPLETED]
        return [item for item in self.items if item.is_ready_to_execute(completed_ids)]
    
    def get_status_summary(self) -> TodoStatusSummary:
        """Get status summary for all TODOs."""
        status_counts = {}
        for status in TodoStatus:
            status_counts[status.value] = 0
        
        for item in self.items:
            status_counts[item.status.value] += 1
        
        return TodoStatusSummary(
            total=len(self.items),
            pending=status_counts['pending'],
            in_progress=status_counts['in_progress'],
            completed=status_counts['completed'],
            cancelled=status_counts['cancelled'],
            failed=status_counts['failed'],
            blocked=status_counts['blocked']
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps([item.to_dict() for item in self.items])
    
    @classmethod
    def from_json(cls, json_str: str, name: str = "default") -> 'TodoList':
        """Create TodoList from JSON string."""
        data = json.loads(json_str)
        items = [TodoItem.from_dict(item_dict) for item_dict in data]
        return cls(items=items, name=name)


# --- Plan Models ---

class TodoExecutionReasoning(StrictBaseModel):
    """Human-readable explanation of TODO execution decisions."""
    
    explanation: str = Field(..., description="Text explaining the TODO execution decisions")
    rationale: str = Field(None, description="Rationale for the decisions")
    decision_factors: List[str] = Field(default_factory=list, description="Factors that influenced the decision")


class TodoExecutionPlan(StrictBaseModel):
    """Plan for executing a TODO item."""
    
    selected_flow: str = Field(..., description="Name of the selected flow to execute the TODO")
    reasoning: TodoExecutionReasoning = Field(..., description="Reasoning behind the TODO execution plan")


class TodoExecutionValidation(StrictBaseModel):
    """Result of TODO execution plan validation."""
    
    is_valid: bool = Field(..., description="Whether the TODO execution plan is valid")
    errors: List[str] = Field(default_factory=list, description="List of validation errors if any")