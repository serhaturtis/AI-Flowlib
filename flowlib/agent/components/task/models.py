"""Task models - re-exports from core and execution modules.

This module maintains backward compatibility by re-exporting models
from their proper locations.

NEW CODE SHOULD IMPORT DIRECTLY FROM:
- task.core.todo for TodoItem, TodoStatus, TodoPriority, TodoList
- task.core.context for RequestContext
- task.execution.models for TodoExecutionContext, TodoExecutionResult
"""

from __future__ import annotations

from .core.context import RequestContext

# Re-export core models
from .core.todo import (
    TodoItem,
    TodoList,
    TodoPriority,
    TodoStatus,
    TodoStatusSummary,
    validate_todo_priority,
    validate_todo_status,
)

# Re-export execution models
from .execution.models import (
    TodoExecutionContext,
    TodoExecutionResult,
)

__all__ = [
    # Core todo models
    "TodoItem",
    "TodoStatus",
    "TodoPriority",
    "TodoList",
    "TodoStatusSummary",
    "validate_todo_status",
    "validate_todo_priority",
    # Context models
    "RequestContext",
    # Execution models
    "TodoExecutionContext",
    "TodoExecutionResult",
]
