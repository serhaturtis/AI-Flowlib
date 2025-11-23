"""Task models - re-exports from core and execution modules.

This module provides a unified import point for task-related models.
It re-exports models from their source modules for convenience.

Models are available via:
- task.models (this module) - recommended for most use cases
- task.core.todo - direct import for TodoItem, TodoStatus, TodoPriority, TodoList
- task.core.context - direct import for RequestContext
- task.execution.models - direct import for TodoExecutionContext, TodoExecutionResult
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
