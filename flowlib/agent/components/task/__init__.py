"""Task management system for AI agents.

This module provides a complete task management system following the
Plan-Execute-Evaluate architecture optimized for local LLMs.

Architecture:
- task/models.py: Shared models (TodoItem, RequestContext, etc.)
- task/core/: Core task components (TodoManager)
- task/planning/: Structured planning (StructuredPlannerComponent)
- task/execution/: Task execution and tool orchestration
- task/evaluation/: Task completion evaluation

Following flowlib principles:
- No circular dependencies (common models at parent level)
- Single source of truth for shared types
- Clean separation of concerns
- Fail fast with no fallbacks
"""

from .models import (
    # Context models
    RequestContext,
    # TODO execution models
    TodoExecutionContext,
    TodoExecutionResult,
    # TODO models
    TodoItem,
    TodoList,
    TodoPriority,
    TodoStatus,
    TodoStatusSummary,
)

__all__ = [
    # Context
    "RequestContext",

    # TODO system
    "TodoItem",
    "TodoList",
    "TodoStatus",
    "TodoPriority",
    "TodoStatusSummary",

    # TODO execution system
    "TodoExecutionContext",
    "TodoExecutionResult",
]
