"""Task management system for AI agents.

This module provides a complete task management system including:
- Task generation and classification  
- Task decomposition and planning
- Task execution and tool orchestration
- Common models shared between components

Architecture:
- task/models.py: Shared models (TodoItem, Plan, RequestContext, etc.)
- task/generation/: Task classification and enrichment logic
- task/decomposition/: Task decomposition and planning logic
- task/execution/: Task execution and tool orchestration

Following flowlib principles:
- No circular dependencies (common models at parent level)
- Single source of truth for shared types
- Clean separation of concerns
"""

from .models import (
    # Context models
    RequestContext,
    
    # TODO models
    TodoItem,
    TodoList,
    TodoStatus,
    TodoPriority,
    TodoStatusSummary,
    
    
    # TODO execution models
    TodoExecutionPlan,
    TodoExecutionReasoning,
    TodoExecutionValidation,
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
    "TodoExecutionPlan",
    "TodoExecutionReasoning", 
    "TodoExecutionValidation",
]