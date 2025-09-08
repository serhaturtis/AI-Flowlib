"""Models for task decomposition flow.

This module provides the input/output models for the task decomposition system
that breaks down user requests into actionable TODO items.
"""

from typing import List, Dict, Any, Optional
from pydantic import Field, ConfigDict
from flowlib.core.models import StrictBaseModel
from ..models import TodoPriority, TodoItem, RequestContext


class TaskDecompositionInput(StrictBaseModel):
    """Input for task decomposition flow."""
    
    task_description: str = Field(..., description="Task description to decompose")
    context: Optional[RequestContext] = Field(default=None, description="Request context")


class TaskDecompositionOutput(StrictBaseModel):
    """Output containing decomposed tasks as TODO items."""
    
    todos: List[TodoItem] = Field(..., description="Generated TODO items with dependencies")
    execution_strategy: str = Field(..., description="Recommended execution approach")
    estimated_duration: Optional[str] = Field(default=None, description="Estimated completion time")
    dependency_map: Dict[str, List[str]] = Field(default_factory=dict, description="Map of TODO dependencies")
    reasoning: str = Field(..., description="Explanation of task decomposition decisions")
    
    def get_user_display(self) -> str:
        """Get user-friendly display text for task decomposition results."""
        if not self.todos:
            return "📝 No tasks generated for this request."
        
        todo_count = len(self.todos)
        
        # Count dependencies
        dependent_todos = sum(1 for todo in self.todos if todo.depends_on)
        
        # Count by priority
        priority_counts = {}
        for todo in self.todos:
            priority = todo.priority.value.title()
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        summary_parts = []
        summary_parts.append(f"📝 Generated {todo_count} task{'s' if todo_count != 1 else ''}")
        
        if dependent_todos > 0:
            summary_parts.append(f" with {dependent_todos} dependencies")
        
        # Add priority breakdown if varied
        if len(priority_counts) > 1:
            priority_summary = ", ".join([f"{count} {priority.lower()}" for priority, count in priority_counts.items()])
            summary_parts.append(f"\nPriorities: {priority_summary}")
        
        if self.estimated_duration:
            summary_parts.append(f"\nEstimated duration: {self.estimated_duration}")
        
        summary_parts.append(f"\n\n**Execution Strategy:**\n{self.execution_strategy}")
        
        if self.reasoning:
            summary_parts.append(f"\n\n**Reasoning:**\n{self.reasoning}")
        
        return "".join(summary_parts)