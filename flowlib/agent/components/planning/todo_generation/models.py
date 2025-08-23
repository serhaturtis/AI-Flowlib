"""Models for TODO generation flow."""

from typing import List, Dict, Any, Optional
from pydantic import Field
from flowlib.core.models import StrictBaseModel
from ..models import Plan
from ..todo import TodoItem, TodoPriority


class TodoGenerationInput(StrictBaseModel):
    """Input for TODO generation flow."""
    
    plan: Plan = Field(..., description="Multi-step plan to convert to TODOs")
    task_description: str = Field(..., description="Original task description")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    default_priority: TodoPriority = Field(default=TodoPriority.MEDIUM, description="Default priority for generated TODOs")


class TodoGenerationOutput(StrictBaseModel):
    """Output containing generated TODOs."""
    
    todos: List[TodoItem] = Field(..., description="Generated TODO items with dependencies")
    execution_strategy: str = Field(..., description="Recommended execution approach")
    estimated_duration: Optional[str] = Field(None, description="Estimated completion time")
    dependency_map: Dict[str, List[str]] = Field(default_factory=dict, description="Map of TODO dependencies")
    reasoning: str = Field(..., description="Explanation of TODO generation decisions")
    
    def get_user_display(self) -> str:
        """Get user-friendly display text for TODO generation results."""
        if not self.todos:
            return "📝 No TODOs generated for this task."
        
        todo_count = len(self.todos)
        
        # Count dependencies
        dependent_todos = sum(1 for todo in self.todos if hasattr(todo, 'depends_on') and todo.depends_on)
        
        # Count by priority
        priority_counts = {}
        for todo in self.todos:
            priority = str(todo.priority).replace('TodoPriority.', '').title()
            if priority in priority_counts:
                priority_counts[priority] += 1
            else:
                priority_counts[priority] = 1
        
        summary_parts = []
        summary_parts.append(f"📝 Generated {todo_count} TODO{'s' if todo_count != 1 else ''}")
        
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