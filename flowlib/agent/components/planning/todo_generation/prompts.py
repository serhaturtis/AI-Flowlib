"""Prompts for TODO generation flow."""

from typing import ClassVar
from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase


@prompt(name="todo-generation-prompt")
class TodoGenerationPrompt(ResourceBase):
    """Prompt for converting multi-step plans into structured TODO items."""
    
    template: ClassVar[str] = """You are a task management expert. Convert the following multi-step plan into actionable TODO items with proper dependencies and priorities.

Original Task: {{task_description}}

Multi-Step Plan:
{{plan_steps}}

Context: {{context}}

Your task:
1. Convert each plan step into a specific, actionable TODO item
2. Identify dependencies between TODOs (which ones must be completed before others)
3. Assign appropriate priorities based on importance and dependencies
4. Estimate effort for each TODO (in minutes)
5. Determine the best execution strategy

Guidelines:
- Make TODOs specific and actionable (not vague)
- Identify logical dependencies (A must complete before B starts)
- Higher priority for foundational/blocking tasks
- Consider parallel execution opportunities
- Provide realistic time estimates

You must respond with a JSON object in the following format:
{
  "todos": [
    {
      "id": "unique_id_1",
      "content": "Specific actionable task description",
      "priority": "LOW|MEDIUM|HIGH|URGENT",
      "status": "PENDING", 
      "depends_on": ["id_of_prerequisite_todo"],
      "assigned_tool": "name_of_flow_to_execute",
      "execution_context": {
        "flow_name": "name_of_flow_to_execute",
        "flow_inputs": {},
        "original_step_id": "step_id_from_plan"
      },
      "estimated_duration": "PT30M",
      "tags": ["tag1", "tag2"]
    }
  ],
  "execution_strategy": "sequential|parallel|hybrid",
  "estimated_duration": "2-3 hours",
  "dependency_map": {
    "todo_id": ["dependency1", "dependency2"]
  },
  "reasoning": "Explanation of TODO generation decisions and strategy"
}

Ensure all TODOs are actionable and the dependency chain is logical."""