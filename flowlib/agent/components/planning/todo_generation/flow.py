"""TODO generation flow implementation."""

import uuid
from typing import Dict, List, Any
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.providers.core.registry import provider_registry
# Removed ProviderType import - using config-driven provider access
from flowlib.resources.registry.registry import resource_registry
from flowlib.resources.models.constants import ResourceType
from .models import TodoGenerationInput, TodoGenerationOutput
from ..todo import TodoItem, TodoPriority, TodoStatus


@flow(
    name="todo-generation",
    description="Convert multi-step plans into structured TODO items with dependencies",
    is_infrastructure=False
)
class TodoGenerationFlow:
    """Converts planning results into actionable TODO items with dependencies."""
    
    @pipeline(input_model=TodoGenerationInput, output_model=TodoGenerationOutput)
    async def run_pipeline(self, input_data: TodoGenerationInput) -> TodoGenerationOutput:
        """Generate TODOs from plan steps with intelligent dependency mapping.
        
        Args:
            input_data: Contains the plan to convert and task context
            
        Returns:
            Generated TODO items with dependencies and execution strategy
        """
        # Get LLM provider using config-driven approach
        llm = await provider_registry.get_by_config("default-llm")
        
        # Get prompt from registry
        prompt_instance = resource_registry.get("todo-generation-prompt")
        
        # Format plan steps for the prompt
        plan_steps_text = self._format_plan_steps(input_data.plan)
        
        # Prepare variables
        prompt_vars = {
            "task_description": input_data.task_description,
            "plan_steps": plan_steps_text,
            "context": str(input_data.context)
        }
        
        # Generate TODO structure using LLM
        result = await llm.generate_structured(
            prompt=prompt_instance,
            output_type=TodoGenerationOutput,
            prompt_variables=prompt_vars
        )
        
        # Post-process and validate TODOs
        validated_todos = self._validate_and_enhance_todos(result.todos, input_data)
        
        # Create new result with validated todos since result is frozen
        return result.model_copy(update={"todos": validated_todos})
    
    def _format_plan_steps(self, plan) -> str:
        """Format plan steps into readable text."""
        if not plan.steps:
            return "No specific steps provided."
        
        formatted_steps = []
        for i, step in enumerate(plan.steps, 1):
            step_text = f"Step {i}: {step.step_intent}\n"
            step_text += f"  Flow: {step.flow_name}\n"
            step_text += f"  Rationale: {step.rationale}\n"
            if step.expected_outcome:
                step_text += f"  Expected: {step.expected_outcome}\n"
            formatted_steps.append(step_text)
        
        return "\n".join(formatted_steps)
    
    def _validate_and_enhance_todos(self, todos: List[TodoItem], input_data: TodoGenerationInput) -> List[TodoItem]:
        """Validate and enhance generated TODOs with proper IDs and validation."""
        validated_todos = []
        
        for todo in todos:
            # Prepare update data
            updates = {}
            
            # Ensure unique ID
            if not todo.id or todo.id == "unique_id_1":  # LLM might use placeholder
                updates["id"] = str(uuid.uuid4())
            
            # Validate priority
            if isinstance(todo.priority, str):
                try:
                    priority = TodoPriority(todo.priority.lower())  # TodoPriority uses lowercase
                    updates["priority"] = priority
                except ValueError:
                    updates["priority"] = input_data.default_priority
            
            # Ensure status is pending
            updates["status"] = TodoStatus.PENDING
            
            # Validate dependencies - map to depends_on field in existing model
            if hasattr(todo, 'dependencies'):
                # Map dependencies to depends_on field
                valid_todo_ids = {t.id for t in todos}
                depends_on = [dep for dep in getattr(todo, 'dependencies', []) if dep in valid_todo_ids]
                updates["depends_on"] = depends_on
            
            # Create new TodoItem with updates if any, otherwise use original
            if updates:
                validated_todo = todo.model_copy(update=updates)
            else:
                validated_todo = todo
                
            validated_todos.append(validated_todo)
        
        return validated_todos