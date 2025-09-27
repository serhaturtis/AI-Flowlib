"""Pure task execution flow without decomposition dependencies.

This module provides clean task execution that only handles tool orchestration,
without any task decomposition or TODO generation logic.
"""

import time
import logging
from typing import List
from pydantic import Field

from flowlib.core.models import StrictBaseModel
from flowlib.flows.decorators.decorators import flow, pipeline
from . import ToolStatus, ToolResult, ToolExecutionContext
from .models import TaskExecutionResult
from ..models import TodoItem

logger = logging.getLogger(__name__)


class TodoBatchExecutionInput(StrictBaseModel):
    """Input model for TODO batch execution pipeline."""
    
    todos: List[TodoItem] = Field(..., description="List of TODO items")
    context: ToolExecutionContext = Field(..., description="Execution context")




@flow(name="task-execution", description="Pure task execution without decomposition")  # type: ignore[arg-type]
class TaskExecutionFlow:
    """Pure task execution flow that only handles tool orchestration.
    
    This flow is responsible for:
    1. Direct task execution for simple tasks
    2. Executing pre-decomposed TODO batches
    3. Tool result aggregation
    
    It does NOT handle:
    - Task decomposition
    - TODO generation 
    - Planning
    """
    
    
    @pipeline(input_model=TodoBatchExecutionInput, output_model=TaskExecutionResult)
    async def execute_todo_batch(
        self,
        request: TodoBatchExecutionInput
    ) -> TaskExecutionResult:
        """Execute a batch of pre-decomposed TODOs.
        
        Args:
            request: TODO batch execution request
            
        Returns:
            Aggregated execution result
        """
        start_time = time.time()
        
        # Extract todos and context from request
        todos = request.todos
        context = request.context
        
        results = []
        completed_todos: set[str] = set()
        executed_todos: list[TodoItem] = []
        
        # Execute TODOs respecting dependencies
        while len(completed_todos) < len(todos):
            executable_todos = []
            for todo in todos:
                if todo.id not in completed_todos:
                    # Check if dependencies are satisfied
                    if all(dep_id in completed_todos for dep_id in getattr(todo, 'depends_on', [])):
                        executable_todos.append(todo)
            
            if not executable_todos:
                logger.warning("No executable TODOs found, breaking execution loop")
                break
            
            # Execute the batch of TODOs using orchestrator
            for todo in executable_todos:
                try:
                    # DEBUG: Print TODO being executed
                    print("🚀 DEBUG: Executing TODO:")
                    print(f"     ID: {todo.id}")
                    print(f"     Content: {todo.content}")
                    print(f"     Tool: {todo.assigned_tool}")
                    print()
                    
                    # Execute todo directly using tool registry
                    from .registry import tool_registry
                    result = await tool_registry.execute_todo(todo, context)
                    results.append(result)
                    completed_todos.add(todo.id)

                    # Add the actual todo to executed list
                    executed_todos.append(todo)
                    
                    logger.info(f"Executed TODO {todo.id}: {result.status}")
                    
                except Exception as e:
                    logger.error(f"Failed to execute TODO {todo.id}: {e}")
                    error_result = ToolResult(
                        status=ToolStatus.ERROR,
                        message=f"Failed to execute: {str(e)}"
                    )
                    results.append(error_result)
                    completed_todos.add(todo.id)
        
        # Aggregate results
        success = all(r.status == ToolStatus.SUCCESS for r in results)
        errors = [r.message for r in results if r.status == ToolStatus.ERROR and r.message]
        
        final_response = await self._generate_final_response(results)
        
        # DEBUG: Print final response details
        print("📝 DEBUG: Final Response Generated:")
        print(f"     Total Results: {len(results)}")
        print(f"     Response: {final_response}")
        print()
        
        return TaskExecutionResult(
            task_description=f"Batch execution of {len(todos)} tasks",
            todos_executed=executed_todos,
            tool_results=results,
            final_response=final_response,
            success=success,
            execution_time_ms=(time.time() - start_time) * 1000,
            error_summary="; ".join(errors) if errors else None
        )
    
    
    
    async def _generate_final_response(self, results: List[ToolResult]) -> str:
        """Generate user-facing response from execution results."""
        if not results:
            return "No tasks were executed."
        
        success_messages = []
        error_messages = []
        
        for result in results:
            if result.status == ToolStatus.SUCCESS:
                content = result.get_display_content()
                if content:
                    success_messages.append(content)
            elif result.status == ToolStatus.ERROR:
                content = result.get_display_content()
                if content:
                    error_messages.append(content)
        
        response_parts = []
        if success_messages:
            response_parts.extend(success_messages)
        if error_messages:
            if response_parts:
                response_parts.append("")
            response_parts.append("Errors encountered:")
            response_parts.extend(error_messages)
        
        return "\n".join(response_parts) if response_parts else "Task execution completed."