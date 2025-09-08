"""Task execution coordinator without decomposition bypass.

This module provides pure coordination for executing pre-decomposed TODOs,
following the architectural principle that ALL tasks go through decomposition first.
"""

import logging
from typing import List

from flowlib.agent.core.base import AgentComponent
from .models import TaskExecutionResult, ToolExecutionContext
from ..models import TodoItem
from flowlib.flows.registry import flow_registry

logger = logging.getLogger(__name__)


class TaskExecutionComponent(AgentComponent):
    """Coordinates execution of pre-decomposed TODOs.
    
    This is NOT a flow - it's a coordinator/service class.
    It ONLY handles execution coordination.
    It does NOT perform task decomposition or classification.
    ALL tasks must go through decomposition first.
    """
    
    def __init__(self, name: str = "task_executor", activity_stream=None):
        """Initialize the task executor.
        
        Args:
            name: Component name
            activity_stream: Optional activity stream for real-time updates
        """
        super().__init__(name)
        self._activity_stream = activity_stream
    
    async def _initialize_impl(self) -> None:
        """Initialize the task executor."""
        logger.info("Task executor initialized")
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the task executor."""
        logger.info("Task executor shutdown")
    
    async def execute_decomposed_todos(
        self,
        todos: List[TodoItem],
        context: ToolExecutionContext
    ) -> TaskExecutionResult:
        """Execute pre-decomposed TODOs through execution flow.
        
        Args:
            todos: Pre-generated TODO items from decomposition system
            context: Execution context
            
        Returns:
            Aggregated execution result
        """
        self._check_initialized()
        
        # Get execution flow from registry and create proper input
        from .execution_flow import TodoBatchExecutionInput
        execution_flow = flow_registry.get("task-execution")
        
        # Create input model for the flow
        flow_input = TodoBatchExecutionInput(todos=todos, context=context)
        return await execution_flow.execute_todo_batch(flow_input)
    
