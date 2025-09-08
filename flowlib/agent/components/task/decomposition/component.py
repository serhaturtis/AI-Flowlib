"""Agent task decomposer component."""

import logging
from typing import List, Optional, Any
from flowlib.agent.core.base import AgentComponent
from ..models import TodoItem

logger = logging.getLogger(__name__)


class TaskDecompositionComponent(AgentComponent):
    """Task decomposer that breaks down tasks into TODOs with assigned tools."""
    
    def __init__(self, config: Any = None, name: str = "task_decomposer", activity_stream=None):
        """Initialize the task decomposer.
        
        Args:
            config: Configuration (currently unused)
            name: Component name
            activity_stream: Activity stream for logging
        """
        super().__init__(name)
        # Config is passed for compatibility but not used currently
        self._activity_stream = activity_stream
    
    async def _initialize_impl(self) -> None:
        """Initialize the task decomposer."""
        logger.info("Task decomposer initialized")
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the task decomposer."""
        logger.info("Task decomposer shutdown")
    
    async def decompose_task(self, task_description: str, context) -> List[TodoItem]:
        """Decompose a task into TODOs with assigned tools.
        
        Args:
            task_description: The task to decompose
            context: Additional context for decomposition
            
        Returns:
            List of TodoItem with assigned tools
        """
        self._check_initialized()
        
        # Get TaskDecompositionFlow from flow registry
        from flowlib.flows.registry import flow_registry
        from .task_models import TaskDecompositionInput
        from ..models import RequestContext
        
        decomposition_flow = flow_registry.get_flow("task-decomposition")
        if not decomposition_flow:
            raise RuntimeError("TaskDecompositionFlow not found in registry")
        
        # Create input for task decomposition
        flow_input = TaskDecompositionInput(
            task_description=task_description,
            context=context  # Pass the actual context instead of creating empty one
        )
        
        # Run the flow
        result = await decomposition_flow.run_pipeline(flow_input)
        
        return result.todos