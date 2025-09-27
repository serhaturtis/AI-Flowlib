"""Agent task decomposer component."""

import logging
from typing import List, Optional, cast, Any
from flowlib.agent.core.base import AgentComponent
from flowlib.agent.components.task.models import TodoItem
from flowlib.agent.core.context.models import ExecutionContext

logger = logging.getLogger(__name__)


class TaskDecompositionComponent(AgentComponent):
    """Task decomposer that breaks down tasks into TODOs with assigned tools."""
    
    def __init__(self, config: Optional[object] = None, name: str = "task_decomposer", activity_stream: Optional[Any] = None) -> None:
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
    
    async def decompose_task(self, context: ExecutionContext, thinking_insights: Optional[str] = None) -> List[TodoItem]:
        """Decompose a task into TODOs with assigned tools.
        
        Args:
            context: Unified execution context containing all necessary information
            
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
        
        # Extract task description from context
        task_description = context.task.description
        
        # Create RequestContext for backward compatibility with existing flow
        request_context = RequestContext(
            session_id=context.session.session_id,
            user_id=context.session.user_id,
            agent_name=context.session.agent_name,
            agent_role=context.session.agent_role,
            previous_messages=context.session.conversation_history,
            working_directory=context.session.working_directory,
            agent_persona=context.session.agent_persona,
            memory_context=f"cycle_{context.task.cycle}"
        )
        
        # Retrieve relevant memories
        relevant_memories = await self._retrieve_memories(task_description, context)


        # Create input for task decomposition
        flow_input = TaskDecompositionInput(
            task_description=task_description,
            context=request_context,
            thinking_insights=thinking_insights,
            relevant_memories=relevant_memories
        )
        
        # Run the flow
        result = await decomposition_flow.run_pipeline(flow_input)
        
        return cast(List[TodoItem], result.todos)

    async def _retrieve_memories(self, task_description: str, context: ExecutionContext) -> Optional[str]:
        """Retrieve relevant memories for task decomposition."""
        assert self._registry is not None, "Component registry not initialized"
        memory_component = cast(Any, self._registry.get("memory"))
        # Create memory search request
        from flowlib.agent.components.memory.models import MemorySearchRequest
        request = MemorySearchRequest(
            query=task_description,
            context=f"task_decomposition_cycle_{context.task.cycle}",
            limit=5,
            threshold=None,
            sort_by=None,
            search_type="hybrid",
            metadata_filter=None
        )
        if memory_component is None:
            return None
        memory_results = await memory_component.search(request)
        if not memory_results:
            return None

        # Extract content from the first/best result
        best_result = memory_results[0]
        # Get content from the memory item
        if hasattr(best_result.item, 'content'):
            content = best_result.item.content
            return str(content) if content is not None else None
        return None

