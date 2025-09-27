"""Task thinking component for strategic task analysis.

This component provides deep strategic analysis of tasks before decomposition,
enabling more intelligent planning and higher success rates.
"""

import logging
import time
from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.errors import ExecutionError
from flowlib.flows.registry import flow_registry
from .models import TaskThinkingInput, TaskThinkingOutput
from flowlib.agent.core.context.models import ExecutionContext
from ..generation.models import GeneratedTask

logger = logging.getLogger(__name__)


class TaskThinkingComponent(AgentComponent):
    """Provides strategic analysis and reasoning about tasks before decomposition."""

    def __init__(self, name: str = "task_thinking"):
        """Initialize task thinking component.

        Args:
            name: Component name
        """
        super().__init__(name)

    async def _initialize_impl(self) -> None:
        """Initialize the task thinking component."""
        # Verify flow exists in registry
        flow = flow_registry.get_flow("task-thinking")
        if not flow:
            raise RuntimeError("TaskThinkingFlow not found in registry")

        logger.info("TaskThinking component initialized")

    async def _shutdown_impl(self) -> None:
        """Shutdown the task thinking component."""
        logger.info("TaskThinking component shutdown")

    async def analyze_task_strategically(
        self,
        context: ExecutionContext,
        generated_task: GeneratedTask
    ) -> TaskThinkingOutput:
        """Analyze task strategically and create comprehensive execution plan.

        Args:
            context: Unified execution context containing all necessary information
            generated_task: Task from the generation component

        Returns:
            TaskThinkingOutput with strategic analysis and enhanced task description
        """
        self._check_initialized()

        start_time = time.time()

        try:
            # Get available tools for this agent role using role-based filtering
            available_tools = self._get_available_tools_for_role(context.session.agent_role)

            # Create RequestContext for the thinking flow
            from ..models import RequestContext
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

            # Create input for task thinking flow
            thinking_input = TaskThinkingInput(
                generated_task=generated_task,
                context=request_context,
                available_tools=available_tools
            )

            # Run the thinking flow
            thinking_flow = flow_registry.get_flow("task-thinking")
            if thinking_flow is None:
                raise RuntimeError("TaskThinkingFlow not found in registry")
            result = await thinking_flow.run_pipeline(thinking_input)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            # Update processing time in result
            return TaskThinkingOutput(
                thinking_result=result.thinking_result,
                enhanced_task_description=result.enhanced_task_description,
                success=True,
                processing_time_ms=processing_time,
                llm_calls_made=result.llm_calls_made
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Task thinking failed: {e}")

            # Fail fast - no fallbacks allowed in flowlib
            raise ExecutionError(f"Task thinking failed: {str(e)}") from e

    def _get_available_tools_for_role(self, agent_role: str) -> list[str]:
        """Get tools available for the agent's role.

        Args:
            agent_role: Agent role string

        Returns:
            List of tool names available to this role
        """
        # Import here to avoid circular imports
        from flowlib.agent.components.task.execution.tool_role_manager import tool_role_manager

        return tool_role_manager.get_allowed_tools(agent_role)