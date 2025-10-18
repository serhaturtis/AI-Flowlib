"""Task execution reflection component."""

import logging
import time

from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.errors import ExecutionError
from flowlib.flows.registry import flow_registry

from .models import ReflectionInput, ReflectionOutput

logger = logging.getLogger(__name__)


class ExecutionReflectorComponent(AgentComponent):
    """Reflects on execution results to determine if re-planning is needed.

    Used in the Plan-Execute-Reflect loop to analyze execution failures
    and decide whether to continue, re-plan, or ask for clarification.
    """

    def __init__(self, name: str = "execution_reflector"):
        """Initialize execution reflector component.

        Args:
            name: Component name
        """
        super().__init__(name)

    async def _initialize_impl(self) -> None:
        """Initialize the execution reflector."""
        # Verify flow exists in registry
        flow = flow_registry.get_flow("execution-reflection")
        if not flow:
            raise RuntimeError("ExecutionReflectionFlow not found in registry")

        logger.info("ExecutionReflector initialized")

    async def _shutdown_impl(self) -> None:
        """Shutdown the execution reflector."""
        logger.info("ExecutionReflector shutdown")

    async def reflect_on_execution(
        self,
        original_goal: str,
        plan_steps: list[dict],
        execution_results: list[dict],
        partial_completion: bool
    ) -> ReflectionOutput:
        """Reflect on execution results.

        Args:
            original_goal: The original user request
            plan_steps: The planned steps
            execution_results: Results from each executed step
            partial_completion: Whether some steps succeeded but not all

        Returns:
            ReflectionOutput with analysis and next action decision
        """
        self._check_initialized()

        start_time = time.time()

        try:
            # Create input for reflection flow
            reflection_input = ReflectionInput(
                original_goal=original_goal,
                plan_steps=plan_steps,
                execution_results=execution_results,
                partial_completion=partial_completion
            )

            # Run reflection flow
            reflection_flow = flow_registry.get_flow("execution-reflection")
            if reflection_flow is None:
                raise RuntimeError("ExecutionReflectionFlow not found in registry")

            result = await reflection_flow.run_pipeline(reflection_input)

            processing_time = (time.time() - start_time) * 1000

            logger.info(
                f"Execution reflection: {result.result.next_action} "
                f"(successful: {result.result.execution_successful}, "
                f"issues: {len(result.result.issues_found)}, "
                f"time: {processing_time:.2f}ms)"
            )

            return result

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Execution reflection failed: {e}")

            # Fail fast - no fallbacks
            raise ExecutionError(f"Execution reflection failed: {str(e)}") from e
