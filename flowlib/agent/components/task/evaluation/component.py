"""Task completion evaluation component."""

import logging
import time

from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.errors import ExecutionError
from flowlib.flows.registry import flow_registry

from .models import EvaluationInput, EvaluationOutput

logger = logging.getLogger(__name__)


class CompletionEvaluatorComponent(AgentComponent):
    """Evaluates whether a task has been completed successfully.

    Used in the Plan-and-Execute loop to determine if more cycles are needed.
    """

    def __init__(self, name: str = "completion_evaluator"):
        """Initialize completion evaluator component.

        Args:
            name: Component name
        """
        super().__init__(name)

    async def _initialize_impl(self) -> None:
        """Initialize the completion evaluator."""
        # Verify flow exists in registry
        flow = flow_registry.get_flow("completion-evaluation")
        if not flow:
            raise RuntimeError("CompletionEvaluationFlow not found in registry")

        logger.info("CompletionEvaluator initialized")

    async def _shutdown_impl(self) -> None:
        """Shutdown the completion evaluator."""
        logger.info("CompletionEvaluator shutdown")

    async def evaluate_completion(
        self,
        original_goal: str,
        plan_reasoning: str,
        executed_steps: list[dict],
        expected_outcome: str
    ) -> EvaluationOutput:
        """Evaluate whether the task is complete.

        Args:
            original_goal: The original user request
            plan_reasoning: Reasoning from the execution plan
            executed_steps: Steps that were executed with results
            expected_outcome: Expected outcome from the plan

        Returns:
            EvaluationOutput with completion decision
        """
        self._check_initialized()

        start_time = time.time()

        try:
            # Create input for evaluation flow
            eval_input = EvaluationInput(
                original_goal=original_goal,
                plan_reasoning=plan_reasoning,
                executed_steps=executed_steps,
                expected_outcome=expected_outcome
            )

            # Run evaluation flow
            eval_flow = flow_registry.get_flow("completion-evaluation")
            if eval_flow is None:
                raise RuntimeError("CompletionEvaluationFlow not found in registry")

            result = await eval_flow.run_pipeline(eval_input)

            processing_time = (time.time() - start_time) * 1000

            logger.info(
                f"Task evaluation: {result.result.next_action} "
                f"(confidence: {result.result.completion_confidence:.2f}, "
                f"time: {processing_time:.2f}ms)"
            )

            return result

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Completion evaluation failed: {e}")

            # Fail fast - no fallbacks
            raise ExecutionError(f"Completion evaluation failed: {str(e)}") from e
